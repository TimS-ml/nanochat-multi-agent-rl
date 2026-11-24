# Inference Logic

How nanochat generates text, from prompt to output tokens.

---

## Overview

Nanochat has two inference paths:

1. **Naive generation** (`GPT.generate` in `nanochat/gpt.py:603`) - Simple autoregressive loop without KV cache. Batch size 1 only. Used for quick sampling during training evaluation.

2. **Engine generation** (`Engine.generate` in `nanochat/engine.py:176`) - Production inference with KV cache, batched multi-sample generation, and tool use (calculator). This is the primary inference path.

---

## Engine Generation Pipeline

### Step 1: Prefill (batch=1)

```
tokens = [BOS, user_start, ..., user_end, assistant_start]
                                                         ^
                                           prompt ends here
```

The engine processes the entire prompt in a single forward pass with batch_size=1:

```python
kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
logits = model.forward(ids, kv_cache=kv_cache_prefill)
```

This populates the KV cache with key/value vectors for all prompt tokens. The logits from the last position are used to sample the first generated token.

### Step 2: KV Cache Replication

For multi-sample generation (e.g. `num_samples=16` for RL), the prefill cache is replicated:

```python
kv_cache_decode = KVCache(batch_size=num_samples, ...)
kv_cache_decode.prefill(kv_cache_prefill)  # copy to all rows
```

This avoids redundant prefill computation. Each sample row gets an independent copy of the cached prompt KVs.

### Step 3: Autoregressive Decode Loop

Each iteration:
1. **Sample** next token from logits via `sample_next_token()` (temperature + top-k)
2. **Process** each row independently:
   - Check for forced tokens (from tool output injection)
   - Detect `<|assistant_end|>` or `<|bos|>` -> mark row as completed
   - Handle tool use state machine (see below)
3. **Yield** the token column `(token_column, token_masks)` to the caller
4. **Forward** the new tokens through the model with KV cache to get next logits

```python
while not all_completed and num_generated < max_tokens:
    next_ids = sample_next_token(logits, rng, temperature, top_k)
    # ... process each row ...
    yield token_column, token_masks
    logits = model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
```

### Step 4: Token Masks

The engine tracks which tokens were **sampled** (mask=1) vs **forced** (mask=0):
- Prompt tokens: mask=0
- Model-generated tokens: mask=1
- Tool output tokens (calculator results): mask=0

This distinction is critical for RL training: we only compute policy gradients on sampled tokens, not forced/prompt tokens.

---

## Tool Use: Calculator State Machine

The engine implements a simple tool use protocol using special tokens:

```
<|python_start|> 3.14 * 2 <|python_end|> <|output_start|> 6.28 <|output_end|>
```

### State Machine

```
Normal Mode ──[python_start]──> Python Block Mode
                                    │
                                    │ (accumulate expression tokens)
                                    │
                                    ├──[python_end]──> Evaluate expression
                                    │                      │
                                    │                      ├── Success: force inject
                                    │                      │   output_start + result + output_end
                                    │                      │
                                    │                      └── Failure: continue without output
                                    │
                                    └──> Back to Normal Mode
```

### Per-Row State (`RowState`)

Each row in the batch has independent state:
- `current_tokens`: full token sequence so far
- `forced_tokens`: deque of tokens to inject before sampling
- `in_python_block`: whether currently inside `<|python_start|>...<|python_end|>`
- `python_expr_tokens`: accumulated expression tokens
- `completed`: whether generation has ended

### Calculator Safety

The `use_calculator` function (`engine.py:46`) evaluates expressions with:
- **Timeout**: 3 seconds max
- **Sandboxed**: `eval()` with `__builtins__` stripped
- **Restricted operations**: only math expressions and `.count()` string operations
- **Blocklisted patterns**: no `__`, `import`, `exec`, `open`, etc.
- **No power operator** (`**`) to prevent resource exhaustion

---

## KV Cache Design (`nanochat/engine.py:82`)

The KV cache is designed for Flash Attention 3's `flash_attn_with_kvcache` API:

### Layout
```
k_cache: (n_layers, B, T_max, H_kv, D)
v_cache: (n_layers, B, T_max, H_kv, D)
cache_seqlens: (B,) int32  # current position per batch element
```

Key differences from FA2-style caches:
- Tensors are `(B, T, H, D)` not `(B, H, T, D)`
- FA3 updates cache **in-place** during `flash_attn_with_kvcache`
- Position tracked per batch element via `cache_seqlens`

### Cache dtype

`Engine.generate` allocates the cache in `COMPUTE_DTYPE`, the dtype the forward pass emits: bf16 on CUDA GPUs with compute capability 8.0+, fp32 otherwise (CPU, MPS, pre-Ampere GPUs), overridable with the `NANOCHAT_DTYPE` environment variable (for example `NANOCHAT_DTYPE=bfloat16` on MPS). bf16 halves the cache bytes relative to fp32.

### Cache Flow in Attention (`gpt.py:154-169`)

**Training** (no cache):
```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

**Inference** (with cache):
```python
k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
y = flash_attn.flash_attn_with_kvcache(
    q, k_cache, v_cache, k=k, v=v,
    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
)
# Advance position after the last layer processes
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

---

## Sliding Window Attention

Nanochat uses a configurable sliding window pattern (default: "SSSL"):
- **S (Short)**: attend to ~`sequence_len / 4` tokens back (ceiled to the FA3 128-tile size, e.g. 2048 -> 512)
- **L (Long)**: attend to full `sequence_len` context

The pattern is tiled across layers, with the final layer always getting full context:
```
Layer 0: S (512 tokens)
Layer 1: S (512 tokens)
Layer 2: S (512 tokens)
Layer 3: L (2048 tokens)  <- always full context
Layer 4: S (512 tokens)
...
Layer N: L (2048 tokens)  <- final layer, always full context
```

Window sizes are stored as `(left, right)` tuples for FA3 (`GPT._compute_window_sizes`, kept in `GPT.window_sizes`):
- L layers: `(sequence_len, 0)`, e.g. `(2048, 0)` = the full causal context (FA3 itself also accepts `-1` for "unlimited left")
- S layers: `(512, 0)` = sliding window of 512 tokens

---

## Sampling Strategy

### `sample_next_token` (`engine.py:141`)

```python
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
```

1. **Temperature = 0**: Greedy decoding (`argmax`)
2. **Top-k filtering**: Keep only the top-k logits, set rest to `-inf`
3. **Temperature scaling**: Divide logits by temperature
4. **Softmax + multinomial**: Convert to probabilities and sample

The function operates on batched logits `(B, vocab_size)` and returns `(B, 1)`.

### Logit Processing in Model (`gpt.py:586-591`)

Before sampling, logits go through:
1. **Padding removal**: Slice to actual `vocab_size` (model pads to multiple of 64)
2. **Float32 upcast**: Switch from bfloat16 for numerical precision
3. **Softcap**: `15 * tanh(logits / 15)` - smoothly squashes extreme logits

---

## Generation Modes

### Streaming (`Engine.generate`)
Yields `(token_column, token_masks)` per step. Used by the CLI (`scripts/chat_cli.py`) and by the inference benchmark (`scripts/infer_bench.py`).

```python
for token_column, token_masks in engine.generate(tokens, num_samples=4, ...):
    # token_column: list of 4 ints (one per sample)
    # token_masks: list of 4 ints (1=sampled, 0=forced)
    print(tokenizer.decode([token_column[0]]), end="")
```

### Batch (`Engine.generate_batch`)
Non-streaming. Returns complete token sequences and masks.

```python
results, masks = engine.generate_batch(tokens, num_samples=16, max_tokens=256)
# results: list of 16 token lists (excluding terminal tokens)
# masks: list of 16 mask lists
```

### Naive (`GPT.generate`)
Simple streaming generator without KV cache. Batch=1 only. Re-processes entire sequence each step (quadratic cost). Used only for testing/comparison.

---

## Inference Cost Model and Benchmark

Inference has two regimes. **Prefill** processes the whole prompt in parallel with large matmuls, so it is compute-bound. **Decode** emits one token per step and re-reads all weights and the KV cache to do little math, so it is memory-bandwidth-bound, and batching is nearly free until compute saturates. `GPT` carries the architecture-side accounting for both regimes, and `scripts/infer_bench.py` measures how close a real run comes to the hardware limits.

Notation: $P$ is the number of matmul params (`GPT.num_matmul_params`), $h$ the number of query heads, $q$ the head dimension, $h_{kv}$ the number of KV heads, $L$ the number of layers, $w_\ell$ the window of layer $\ell$ (from `GPT.window_sizes`), and $b$ the bytes per element of `COMPUTE_DTYPE`.

### Architecture-side accounting (`nanochat/gpt.py`)

Attention at one query position over $m$ keys costs $2hqm$ FLOPs for $QK^\top$ plus $2hqm$ for the weighted sum over $V$, i.e. $4hqm$. Each matmul param costs 2 FLOPs per token (one multiply, one add). Layer $\ell$ attends to at most $w_\ell$ tokens, so:

| Method | Quantity | Formula |
|--------|----------|---------|
| `estimate_decode_flops(c)` | Forward FLOPs to decode one token at context length $c$ | $2P + \sum_{\ell=1}^{L} 4hq\,\min(c, w_\ell)$ |
| `estimate_prefill_flops(T)` | Forward FLOPs to prefill $T$ prompt tokens | $2PT + \sum_{\ell=1}^{L} 4hq\,A_\ell$, with $A_\ell = \frac{w(w+1)}{2} + (T-w)\,w$ and $w = \min(w_\ell, T)$ |
| `kv_bytes_per_token()` | KV bytes stored per token per row | $L \cdot 2 \cdot h_{kv} \cdot q \cdot b$ |
| `kv_read_bytes(c)` | KV bytes read by one decode step at context length $c$ | $\sum_{\ell=1}^{L} 2\,h_{kv}\,q\,b\,\min(c, w_\ell)$ |

$A_\ell$ counts the (query, key) pairs of causal attention with a window: the first $w$ tokens attend to $1, 2, \dots, w$ tokens, which sums to $w(w+1)/2$, and each of the remaining $T-w$ tokens attends to exactly $w$ tokens. The factor 2 in the KV formulas is one K and one V tensor.

### Measurement (`scripts/infer_bench.py`)

CUDA only, single GPU: `python -m scripts.infer_bench -i base -g d12`. Let $W$ be the weight bytes as stored (`sum(p.numel() * p.element_size())`), $c_{mid}$ the representative decode context (prompt length plus half the decode tokens), $R$ the KV bytes read per row at that context (`kv_read_bytes`), $n$ the decode batch size, $t$ the median decode step time, $\mathrm{BW}$ the peak memory bandwidth and $F$ the peak FLOP/s of the GPU (`get_peak_bandwidth` and `get_peak_flops` in `nanochat/common.py`; unknown GPUs yield infinity, which shows as 0%).

| Metric | Definition |
|--------|------------|
| TTFT | Time of the first `next()` on `Engine.generate`: batch=1 prefill, KV cache replication to $n$ rows, and sampling of the first token |
| TPOT | $t$, the median time of one decode step for all $n$ rows |
| Throughput | $n \cdot (\text{decode steps}) / \sum(\text{step times})$ tokens/s |
| Batch-1 ceiling | Every step must re-read the weights and the KV cache, so a step takes at least $(W+R)/\mathrm{BW}$ seconds, giving at most $\mathrm{BW}/(W+R)$ tokens/s |
| MBU (model bandwidth utilization) | $\dfrac{(W + nR)/t}{\mathrm{BW}}$, the decode counterpart of training MFU. It binds at small $n$ |
| Decode MFU | $\dfrac{n \cdot \texttt{estimate\_decode\_flops}(c_{mid})/t}{F}$. It binds at large $n$ |
| Prefill MFU | $\dfrac{\texttt{estimate\_prefill\_flops}(T)/t_{prefill}}{F}$, measured at batch 1, with $t_{prefill}$ the TTFT of a generation capped at 2 tokens |

The script prints a static card (weights, KV bytes per token, ceilings), the prefill line, and a table over `--batch-sizes` (default 1,8,32,128). The last line of stdout is the same data as one JSON document, so other scripts can read it with `json.loads(stdout.splitlines()[-1])`.

---

## Conversation Format

Conversations use special tokens:

```
<|bos|>
<|user_start|> What is 2+2? <|user_end|>
<|assistant_start|> Let me calculate: <|python_start|> 2+2 <|python_end|>
<|output_start|> 4 <|output_end|>
The answer is 4. <|assistant_end|>
```

The tokenizer's `render_for_completion` method prepares prompts for generation by including everything up to and including `<|assistant_start|>`, so the model generates the assistant's response.
