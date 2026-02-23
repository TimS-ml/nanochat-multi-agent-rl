# Inference Logic

How nanochat generates text, from prompt to output tokens.

---

## Overview

Nanochat has two inference paths:

1. **Naive generation** (`GPT.generate` in `nanochat/gpt.py:498`) - Simple autoregressive loop without KV cache. Batch size 1 only. Used for quick sampling during training evaluation.

2. **Engine generation** (`Engine.generate` in `nanochat/engine.py:171`) - Production inference with KV cache, batched multi-sample generation, and tool use (calculator). This is the primary inference path.

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

The `use_calculator` function (`engine.py:47`) evaluates expressions with:
- **Timeout**: 3 seconds max
- **Sandboxed**: `eval()` with `__builtins__` stripped
- **Restricted operations**: only math expressions and `.count()` string operations
- **Blocklisted patterns**: no `__`, `import`, `exec`, `open`, etc.
- **No power operator** (`**`) to prevent resource exhaustion

---

## KV Cache Design (`nanochat/engine.py:83`)

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

### Cache Flow in Attention (`gpt.py:142-157`)

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
- **S (Short)**: attend to `sequence_len / 2` tokens back
- **L (Long)**: attend to full `sequence_len` context

The pattern is tiled across layers, with the final layer always getting full context:
```
Layer 0: S (1024 tokens)
Layer 1: S (1024 tokens)
Layer 2: S (1024 tokens)
Layer 3: L (2048 tokens)  <- always full context
Layer 4: S (1024 tokens)
...
Layer N: L (2048 tokens)  <- final layer, always full context
```

Window sizes are stored as `(left, right)` tuples for FA3:
- `(-1, 0)` = full causal context (unlimited left)
- `(1024, 0)` = sliding window of 1024 tokens

---

## Sampling Strategy

### `sample_next_token` (`engine.py:136`)

```python
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
```

1. **Temperature = 0**: Greedy decoding (`argmax`)
2. **Top-k filtering**: Keep only the top-k logits, set rest to `-inf`
3. **Temperature scaling**: Divide logits by temperature
4. **Softmax + multinomial**: Convert to probabilities and sample

The function operates on batched logits `(B, vocab_size)` and returns `(B, 1)`.

### Logit Processing in Model (`gpt.py:483-487`)

Before sampling, logits go through:
1. **Padding removal**: Slice to actual `vocab_size` (model pads to multiple of 64)
2. **Float32 upcast**: Switch from bfloat16 for numerical precision
3. **Softcap**: `15 * tanh(logits / 15)` - smoothly squashes extreme logits

---

## Generation Modes

### Streaming (`Engine.generate`)
Yields `(token_column, token_masks)` per step. Used by CLI and web interfaces.

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
