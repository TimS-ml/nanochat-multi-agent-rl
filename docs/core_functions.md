# Core Functions Reference

A comprehensive reference of all key functions and classes in the nanochat codebase, organized by module.

---

## Model Architecture (`nanochat/gpt.py`)

### GPTConfig (dataclass, line 29)
Model configuration with depth-based scaling. Key fields:
- `sequence_len` (2048): max context length
- `vocab_size` (32768): tokenizer vocabulary size
- `n_layer` (12): number of transformer blocks
- `n_head` / `n_kv_head` (6): query and key/value head counts (GQA support)
- `n_embd` (768): embedding dimension (`depth * aspect_ratio`, where `aspect_ratio=64`)
- `window_pattern` ("SSSL"): sliding window attention pattern tiled across layers

### GPT (class, line 218)
Main GPT model. Notable architectural choices:
- **RoPE** (no learned positional embeddings; rotates by -theta, the transpose of the textbook convention, which is functionally equivalent)
- **QK norm** for training stability
- **Untied** token embedding and lm_head weights
- **ReLU^2** activation in MLP
- **Logit softcapping** at 15 (smooth squashing via tanh)
- **Learnable per-layer scalars**: `resid_lambdas` (residual scale) and `x0_lambdas` (input embedding skip connection)
- **Smear gate** (`smear_gate` + `smear_lambda`): mixes the previous token's embedding into the current token for cheap bigram-like context
- **Backout** (`backout_lambda`): subtracts a cached mid-layer residual before the final norm to remove low-level features
- **Value Embeddings** (ResFormer-style): alternating layers get a separate value embedding from the input tokens
- **Sliding window attention**: configurable SSSL pattern (3 short + 1 long, repeating)

| Method | Line | Description |
|--------|------|-------------|
| `__init__` | 219 | Builds model on meta device (shapes only, no data) |
| `init_weights` | 267 | Initializes all parameters: uniform for attn/MLP, normal for embeddings, zeros for output projections |
| `_precompute_rotary_embeddings` | 333 | Computes cos/sin tables for RoPE (10X over-allocation for safety) |
| `_compute_window_sizes` | 359 | Converts window pattern string ("SSSL") to per-layer (left, right) window size tuples |
| `setup_optimizer` | 495 | Creates the MuonAdamW optimizer: 6 AdamW groups (lm_head, token embeddings, value embeddings, `resid_lambdas`, `x0_lambdas`, smear/backout scalars) plus one Muon group per distinct matrix shape |
| `forward` | 535 | Full forward pass: embed -> norm -> blocks (with resid/x0 lambdas + value embeds) -> norm -> lm_head -> softcap -> loss or logits |
| `generate` | 603 | Naive autoregressive generation (no KV cache, batch=1, streaming) |
| `estimate_flops` | 395 | Estimates training FLOPs/token (forward + backward): 6 x matmul params + attention FLOPs that account for sliding windows |
| `num_matmul_params` | 417 | Counts the weights of all `Linear` modules, i.e. the params that take part in matmuls with the token stream (embeddings and per-layer scalars are excluded; the tiny smear gate is a `Linear`, so it is counted) |
| `estimate_decode_flops` | 427 | Forward FLOPs to decode one token at a given context length: 2 per matmul param plus attention over `min(context, window)` per layer |
| `estimate_prefill_flops` | 438 | Forward FLOPs to prefill a prompt of N tokens (causal, so token t attends to `min(t, window)`) |
| `kv_bytes_per_token` | 450 | Bytes of KV cache *stored* per token per row (all layers, in `COMPUTE_DTYPE`) |
| `kv_read_bytes` | 456 | Bytes of KV cache *read* by one decode step at a given context length (sliding-window layers read only the last `window` tokens) |
| `num_scaling_params` | 466 | Returns parameter counts by group for scaling law analysis |

The four inference cost methods feed `scripts/infer_bench.py` (see below and `inference_logic.md`).

### CausalSelfAttention (class, line 99)
Multi-head attention with GQA, RoPE, QK norm, value residual (ResFormer), and Flash Attention 3 / SDPA fallback.

### MLP (class, line 177)
Feedforward network: `Linear(n_embd, 4*n_embd)` -> `ReLU^2` -> `Linear(4*n_embd, n_embd)`. No bias.

### Block (class, line 200)
Pre-LN transformer block: `x + attn(norm(x))`, then `x + mlp(norm(x))`.

### Helper Functions
| Function | Line | Description |
|----------|------|-------------|
| `norm(x)` | 42 | Parameterless RMS normalization via `F.rms_norm` |
| `Linear` (class) | 60 | `nn.Linear` subclass that casts weights to the input dtype in forward (replaces autocast; master weights stay fp32) |
| `has_ve(layer_idx, n_layer)` | 68 | Returns True if layer should have value embedding (alternating pattern) |
| `apply_rotary_emb(x, cos, sin)` | 72 | Applies RoPE by rotating pairs of dimensions (by -theta; only the relative q/k rotation matters, and the sign is kept for checkpoint compatibility) |

---

## Optimizers (`nanochat/optim.py`)

### MuonAdamW (class, line 184)
Combined optimizer for single-GPU and distributed training. Uses **Muon** (Newton-Schulz orthogonalized momentum SGD) for 2D weight matrices and **AdamW** for everything else (embeddings, lm_head, scalars). One class covers both cases: without a multi-rank process group all communication is skipped and the only rank owns every parameter; with several ranks it synchronizes gradients itself (nanochat does not use DDP) and shards optimizer state ZeRO-2 style.

Key algorithmic components:
- **Nesterov momentum** with warmup from 0.85 to 0.97 (schedule set in `base_train.py`)
- **MuonEq** row equilibration: rescales every row to the mean row norm before orthogonalization (applied in every Muon step, no flag)
- **Polar Express** orthogonalization (5 iterations, replaces original Newton-Schulz)
- **Muon+** renormalization: after orthogonalization, rescales the Frobenius norm to `sqrt(min(m, n))`, the norm of an exactly semi-orthogonal matrix (applied in every Muon step, no flag)
- **NorMuon variance reduction**: per-neuron adaptive LR normalization
- **Cautious weight decay**: only decays weights that agree in sign with the gradient
- **Fused kernels** via `@torch.compile` for both AdamW and Muon steps. AdamW math runs in fp32, so bf16 params (`wte`, value embeddings) also work on MPS, which does not promote mixed dtypes

| Method | Line | Description |
|--------|------|-------------|
| `adamw_step_fused` | 24 | Compiled fused AdamW (fp32 math): decay -> lerp moments -> bias correct -> update |
| `muon_step_fused` | 112 | Compiled fused Muon: momentum -> MuonEq -> polar_express -> Muon+ -> variance_reduction -> cautious update |
| `_reduce_adamw` | 274 | Phase 1, AdamW group: async `all_reduce` for params under 1024 elements, `reduce_scatter` for larger ones (`shape[0]` must divide by the world size). No communication on a single rank |
| `_reduce_muon` | 295 | Phase 1, Muon group: stacks the K same-shape grads and `reduce_scatter`s them, so each rank receives its chunk of ceil(K/N) params |
| `_compute_adamw` | 320 | Phase 2: waits for the reduce, runs the AdamW kernel on this rank's slice (optimizer state is sharded), launches `all_gather` for large params |
| `_compute_muon` | 362 | Phase 2: stacked Muon update of the params this rank owns (all same-shape params processed together), launches `all_gather` |
| `_finish_gathers` | 419 | Phase 3: waits for the gathers and copies the updated Muon params back from the stacked buffer |
| `step` | 429 | Runs the three phases over all param groups |

`step` uses a 3-phase async communication pattern so that communication overlaps with compute:
1. Launch all async reduce_scatter/all_reduce ops
2. Wait for reduce, compute update, launch all_gather
3. Wait for gathers, copy back

---

## Inference Engine (`nanochat/engine.py`)

### KVCache (class, line 82)
KV cache for Flash Attention 3's `flash_attn_with_kvcache` API. Tensors are `(B, T, H, D)` format.

| Method | Line | Description |
|--------|------|-------------|
| `reset` | 106 | Zero out cache positions |
| `get_pos` | 111 | Get current sequence position |
| `get_layer_cache` | 115 | Return (k, v) cache views for a layer |
| `advance` | 119 | Move position forward by N tokens |
| `prefill` | 123 | Copy from another cache (for multi-sample generation) |

### Engine (class, line 169)
Efficient batched generation with KV cache and tool use (calculator). The KV cache is allocated in `COMPUTE_DTYPE` (bf16 on CUDA GPUs with compute capability 8.0+, fp32 otherwise, overridable with `NANOCHAT_DTYPE`), so it always matches the dtype the forward pass emits.

| Method | Line | Description |
|--------|------|-------------|
| `generate` | 176 | Streaming generation: batch=1 prefill, then clone KV cache for N samples. Handles tool use state machine (python_start/end, calculator eval, output injection). Yields `(token_column, token_masks)` per step. |
| `generate_batch` | 277 | Non-streaming wrapper that collects all tokens into final sequences. |

### sample_next_token (function, line 141)
Samples from logits with temperature and top-k. Returns `(B, 1)` tensor.

### Calculator Tool Functions
| Function | Line | Description |
|----------|------|-------------|
| `use_calculator` | 46 | Safely evaluates math expressions and `.count()` string operations |
| `eval_with_timeout` | 35 | Evaluates expression with 3-second timeout |

---

## Data Loading (`nanochat/dataloader.py`)

### tokenizing_distributed_data_loader_with_state_bos_bestfit (function, line 74)
Primary dataloader for pretraining. BOS-aligned with best-fit document packing.

**Algorithm per batch row:**
1. From a buffer of tokenized documents, pick the **largest** doc that fits entirely
2. Repeat until no doc fits
3. Crop the **shortest** remaining doc to fill the rest (100% utilization, ~35% tokens cropped)

Properties: every row starts with BOS, no padding, supports DDP sharding, epoch tracking, resume from checkpoint.

### _document_batches (generator, line 25)
Infinite iterator over Parquet row groups, yielding batches of text strings with resume support.

---

## Flash Attention (`nanochat/flash_attention.py`)

Unified interface that exports the FA3 API and decides once at import time (`USE_FA3`) which implementation to use:
- **FA3**, when a kernel exists for the GPU (loaded through the `kernels` package: `varunneal/flash-attention-3` on Hopper sm90, and the community kernel `kernels-community/flash-attn3` on other CUDA GPUs it has a build for, currently Ada sm89 and Ampere sm80/sm86) and `COMPUTE_DTYPE` is bf16
- **PyTorch SDPA** otherwise: Blackwell, MPS, CPU, fp16/fp32 compute, or no kernel build for the GPU

| Function | Line | Description |
|----------|------|-------------|
| `_load_flash_attention_3` | 23 | Loads the FA3 kernel for the current GPU (Hopper: varunneal kernel; other CUDA GPUs: community kernel if `has_kernel`); returns None if unavailable |
| `_resolve_use_fa3` | 56 | Decides once whether to use FA3: availability, test override (`_override_impl`), and bf16 compute dtype |
| `flash_attn_func` | 115 | Training attention (no KV cache). Input: `(B, T, H, D)` |
| `flash_attn_with_kvcache` | 139 | Inference attention with KV cache management |
| `_sdpa_attention` | 77 | SDPA fallback with sliding window support via explicit masks |

---

## FP8 Training (`nanochat/fp8.py`)

Minimal FP8 implementation (~150 lines vs torchao's ~2000 lines). Drop-in replacement using `torch._scaled_mm`.

| Component | Line | Description |
|-----------|------|-------------|
| `_to_fp8` | 82 | Dynamic tensorwise quantization: compute scale, clamp, cast |
| `_Float8Matmul` | 125 | Custom autograd: forward (e4m3fn), backward (e5m2 for grads) |
| `Float8Linear` | 195 | Drop-in `nn.Linear` replacement |
| `convert_to_float8_training` | 243 | Module tree walker that swaps Linear -> Float8Linear |

---

## Code Execution (`nanochat/execution.py`)

Runs Python code produced by the model (used by `tasks/humaneval.py`) in a fresh subprocess. It protects against accidents (fork bombs, deleted files, killed processes), not against adversarial code: network access is not blocked and there is no kernel-level isolation.

| Component | Line | Description |
|-----------|------|-------------|
| `ExecutionResult` | 33 | Dataclass: `success`, `stdout`, `stderr`, `error` (last line of the traceback), `timeout`, `memory_exceeded` |
| `GUARD` | 47 | Preamble that runs inside the subprocess before the untrusted code: applies address-space/data/stack rlimits (256MB by default, skipped on macOS), disables destructive functions (`os.system`, `os.kill`, `shutil.rmtree`, `subprocess.Popen`, ...) and blocks a few modules |
| `execute_code` | 74 | Runs `GUARD` + code with `sys.executable -c`: scrubbed environment (`PATH` only), stdin closed, a temporary working directory that is deleted afterwards, and a hard timeout (5 seconds by default, the child is killed) |

---

## Evaluation

### CORE Evaluation (`nanochat/core_eval.py`)
Implements the CORE metric from the DCLM paper. Supports multiple choice, schema, and language modeling tasks with few-shot prompting.

| Function | Line | Description |
|----------|------|-------------|
| `evaluate_task` | 423 | Evaluates full dataset with distributed support |
| `evaluate_example` | 318 | Evaluates single example (MC: lowest avg loss wins; LM: exact match) |
| `forward_model` | 278 | Forward pass returning per-token losses and predictions |

### Loss Evaluation (`nanochat/loss_eval.py`)
| Function | Description |
|----------|-------------|
| `evaluate_bpb` | Computes bits-per-byte on validation data |

---

## Inference Benchmark (`scripts/infer_bench.py`)

CUDA-only benchmark of a trained checkpoint: `python -m scripts.infer_bench -i base -g d12`. It prints a static card (the inference cost implied by the architecture), a batch-1 prefill measurement (MFU), and a sweep over decode batch sizes with TTFT, TPOT, tokens/sec, MBU, MFU and peak VRAM. The last stdout line is the same data as one JSON document. The cost model is explained in `inference_logic.md`.

---

## Tokenizer (`nanochat/tokenizer.py`)

Custom BPE tokenizer trained with rustbpe, inference via tiktoken (`RustBPETokenizer` is the only tokenizer implementation). GPT-4 style split pattern with `\p{N}{1,2}` digit grouping. `decode_single_token_bytes` returns the raw bytes of one token; `scripts/tok_train.py` uses it to build the `token_bytes` table for bits-per-byte, which avoids the str round-trip that miscounted tokens that are not valid standalone UTF-8.

**9 special tokens** for conversation format:
`<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`

---

## Checkpoint Management (`nanochat/checkpoint_manager.py`)

| Function | Line | Description |
|----------|------|-------------|
| `save_checkpoint` | 59 | Saves model/optimizer/meta to disk (model+meta on rank 0, optimizer per-rank) |
| `load_checkpoint` | 96 | Loads checkpoint from disk |
| `load_model` | 215 | High-level: loads model by phase (base/sft/rl) with auto-discovery |
| `build_model` | 128 | Builds GPT from checkpoint with config patching for old checkpoints |

---

## Training Scripts

### Pretraining (`scripts/base_train.py`)
Full pretraining pipeline with:
- Auto-computed optimal batch size (Power Lines paper: B_opt proportional to D^0.383)
- Scaling law-based training horizon (tokens = ratio * params)
- WSD learning rate schedule (warmup -> stable -> warmdown)
- Muon momentum schedule (0.85 -> 0.97 over 400 steps, then warmdown to 0.90 during LR warmdown)
- Cosine weight decay rampdown to zero
- Trains on the ClimbMix-400B dataset (upgraded from FineWeb-EDU-100B)
- CORE metric evaluation, model sampling, wandb logging

### SFT (`scripts/chat_sft.py`)
Supervised fine-tuning on a conversation mixture (~790K rows with the default epochs):
- SmolTalk (460K rows), MMLU `auxiliary_train` (100K rows per epoch, `--mmlu-epochs`, default 3), GSM8K (8K rows per epoch, `--gsm8k-epochs`, default 4)
- Inherits hyperparameters from pretrained checkpoint
- BOS-aligned bestfit-pad packing (no token cropping, pads instead)
- ChatCORE evaluation during training: mean centered accuracy over ARC-Easy, ARC-Challenge, MMLU, GSM8K and HumanEval

### RL (`scripts/chat_rl.py`)
Reinforcement learning on GSM8K math problems:
- Simplified REINFORCE / GRPO variant (no KL, no PPO ratio, on-policy)
- DAPO-style token-level normalization
- Binary rewards from task.reward() (1.0 correct, 0.0 incorrect)
- Advantages: simple centering (r - mean)
- pass@k evaluation on GSM8K validation set

---

## Tasks (`tasks/`)

| File | Class | Description |
|------|-------|-------------|
| `common.py` | `TaskMixture`, `TaskSequence`, `HubDataset`, `load_hub_dataset` | Task composition utilities. `load_hub_dataset` (line 59) downloads a dataset's parquet export from the HuggingFace Hub into `<base_dir>/task_data/` once (one rank downloads, the others wait) and returns a `HubDataset`: a pyarrow-backed stand-in for a `datasets.Dataset` with the same seeded `shuffle`. The `datasets` library is no longer a dependency |
| `gsm8k.py` | `GSM8K` | Math word problems with `reward()` and `evaluate()` methods |
| `arc.py` | `ARC` | Science multiple choice (Easy + Challenge) |
| `mmlu.py` | `MMLU` | Massive Multitask Language Understanding (single "all" subset; training data is the `auxiliary_train` split) |
| `humaneval.py` | `HumanEval` | Code generation benchmark; completions are run with `execute_code` |
| `smoltalk.py` | `SmolTalk` | General conversation data |
