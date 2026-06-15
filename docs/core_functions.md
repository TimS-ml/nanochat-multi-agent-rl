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

### GPT (class, line 216)
Main GPT model. Notable architectural choices:
- **RoPE** (no learned positional embeddings)
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
| `__init__` | 217 | Builds model on meta device (shapes only, no data) |
| `init_weights` | 265 | Initializes all parameters: uniform for attn/MLP, normal for embeddings, zeros for output projections |
| `_precompute_rotary_embeddings` | 331 | Computes cos/sin tables for RoPE (10X over-allocation for safety) |
| `_compute_window_sizes` | 357 | Converts window pattern string ("SSSL") to per-layer (left, right) window size tuples |
| `setup_optimizer` | 450 | Creates MuonAdamW optimizer with 6 param groups (Muon for matrices, AdamW for embeddings/scalars) |
| `forward` | 492 | Full forward pass: embed -> norm -> blocks (with resid/x0 lambdas + value embeds) -> norm -> lm_head -> softcap -> loss or logits |
| `generate` | 560 | Naive autoregressive generation (no KV cache, batch=1, streaming) |
| `estimate_flops` | 393 | Estimates FLOPs/token accounting for sliding window attention |
| `num_scaling_params` | 421 | Returns parameter counts by group for scaling law analysis |

### CausalSelfAttention (class, line 97)
Multi-head attention with GQA, RoPE, QK norm, value residual (ResFormer), and Flash Attention 3 / SDPA fallback.

### MLP (class, line 175)
Feedforward network: `Linear(n_embd, 4*n_embd)` -> `ReLU^2` -> `Linear(4*n_embd, n_embd)`. No bias.

### Block (class, line 198)
Pre-LN transformer block: `x + attn(norm(x))`, then `x + mlp(norm(x))`.

### Helper Functions
| Function | Line | Description |
|----------|------|-------------|
| `norm(x)` | 42 | Parameterless RMS normalization via `F.rms_norm` |
| `Linear` (class) | 60 | `nn.Linear` subclass that casts weights to the input dtype in forward (replaces autocast; master weights stay fp32) |
| `has_ve(layer_idx, n_layer)` | 68 | Returns True if layer should have value embedding (alternating pattern) |
| `apply_rotary_emb(x, cos, sin)` | 72 | Applies RoPE by rotating pairs of dimensions |

---

## Optimizers (`nanochat/optim.py`)

### MuonAdamW (class, line 154)
Single-GPU combined optimizer. Uses **Muon** (Newton-Schulz orthogonalized momentum SGD) for 2D weight matrices and **AdamW** for everything else (embeddings, lm_head, scalars).

Key algorithmic components:
- **Polar Express** orthogonalization (5 iterations, replaces original Newton-Schulz)
- **NorMuon variance reduction**: per-neuron adaptive LR normalization
- **Cautious weight decay**: only decays weights that agree in sign with the gradient
- **Nesterov momentum** with warmup from 0.85 to 0.97 (schedule set in `base_train.py`)
- **Fused kernels** via `@torch.compile` for both AdamW and Muon steps

| Method | Line | Description |
|--------|------|-------------|
| `adamw_step_fused` | 22 | Compiled fused AdamW: decay -> lerp moments -> bias correct -> update |
| `muon_step_fused` | 92 | Compiled fused Muon: momentum -> polar_express -> variance_reduction -> cautious update |
| `_step_adamw` | 196 | Per-param AdamW update with lazy state init |
| `_step_muon` | 231 | Stacked Muon update (all same-shape params processed together) |

### DistMuonAdamW (class, line 299)
Distributed version with ZeRO-2 style optimizer sharding. Uses a 3-phase async communication pattern:
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
Efficient batched generation with KV cache and tool use (calculator).

| Method | Line | Description |
|--------|------|-------------|
| `generate` | 176 | Streaming generation: batch=1 prefill, then clone KV cache for N samples. Handles tool use state machine (python_start/end, calculator eval, output injection). Yields `(token_column, token_masks)` per step. |
| `generate_batch` | 282 | Non-streaming wrapper that collects all tokens into final sequences. |

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

Unified interface that auto-selects FA3 on Hopper GPUs (sm90) and falls back to PyTorch SDPA elsewhere.

| Function | Line | Description |
|----------|------|-------------|
| `flash_attn_func` | 107 | Training attention (no KV cache). Input: `(B, T, H, D)` |
| `flash_attn_with_kvcache` | 131 | Inference attention with KV cache management |
| `_sdpa_attention` | 69 | SDPA fallback with sliding window support via explicit masks |

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

## Tokenizer (`nanochat/tokenizer.py`)

Custom BPE tokenizer trained with rustbpe, inference via tiktoken. GPT-4 style split pattern with `\p{N}{1,2}` digit grouping.

**9 special tokens** for conversation format:
`<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`

---

## Checkpoint Management (`nanochat/checkpoint_manager.py`)

| Function | Line | Description |
|----------|------|-------------|
| `save_checkpoint` | 60 | Saves model/optimizer/meta to disk (model+meta on rank 0, optimizer per-rank) |
| `load_checkpoint` | 97 | Loads checkpoint from disk |
| `load_model` | 216 | High-level: loads model by phase (base/sft/rl) with auto-discovery |
| `build_model` | 129 | Builds GPT from checkpoint with config patching for old checkpoints |

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
Supervised fine-tuning on conversation mixture (~23K examples):
- SmolTalk, MMLU, GSM8K, identity conversations, spelling tasks
- Inherits hyperparameters from pretrained checkpoint
- BOS-aligned bestfit-pad packing (no token cropping, pads instead)
- ChatCORE evaluation during training

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
| `common.py` | `TaskMixture`, `TaskSequence` | Task composition utilities |
| `gsm8k.py` | `GSM8K` | Math word problems with `reward()` and `evaluate()` methods |
| `arc.py` | `ARC` | Science multiple choice (Easy + Challenge) |
| `mmlu.py` | `MMLU` | Massive Multitask Language Understanding (single "all" subset; training data is the `auxiliary_train` split) |
| `humaneval.py` | `HumanEval` | Code generation benchmark |
| `spellingbee.py` | `SpellingBee`, `SimpleSpelling` | Character counting + spelling tasks with `reward()` |
| `smoltalk.py` | `SmolTalk` | General conversation data |
| `customjson.py` | `CustomJSON` | Load custom JSONL conversation data |
