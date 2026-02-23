"""
Chat Model RL with GRPO (Group Relative Policy Optimization)

A more advanced RL training script compared to chat_rl.py (simplified REINFORCE).
Ported from reference/nanuGPT implementation, adapted for nanochat's architecture.

Key differences from chat_rl.py:
- Reference model: frozen copy of the policy for KL regularization
- PPO-style clipping: importance sampling ratio clamped to [1-eps, 1+eps]
- KL penalty: reverse KL divergence term to prevent policy drift
- GRPO / Dr.GRPO variants: with or without std normalization of advantages

Algorithm (GRPO):
1. Generate K completions per problem using current policy
2. Compute per-token log-probs for policy AND frozen reference model
3. Calculate rewards (binary: 1.0 correct, 0.0 incorrect)
4. Compute advantages: (r - mean(r)) / (std(r) + eps)  [GRPO]
                     or (r - mean(r))                    [Dr.GRPO]
5. Compute importance ratio: exp(current_logp - old_logp)
6. PPO clipped surrogate: min(ratio * adv, clip(ratio) * adv)
7. KL penalty: beta * (exp(ref_logp - cur_logp) - (ref_logp - cur_logp) - 1)
8. Loss = -(surrogate - KL_penalty) averaged over valid tokens

Reference:
- GRPO: https://arxiv.org/abs/2402.03300
- Dr.GRPO: removes std normalization and sequence-level normalization
- Adapted from: ~/offline-git/GPT/mini/nanoGPT/reference/nanuGPT/scripts/rl/nanu_rlvr.py

Usage:
    Single GPU:
        python -m scripts.chat_rl_grpo

    8 GPUs distributed:
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_grpo -- --run=default

    Dr.GRPO variant:
        python -m scripts.chat_rl_grpo --grpo-variant=dr_grpo
"""

import argparse
import copy
import os
import itertools
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="GRPO reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# GRPO-specific
parser.add_argument("--grpo-variant", type=str, default="grpo", choices=["grpo", "dr_grpo"],
                    help="GRPO variant: 'grpo' (standard) or 'dr_grpo' (no std normalization, token-level aggregation)")
parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient (higher = more conservative updates)")
parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clipping parameter")
parser.add_argument("--mu", type=int, default=1, help="number of GRPO updates per rollout batch (reuse same rollout data)")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl-grpo", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# Create frozen reference model (deep copy, no gradients)
print0("Creating frozen reference model...")
ref_model = copy.deepcopy(model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
print0("Reference model created.")

# -----------------------------------------------------------------------------
# Helper: compute per-token log probabilities efficiently
# (Ported from nanoGPT reference/nanuGPT/scripts/rl/nanu_rlvr.py:531-551)

def selective_log_softmax(logits, input_ids):
    """
    Compute log P(token_t) for each position t in input_ids, using logits.
    Memory-efficient: only extracts log-probs for the tokens that actually appear.

    Args:
        logits: (B, T, V) model output logits
        input_ids: (B, T) token IDs

    Returns:
        (B, T) tensor of log probabilities
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # float32 for precision
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def compute_log_probs(model_to_use, inputs, targets):
    """
    Compute per-token log-probs for the given model on (inputs, targets).
    Uses the model's forward to get per-token NLL, then negates to get log-probs.

    Args:
        model_to_use: GPT model (policy or reference)
        inputs: (B, T) input token IDs
        targets: (B, T) target token IDs (with -1 for ignored positions)

    Returns:
        (B, T) tensor of log-probs (0 where target is -1)
    """
    with autocast_ctx:
        # model.forward with loss_reduction='none' returns NLL = -log_p per token
        nll = model_to_use(inputs, targets, loss_reduction='none').view_as(inputs)
    log_probs = -nll  # negate to get log-probs
    # Zero out positions where target is -1 (padding/prompt/tool tokens)
    log_probs = log_probs * (targets >= 0).float()
    return log_probs


# -----------------------------------------------------------------------------
# GRPO loss function
# (Ported and adapted from nanoGPT reference/nanuGPT/scripts/rl/nanu_rlvr.py:636-690)

def grpo_loss(current_log_probs, old_log_probs, ref_log_probs, rewards, targets,
              beta, epsilon, grpo_variant, num_samples):
    """
    Compute the GRPO (or Dr.GRPO) policy gradient loss.

    Args:
        current_log_probs: (B, T) log-probs from current policy (with gradients)
        old_log_probs: (B, T) log-probs from policy at rollout time (detached)
        ref_log_probs: (B, T) log-probs from frozen reference model (detached)
        rewards: (B,) scalar rewards per sample
        targets: (B, T) target token IDs (-1 = ignore)
        beta: KL penalty coefficient
        epsilon: PPO clipping parameter
        grpo_variant: "grpo" or "dr_grpo"
        num_samples: number of samples per example (for reshaping rewards)

    Returns:
        (loss, avg_reward) tuple
    """
    B = rewards.shape[0]
    num_examples = B // num_samples

    # Importance sampling ratio: how much has the policy changed since rollout?
    ratio = torch.exp(current_log_probs - old_log_probs)

    # Compute advantages from rewards
    # Reshape rewards: (num_examples, num_samples) to compute per-example statistics
    rewards_matrix = rewards.view(num_examples, num_samples)
    avg_reward = rewards_matrix.mean().item()
    mean_rewards = rewards_matrix.mean(dim=1).repeat_interleave(num_samples)

    if grpo_variant == "dr_grpo":
        # Dr.GRPO: use only (r - mean) as the advantage, no std normalization
        advantages = (rewards.view(-1) - mean_rewards).unsqueeze(1)  # (B, 1) broadcast over T
    else:
        # Standard GRPO: z-score normalization
        std_rewards = rewards_matrix.std(dim=1).repeat_interleave(num_samples)
        advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

    # PPO-style clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate = torch.min(surr1, surr2)

    # Reverse KL divergence: stable approximation around 0
    # KL(pi || ref) approx= exp(ref_logp - cur_logp) - (ref_logp - cur_logp) - 1
    delta = ref_log_probs - current_log_probs
    kl = torch.exp(delta) - delta - 1.0

    # Per-token objective: clipped surrogate - KL penalty
    per_token = surrogate - beta * kl

    # Valid token mask (positions where we have real targets, not prompt/padding)
    valid_mask = (targets >= 0).float()

    if grpo_variant == "grpo":
        # Standard GRPO: normalize by per-sequence token count, then mean over sequences
        token_counts = valid_mask.sum(dim=1).clamp_min(1)
        loss = -((per_token * valid_mask).sum(dim=1) / token_counts).mean()
    elif grpo_variant == "dr_grpo":
        # Dr.GRPO: token-level aggregation without per-sequence normalization
        denom = max(per_token.size(0), 1)
        loss = -(per_token * valid_mask).sum() / denom
    else:
        raise ValueError(f"Unknown GRPO variant: {grpo_variant}")

    return loss, avg_reward


# -----------------------------------------------------------------------------
# Rollout / sampling generator

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    """
    Generate a batch of rollouts for GRPO training.

    For each example:
    1. Sample K completions from current policy
    2. Compute per-token log-probs for both policy and reference model
    3. Calculate rewards
    4. Prepare training tensors (inputs, targets, old_log_probs, ref_log_probs, rewards, advantages)

    Yields:
        Tuple of (sequences, inputs, targets, old_log_probs, ref_log_probs, rewards, advantages)
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        # Get conversation and prepare prompt
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate K samples using batched generation
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            with autocast_ctx:
                seqs_batch, masks_batch = engine.generate_batch(
                    tokens, num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_k=args.top_k, seed=seed,
                )
            generated_token_sequences.extend(seqs_batch)
            masks.extend(masks_batch)

        # Calculate rewards
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad sequences and build tensors
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_seqs = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1  # mask prompt and tool tokens

        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        # Compute old_log_probs (from current policy, detached) and ref_log_probs
        old_log_probs = compute_log_probs(model, inputs, targets)
        ref_log_probs = compute_log_probs(ref_model, inputs, targets)

        # Compute advantages (same as chat_rl.py for compatibility)
        mu = rewards_tensor.mean()
        advantages = rewards_tensor - mu

        yield generated_token_sequences, inputs, targets, old_log_probs, ref_log_probs, rewards_tensor, advantages


# -----------------------------------------------------------------------------
# Evaluation (reused from chat_rl.py)

def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None, num_samples=1, max_completion_tokens=256,
    temperature=0.0, top_k=50
):
    """Evaluate GSM8K pass@k. Yields records for distributed aggregation."""
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens, num_samples=num_samples,
            max_tokens=max_completion_tokens, temperature=temperature, top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        yield {"idx": idx, "outcomes": outcomes}


# -----------------------------------------------------------------------------
# Training loop

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set initial learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# LR schedule: linear rampdown to zero
def get_lr_multiplier(it):
    return 1.0 - it / num_steps

# Per-rank batch size
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size == 0
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Examples per rank: {examples_per_rank}")
print0(f"GRPO variant: {args.grpo_variant} | beta={args.beta} | epsilon={args.epsilon} | mu={args.mu}")

batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate pass@k periodically
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        with autocast_ctx:
            records = list(run_gsm8k_eval(
                val_task, tokenizer, engine,
                num_samples=args.device_batch_size,
                max_examples=args.eval_examples, temperature=1.0,
            ))
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({"step": step, **log_passk})

    # Collect rollouts and compute GRPO loss
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        (sequences_all, inputs_all, targets_all,
         old_logp_all, ref_logp_all, rewards_all, advantages_all) = next(batch_iterator)

        model.train()

        # Multiple GRPO updates per rollout batch (controlled by --mu)
        for mu_step in range(args.mu):
            assert inputs_all.size(0) % args.device_batch_size == 0
            num_passes = inputs_all.size(0) // args.device_batch_size
            for pass_idx in range(num_passes):
                b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                old_logp = old_logp_all[b0:b1]
                ref_logp = ref_logp_all[b0:b1]
                rewards = rewards_all[b0:b1]

                # Compute current log-probs (with gradients)
                with autocast_ctx:
                    current_logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
                current_logp = current_logp * (targets >= 0).float()

                # GRPO loss
                loss, avg_reward = grpo_loss(
                    current_logp, old_logp, ref_logp, rewards, targets,
                    beta=args.beta, epsilon=args.epsilon,
                    grpo_variant=args.grpo_variant,
                    num_samples=args.num_samples,
                )
                # Normalize by accumulation factors
                loss = loss / (num_passes * examples_per_rank * args.mu)
                loss.backward()
                print0(f"Step {step}/{num_steps} | Ex {example_step} | Mu {mu_step} | Pass {pass_idx} | "
                       f"loss: {loss.item():.6f} | reward: {rewards.mean().item():.3f}")

        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Log rollout statistics
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mean_reward_t = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_seq_len_t = torch.tensor(mean_seq_len, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_seq_len_t, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_t.item()
        mean_seq_len = mean_seq_len_t.item()
    print0(f"Step {step}/{num_steps} | Avg reward: {mean_reward:.3f} | Avg seq len: {mean_seq_len:.1f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_seq_len})

    # Optimizer step
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm})

    # Save checkpoints
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_grpo_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir, step, model.state_dict(), None,
            {"model_config": model.config.__dict__},
        )
        print(f"Saved GRPO checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL (GRPO)", data=[user_config])

wandb_run.finish()
compute_cleanup()
