# HW4: LLM Reinforcement Learning — Report

## Question 1: Approximate KL

The estimator `e^Δ - Δ - 1` where `Δ = log(π_ref / π_new)` is a valid sampled-token KL estimator because:

- When we take the expectation under π_new, `E_π_new[e^Δ - Δ - 1]` equals `KL(π_new || π_ref)` exactly. This follows from the identity: `E_π_new[π_ref/π_new - log(π_ref/π_new) - 1] = Σ π_new(x) [π_ref(x)/π_new(x) - log(π_ref(x)/π_new(x)) - 1]`, which simplifies to the standard KL divergence formula. The estimator is always non-negative (since `e^x - x - 1 ≥ 0` for all x), matching the non-negativity property of KL divergence.

- Computing the **exact full-vocabulary KL** at every token position would require a full softmax over the entire vocabulary (151,665 tokens for Qwen2.5) for **both** the current policy and the reference policy at every completion token. This means two full forward passes with all logits materialized, plus storing two [B, L, V] tensors in memory. With B=64, L=512, V=151K, that's ~40GB just for logits. The approximate estimator only needs the log-probability of the **actually sampled token** from each model — a single scalar per position — making it orders of magnitude cheaper in both compute and memory.

## Question 2: Implementation

I implemented the TODOs in order (1 through 8):

1. `compute_per_token_logprobs` — forward pass, shift logits by 1, `F.cross_entropy`, negate
2. `build_completion_mask` — [B, L-1] mask starting at `prompt_input_len - 1`
3. `approx_kl_from_logprobs` — clamped delta, `exp(delta) - delta - 1`, masked_mean
4. `iter_minibatches` — shuffle indices, slice all RolloutBatch fields consistently
5. `compute_group_advantages` — reshape rewards to [num_groups, group_size], z-score within groups
6. `maybe_normalize_advantages` — global z-score if enabled
7. `Reinforce.update` — new_logp, masked_mean_per_row for seq_logp, pg_loss, kl, entropy
8. `GRPO.update` — importance ratio with clamped log-ratio, PPO clip objective, clipfrac

**Bug/confusion resolved:** In TODO 2 (`build_completion_mask`), I initially created a mask of shape [B, L] instead of [B, L-1], and used offset `prompt_input_len` instead of `prompt_input_len - 1`. The key insight is that logprobs are shifted by 1 relative to input_ids (logprob at index t scores token t+1), so the completion mask must also be [B, L-1] and start one position earlier than the raw prompt length.

## Question 3: GR-REINFORCE vs. GRPO on Math Hard

<!-- TODO: Paste WandB charts comparing math_hard_reinforce (201 steps) vs math_hard_grpo (first 200 of 501 steps) -->

**Observations over the first 200 iterations:**

- **REINFORCE** (201 steps): Expected final train reward ~0.32, eval ~0.31
- **GRPO** (501 steps): At step 200, GRPO should show [FILL: similar/better/worse] reward compared to REINFORCE at step 200

**Why this comparison is interesting:** The commands were deliberately chosen so that both methods see the same number of unique prompts over the first 200 steps (same batch_size=8, group_size=8). The key difference is that GRPO reuses each rollout batch for `ppo_epochs=2` optimization passes with clipped importance ratios, while REINFORCE does a single on-policy update and discards the data. This means GRPO extracts more gradient signal per rollout at the cost of slightly stale updates — the clipping mechanism prevents these off-policy updates from diverging.

<!-- TODO: Fill in actual numbers and paste WandB screenshots after both runs complete -->

## Question 4: GRPO Ablations on Format Copy

### Baseline
- Default format_copy + GRPO: `clip_eps=0.2, kl_coef=0.05, ppo_epochs=2, group_size=6, lr=3e-5, minibatch_size=48, grad_accum_steps=1`

### Ablation Runs

<!-- TODO: Run these and fill in results -->

| Run | Changed Param | Value | Final Reward | Final Eval Match | Notes |
|-----|--------------|-------|-------------|-----------------|-------|
| baseline | — | — | [FILL] | [FILL] | default settings |
| ablation_1 | ppo_epochs | 4 | [FILL] | [FILL] | more reuse per rollout |
| ablation_2 | kl_coef | 0.01 | [FILL] | [FILL] | less KL penalty |
| ablation_3 | kl_coef | 0.2 | [FILL] | [FILL] | more KL penalty |
| ablation_4 | clip_eps | 0.1 | [FILL] | [FILL] | tighter clipping |
| ablation_5 | grad_accum_steps | 1 (with minibatch_size=8) | [FILL] | [FILL] | more optimizer updates per rollout |

### Ablation Commands

```bash
# Ablation 1: ppo_epochs=4
uv run python -u -m hw4.train \
  --task format_copy --algo grpo --output_dir runs/ablation_ppo_epochs_4 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 4 --minibatch_size 48 --grad_accum_steps 1 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name ablation_ppo_epochs_4 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 2: kl_coef=0.01 (smaller KL penalty)
uv run python -u -m hw4.train \
  --task format_copy --algo grpo --output_dir runs/ablation_kl_coef_0.01 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 48 --grad_accum_steps 1 \
  --clip_eps 0.2 --kl_coef 0.01 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name ablation_kl_coef_0.01 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 3: kl_coef=0.2 (larger KL penalty)
uv run python -u -m hw4.train \
  --task format_copy --algo grpo --output_dir runs/ablation_kl_coef_0.2 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 48 --grad_accum_steps 1 \
  --clip_eps 0.2 --kl_coef 0.2 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name ablation_kl_coef_0.2 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 4: clip_eps=0.1 (tighter clipping)
uv run python -u -m hw4.train \
  --task format_copy --algo grpo --output_dir runs/ablation_clip_eps_0.1 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 48 --grad_accum_steps 1 \
  --clip_eps 0.1 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name ablation_clip_eps_0.1 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 5: grad_accum_steps=1, minibatch_size=8 (more optimizer updates)
uv run python -u -m hw4.train \
  --task format_copy --algo grpo --output_dir runs/ablation_grad_accum_1_mb8 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 1 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name ablation_grad_accum_1_mb8 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10
```

### Analysis

<!-- TODO: Fill in after running ablations -->

**Which hyperparameters mattered most:** [FILL]

**Which settings made learning worse and why:** [FILL]

## Question 5: Qualitative Behavior (Math Hard)

<!-- TODO: Paste 1-2 example model generations from WandB math_hard runs -->

### Example 1

**Prompt:** [FILL from WandB]

**Model completion:** [FILL from WandB]

**Why this is interesting:** [FILL]

### Example 2

**Prompt:** [FILL from WandB]

**Model completion:** [FILL from WandB]

**Why this is interesting:** [FILL]

---

## Submission Checklist

- [ ] 4 required training runs completed (format_copy GRPO, format_copy REINFORCE, math_hard REINFORCE, math_hard GRPO)
- [ ] 5+ GRPO ablation runs on format_copy completed
- [ ] Report questions 1-5 answered with actual data filled in
- [ ] WandB screenshots/plots pasted into report
- [ ] Gradescope submission bundle built
- [ ] Code + artifacts zipped and uploaded to autograder
- [ ] Report PDF uploaded to report assignment
