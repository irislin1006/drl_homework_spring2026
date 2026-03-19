# HW 4: LLM RL

This repository is set up to run on Modal by default via `scripts/modal_train.py`. The default Modal GPU is an `H100`.

## Quickstart

```bash
# Install ALL dependencies (torch, transformers, peft, etc.)
# NOTE: base `uv sync` only installs modal. You MUST use --extra remote
# for local GPU training, otherwise you get: ModuleNotFoundError: No module named 'torch'
uv sync --extra remote

uv run modal token new    # only needed for Modal remote runs
uvx wandb login            # only needed if using wandb
```

All training commands below are intended to be run from the repository root.

## Local GPU Runs

If you have a local GPU (H100/H200), you can skip Modal and run `hw4.train` directly.

### Format Copy + GRPO (local)

```bash
uv run python -u -m hw4.train \
  --task format_copy \
  --algo grpo \
  --output_dir runs/format_copy_grpo \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 48 \
  --grad_accum_steps 1 \
  --clip_eps 0.2 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### Format Copy + REINFORCE (local)

```bash
uv run python -u -m hw4.train \
  --task format_copy \
  --algo reinforce \
  --output_dir runs/format_copy_reinforce \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --minibatch_size 48 \
  --grad_accum_steps 1 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### Math Hard + REINFORCE (local)

```bash
uv run python -u -m hw4.train \
  --task math_hard \
  --algo reinforce \
  --output_dir runs/math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 64 \
  --grad_accum_steps 1 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```

### Math Hard + GRPO (local)

```bash
uv run python -u -m hw4.train \
  --task math_hard \
  --algo grpo \
  --output_dir runs/math_hard_grpo \
  --steps 501 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 64 \
  --grad_accum_steps 1 \
  --clip_eps 0.2 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name math_hard_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```
