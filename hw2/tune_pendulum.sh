#!/bin/bash
# Hyperparameter tuning for InvertedPendulum-v4
# Goal: reach return of 1000 within 100K environment steps
# Default: -n 100 -b 5000 = 500K steps (too many)
# Strategy: reduce batch size so we get more updates within 100K steps

# Run 1: Default (baseline comparison)
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    --exp_name pendulum_default

# Run 2: Smaller batch + rtg + na (100 iters * 1000 = 100K steps)
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
    -rtg -na -lr 0.02 --exp_name pendulum_rtg_na_lr02

# Run 3: Even smaller batch (200 iters * 500 = 100K steps)
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 200 -b 500 -eb 1000 \
    -rtg -na -lr 0.02 --exp_name pendulum_b500_lr02

# Run 4: With baseline
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
    -rtg -na -lr 0.02 --use_baseline -blr 0.01 -bgs 5 --exp_name pendulum_baseline

# Run 5: Higher learning rate
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
    -rtg -na -lr 0.05 --exp_name pendulum_lr05

# Run 6: Smaller network (faster to train for simple task)
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
    -rtg -na -lr 0.02 -l 1 -s 32 --exp_name pendulum_small_net

# Run 7: Lower discount (focus on immediate rewards)
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 \
    -rtg -na -lr 0.02 --discount 0.95 --exp_name pendulum_disc095

echo "Done! Check WandB and run: python plot.py --section 6"
