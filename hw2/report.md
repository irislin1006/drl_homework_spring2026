# CS 185/285 HW2: Policy Gradients — Report

---

## Experiment 1: CartPole (Section 3)

### Commands Used

```bash
# Small batch experiments
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

# Large batch experiments
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

No parameters were changed from defaults beyond those shown above.

### Learning Curves

#### Small Batch (b=1000)

![Small batch learning curves](figures/cartpole_small_batch.png)

#### Large Batch (b=4000)

![Large batch learning curves](figures/cartpole_large_batch.png)

### Questions

**Which value estimator has better performance without advantage normalization: the trajectory-centric one, or the one using reward-to-go?**

Reward-to-go performs better. In the small batch case, `cartpole_rtg` (orange) reaches ~150-160 and trends upward, while `cartpole` (blue) stays around 75-80 and never consistently improves. In the large batch case, both eventually reach 200, but `cartpole_lb_rtg` converges faster and more stably, while `cartpole_lb` (blue) still shows significant drops even at 300K+ steps.

**Between the two value estimators, why do you think one is generally preferred over the other?**

Reward-to-go is preferred because it exploits causality: an action at time t cannot affect rewards that already occurred before t. The trajectory-centric estimator assigns credit for *all* rewards (including past ones) to every action, which adds noise to the gradient. For example, if reward at t=0 was unusually high, the trajectory-centric estimator would incorrectly reinforce the action at t=50 for that reward. Reward-to-go removes this irrelevant signal, reducing variance without introducing any bias.

**Did advantage normalization help?**

Yes, substantially. In the small batch plot, the two runs with advantage normalization (`na` in green, `rtg + na` in red) are the first to reach and sustain 200, converging around 15K-20K steps. Without normalization, `cartpole` (blue) stays below 100 and `cartpole_rtg` (orange) is unstable. Normalization ensures that within each batch, roughly half the advantages are positive (reinforced) and half are negative (discouraged), rather than all being pushed in the same direction when returns are uniformly positive.

**Did the batch size make an impact?**

Yes. With b=4000, all configurations eventually approach 200, including vanilla (blue) which struggled at b=1000. The large batch curves are also smoother overall. The `rtg + na` configuration (red) converges rapidly and stays at 200 almost perfectly from ~100K steps onward. Larger batches provide a lower-variance estimate of the policy gradient, since averaging over more trajectories reduces the noise in the gradient.

---

## Experiment 2: HalfCheetah (Section 4)

### Commands Used

```bash
# No baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```

No parameters were changed from defaults beyond those shown above.

### Learning Curves

#### Baseline Loss

![Baseline loss](figures/cheetah_baseline_loss.png)

#### Eval Return

![Eval return](figures/cheetah_eval_return.png)

### Reduced Baseline Gradient Steps / Learning Rate Experiment

```bash
# Lower Baseline Gradient Steps
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline_low_bgs

# Lower Baseline Learning Rate
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.005 -bgs 5 --exp_name cheetah_baseline_low_blr

# Optional
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 -na --video_log_freq 10 --exp_name cheetah_baseline_na
```

**How does reducing baseline gradient steps (or baseline learning rate) affect:**

**(a) The baseline learning curve?**

With the default settings (bgs=5, blr=0.01), the baseline loss drops from ~205 to ~14 by the end of training. Reducing baseline gradient steps to 3 results in a slightly higher final baseline loss (~19), indicating the critic has fewer optimization steps per iteration to fit the changing target values. Reducing the baseline learning rate to 0.005 shows a similar final loss (~15) but converges more slowly in the early iterations. In both reduced settings, the critic is less able to keep up with the evolving policy, leading to slightly less accurate value predictions.

**(b) The performance of the policy?**

All three baseline configurations significantly outperform the no-baseline run (which plateaus around -240 eval return). The default baseline achieves ~210 average eval return over the last 5 iterations, while the low-bgs run reaches ~200 and the low-blr run reaches ~206. The differences are modest, suggesting that even a slightly worse critic still provides a useful variance reduction signal. However, the default configuration is the most stable and reaches the highest peak performance.

---

## Experiment 3: LunarLander (Section 5)

### Commands Used

```bash
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0 --exp_name lunar_lander_lambda0

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.95 --exp_name lunar_lander_lambda0_95

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.98 --exp_name lunar_lander_lambda0_98

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.99 --exp_name lunar_lander_lambda0_99

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 1 --exp_name lunar_lander_lambda1
```

No parameters were changed from defaults beyond those shown above.

### Learning Curves

![LunarLander lambda comparison](figures/lunar_lander_lambda_comparison.png)

### Questions

**How did lambda affect task performance?**

lambda=0 (blue) performed the worst, remaining mostly negative throughout training and failing to learn a good policy. The intermediate values lambda=0.95, 0.98, and 0.99 all performed significantly better, with lambda=0.98 (green) and lambda=0.99 (red) achieving the highest returns, reaching above 100-150 during training. lambda=1 (purple) showed more oscillation and instability compared to the intermediate values, though it still outperformed lambda=0. Overall, moderate lambda values (0.95-0.99) provided the best balance, while the extremes (0 and 1) performed worse for different reasons.

**What does lambda=0 correspond to? What about lambda=1?**

lambda=0 uses only the 1-step TD error (delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)) as the advantage estimate. This relies entirely on the critic's predictions, giving low variance but high bias — if the critic is inaccurate (especially early in training), the advantage estimates are misleading. lambda=1 uses the full Monte Carlo return minus the baseline, equivalent to the standard advantage (q_values - V(s)). This introduces no bias but has high variance since it depends on the full sequence of sampled rewards. In LunarLander, lambda=0 fails because the critic is too inaccurate early on to provide useful signal alone, while lambda=1 is noisier than the intermediate values that blend both approaches.

---

## Experiment 4: InvertedPendulum (Section 6)

### Commands Used

```bash
# Default
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    --exp_name pendulum_default

# Best hyperparameters
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 200 -b 500 -eb 1000 \
    -rtg -na -lr 0.02 --exp_name pendulum_b500_lr02
```

### Best Hyperparameters

The most impactful change was **reducing the batch size** from 5000 to 500. The default uses 100 iterations * 5000 steps = 500K total environment steps, but only gets one gradient update per 5000 steps. By reducing to b=500 with n=200, we get 200 gradient updates within 100K total steps — 4x more learning per step budget.

The other changes that helped:
- **Reward-to-go (`-rtg`)**: Reduces variance by only crediting actions for future rewards, not past ones.
- **Advantage normalization (`-na`)**: Ensures a balanced gradient signal (some actions reinforced, some discouraged) rather than pushing all actions in the same direction.
- **Higher learning rate (`-lr 0.02`)**: Since InvertedPendulum is a simple task, a higher learning rate allows faster convergence without instability. The default 0.005 is too conservative.

Batch size mattered the most — it determines how many gradient updates you get within the 100K step budget. The variance reduction tricks (rtg, na) and learning rate were secondary but still contributed to faster convergence.

### Learning Curves

![InvertedPendulum comparison](figures/pendulum_comparison.png)

The tuned run reaches a return of 1000 within ~20K-30K environment steps, well under the 100K budget. The default run (b=5000) only reaches ~350 after 500K steps and never hits 1000.