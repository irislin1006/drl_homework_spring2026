#!/usr/bin/env python3
"""Generate all report plots from metrics.jsonl files."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_metrics(run_name):
    path = os.path.join(RUNS_DIR, run_name, "metrics.jsonl")
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    # If the run was executed twice (appended), keep only the last run.
    # Detect by finding the last occurrence of step 0 with eval metrics.
    last_start = 0
    for i, l in enumerate(lines):
        step = l.get("step")
        m = l.get("metrics", {})
        if step == 0 and any("eval/" in k for k in m):
            last_start = i
    return lines[last_start:]


def extract_train_series(lines, key_substr):
    """Extract (step, value) pairs for a training metric."""
    steps, vals = [], []
    for l in lines:
        m = l.get("metrics", {})
        step = l.get("step")
        if step is None:
            continue
        for k, v in m.items():
            if key_substr in k and isinstance(v, (int, float)):
                steps.append(step)
                vals.append(v)
                break
    return np.array(steps), np.array(vals)


def extract_eval_series(lines, key_substr):
    """Extract (step, value) pairs for an eval metric."""
    steps, vals = [], []
    for l in lines:
        m = l.get("metrics", {})
        step = l.get("step")
        if step is None:
            continue
        for k, v in m.items():
            if "eval/" in k and key_substr in k and isinstance(v, (int, float)):
                steps.append(step)
                vals.append(v)
                break
    return np.array(steps), np.array(vals)


# ============================================================
# Plot 1: Format Copy — REINFORCE vs GRPO (reward + eval)
# ============================================================
def plot_format_copy_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run, label, color in [
        ("format_copy_grpo", "GRPO", "tab:blue"),
        ("format_copy_reinforce", "REINFORCE", "tab:orange"),
    ]:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "mean_total_reward")
        axes[0].plot(s, v, label=label, color=color, alpha=0.8)
        s, v = extract_eval_series(lines, "fraction_predicted_number_matches")
        axes[1].plot(s, v, "o-", label=label, color=color, markersize=6)

    axes[0].set_title("Format Copy — Train Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Format Copy — Eval Exact Match")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Exact Match")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "format_copy_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/format_copy_comparison.png")


# ============================================================
# Plot 2: Math Hard — REINFORCE vs GRPO (reward + eval)
# ============================================================
def plot_math_hard_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run, label, color, max_step in [
        ("math_hard_reinforce", "REINFORCE (201 steps)", "tab:orange", 201),
        ("math_hard_grpo", "GRPO (501 steps)", "tab:blue", 501),
    ]:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "mean_total_reward")
        axes[0].plot(s, v, label=label, color=color, alpha=0.7)
        s, v = extract_eval_series(lines, "fraction_exact_match_using_boxed")
        axes[1].plot(s, v, "o-", label=label, color=color, markersize=6)

    axes[0].set_title("Math Hard — Train Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Math Hard — Eval Exact Match (boxed)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Exact Match")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "math_hard_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/math_hard_comparison.png")


# ============================================================
# Plot 3: Math Hard — KL divergence comparison
# ============================================================
def plot_math_hard_kl():
    fig, ax = plt.subplots(figsize=(8, 4))

    for run, label, color in [
        ("math_hard_reinforce", "REINFORCE", "tab:orange"),
        ("math_hard_grpo", "GRPO", "tab:blue"),
    ]:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "approximate_kl")
        ax.plot(s, v, label=label, color=color, alpha=0.7)

    ax.set_title("Math Hard — KL Divergence (policy vs reference)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Approx KL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "math_hard_kl.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/math_hard_kl.png")


# ============================================================
# Plot 4: GRPO Ablations — Reward curves
# ============================================================
def plot_ablation_reward():
    fig, ax = plt.subplots(figsize=(10, 5))

    ablations = [
        ("format_copy_grpo", "Baseline (default)", "black", 2.0),
        ("ablation_ppo_epochs_4", "ppo_epochs=4", "tab:red", 1.2),
        ("ablation_kl_coef_0.01", "kl_coef=0.01", "tab:green", 1.2),
        ("ablation_kl_coef_0.2", "kl_coef=0.2", "tab:purple", 1.2),
        ("ablation_clip_eps_0.1", "clip_eps=0.1", "tab:cyan", 1.2),
        ("ablation_grad_accum_1_mb8", "mb=8, grad_accum=1", "tab:brown", 1.2),
    ]

    for run, label, color, lw in ablations:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "mean_total_reward")
        ax.plot(s, v, label=label, color=color, alpha=0.8, linewidth=lw)

    ax.set_title("GRPO Ablations on Format Copy — Train Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "ablation_reward.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/ablation_reward.png")


# ============================================================
# Plot 5: GRPO Ablations — KL divergence curves
# ============================================================
def plot_ablation_kl():
    fig, ax = plt.subplots(figsize=(10, 5))

    ablations = [
        ("format_copy_grpo", "Baseline (default)", "black", 2.0),
        ("ablation_ppo_epochs_4", "ppo_epochs=4", "tab:red", 1.2),
        ("ablation_kl_coef_0.01", "kl_coef=0.01", "tab:green", 1.2),
        ("ablation_kl_coef_0.2", "kl_coef=0.2", "tab:purple", 1.2),
        ("ablation_clip_eps_0.1", "clip_eps=0.1", "tab:cyan", 1.2),
        ("ablation_grad_accum_1_mb8", "mb=8, grad_accum=1", "tab:brown", 1.2),
    ]

    for run, label, color, lw in ablations:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "approximate_kl")
        ax.plot(s, v, label=label, color=color, alpha=0.8, linewidth=lw)

    ax.set_title("GRPO Ablations on Format Copy — KL Divergence")
    ax.set_xlabel("Step")
    ax.set_ylabel("Approx KL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "ablation_kl.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/ablation_kl.png")


# ============================================================
# Plot 6: GRPO Ablations — Clip fraction (GRPO-specific)
# ============================================================
def plot_ablation_clipfrac():
    fig, ax = plt.subplots(figsize=(10, 5))

    ablations = [
        ("format_copy_grpo", "Baseline (default)", "black", 2.0),
        ("ablation_ppo_epochs_4", "ppo_epochs=4", "tab:red", 1.2),
        ("ablation_kl_coef_0.01", "kl_coef=0.01", "tab:green", 1.2),
        ("ablation_kl_coef_0.2", "kl_coef=0.2", "tab:purple", 1.2),
        ("ablation_clip_eps_0.1", "clip_eps=0.1", "tab:cyan", 1.2),
        ("ablation_grad_accum_1_mb8", "mb=8, grad_accum=1", "tab:brown", 1.2),
    ]

    for run, label, color, lw in ablations:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "clip")
        if len(s) > 0:
            ax.plot(s, v, label=label, color=color, alpha=0.8, linewidth=lw)

    ax.set_title("GRPO Ablations on Format Copy — Clip Fraction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Clip Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "ablation_clipfrac.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/ablation_clipfrac.png")


# ============================================================
# Plot 7: Math Hard — Completion length over training
# ============================================================
def plot_math_hard_completion_length():
    fig, ax = plt.subplots(figsize=(8, 4))

    for run, label, color in [
        ("math_hard_reinforce", "REINFORCE", "tab:orange"),
        ("math_hard_grpo", "GRPO", "tab:blue"),
    ]:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "mean_generated_completion_token_count")
        ax.plot(s, v, label=label, color=color, alpha=0.7)

    ax.set_title("Math Hard — Mean Completion Length")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "math_hard_completion_length.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/math_hard_completion_length.png")


# ============================================================
# Plot 8: Math Hard — Boxed answer fraction
# ============================================================
def plot_math_hard_boxed():
    fig, ax = plt.subplots(figsize=(8, 4))

    for run, label, color in [
        ("math_hard_reinforce", "REINFORCE", "tab:orange"),
        ("math_hard_grpo", "GRPO", "tab:blue"),
    ]:
        lines = load_metrics(run)
        s, v = extract_train_series(lines, "completion_contains_boxed_answer_pattern")
        if len(s) > 0:
            ax.plot(s, v, label=label, color=color, alpha=0.7)

    ax.set_title("Math Hard — Fraction of Completions with \\\\boxed{} Answer")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "math_hard_boxed.png"), dpi=150)
    plt.close(fig)
    print("Saved: plots/math_hard_boxed.png")


if __name__ == "__main__":
    print("Generating plots from metrics.jsonl files...\n")
    plot_format_copy_comparison()
    plot_math_hard_comparison()
    plot_math_hard_kl()
    plot_ablation_reward()
    plot_ablation_kl()
    plot_ablation_clipfrac()
    plot_math_hard_completion_length()
    plot_math_hard_boxed()
    print(f"\nAll plots saved to: {os.path.abspath(PLOTS_DIR)}/")
    print("Include these in your report PDF.")
