"""
Plotting script for HW2 deliverables.
Usage:
    python plot.py                  # generate all plots
    python plot.py --section 3      # only Section 3 (CartPole)
    python plot.py --section 4      # only Section 4 (HalfCheetah)
    python plot.py --section 5      # only Section 5 (LunarLander)
    python plot.py --section 6      # only Section 6 (InvertedPendulum)
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.join(os.path.dirname(__file__), "exp")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def find_log(prefix):
    """Find the most recent log.csv matching a given experiment prefix."""
    pattern = os.path.join(EXP_DIR, f"{prefix}*", "log.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  WARNING: No log found for prefix '{prefix}'")
        return None
    return matches[-1]  # most recent


def load_run(prefix, x_col="Train_EnvstepsSoFar", y_col="Eval_AverageReturn"):
    """Load a single run's data."""
    path = find_log(prefix)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    return df[x_col].values, df[y_col].values


def plot_runs(prefixes, labels, title, xlabel, ylabel, save_path):
    """Plot multiple runs on a single figure."""
    plt.figure(figsize=(10, 6))
    for prefix, label in zip(prefixes, labels):
        x, y = load_run(prefix, x_col=xlabel if xlabel == "Train_EnvstepsSoFar" else "Train_EnvstepsSoFar", y_col=ylabel)
        if x is not None:
            plt.plot(x, y, label=label)
    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel(ylabel.replace("_", " "))
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_section3():
    """Experiment 1: CartPole — small batch and large batch comparisons."""
    print("\n=== Section 3: CartPole ===")

    # Small batch (b=1000)
    prefixes = [
        "CartPole-v0_cartpole_sd",
        "CartPole-v0_cartpole_rtg_sd",
        "CartPole-v0_cartpole_na_sd",
        "CartPole-v0_cartpole_rtg_na_sd",
    ]
    labels = [
        "vanilla",
        "rtg",
        "na",
        "rtg + na",
    ]
    plot_runs(prefixes, labels,
              "CartPole — Small Batch (b=1000)",
              "Train_EnvstepsSoFar", "Eval_AverageReturn",
              os.path.join(FIG_DIR, "cartpole_small_batch.png"))

    # Large batch (b=4000)
    prefixes = [
        "CartPole-v0_cartpole_lb_sd",
        "CartPole-v0_cartpole_lb_rtg_sd",
        "CartPole-v0_cartpole_lb_na_sd",
        "CartPole-v0_cartpole_lb_rtg_na_sd",
    ]
    labels = [
        "vanilla",
        "rtg",
        "na",
        "rtg + na",
    ]
    plot_runs(prefixes, labels,
              "CartPole — Large Batch (b=4000)",
              "Train_EnvstepsSoFar", "Eval_AverageReturn",
              os.path.join(FIG_DIR, "cartpole_large_batch.png"))


def plot_section4():
    """Experiment 2: HalfCheetah — baseline vs no baseline."""
    print("\n=== Section 4: HalfCheetah ===")

    # Eval return comparison
    prefixes = [
        "HalfCheetah-v4_cheetah_sd",
        "HalfCheetah-v4_cheetah_baseline_sd",
    ]
    labels = [
        "no baseline",
        "baseline",
    ]
    plot_runs(prefixes, labels,
              "HalfCheetah — Eval Return",
              "Train_EnvstepsSoFar", "Eval_AverageReturn",
              os.path.join(FIG_DIR, "cheetah_eval_return.png"))

    # Baseline loss (only for the baseline run)
    path = find_log("HalfCheetah-v4_cheetah_baseline_sd")
    if path:
        df = pd.read_csv(path)
        if "Baseline Loss" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Train_EnvstepsSoFar"], df["Baseline Loss"])
            plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
            plt.ylabel("Baseline Loss")
            plt.title("HalfCheetah — Baseline Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(FIG_DIR, "cheetah_baseline_loss.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Saved: {save_path}")
        else:
            print("  WARNING: 'Baseline Loss' column not found. Available columns:", list(df.columns))


def plot_section5():
    """Experiment 3: LunarLander — GAE lambda comparison."""
    print("\n=== Section 5: LunarLander ===")

    prefixes = [
        "LunarLander-v2_lunar_lander_lambda0_sd",
        "LunarLander-v2_lunar_lander_lambda0_95_sd",
        "LunarLander-v2_lunar_lander_lambda0_98_sd",
        "LunarLander-v2_lunar_lander_lambda0_99_sd",
        "LunarLander-v2_lunar_lander_lambda1_sd",
    ]
    labels = [
        "lambda=0",
        "lambda=0.95",
        "lambda=0.98",
        "lambda=0.99",
        "lambda=1",
    ]
    plot_runs(prefixes, labels,
              "LunarLander — GAE Lambda Comparison",
              "Train_EnvstepsSoFar", "Eval_AverageReturn",
              os.path.join(FIG_DIR, "lunar_lander_lambda_comparison.png"))


def plot_section6():
    """Experiment 4: InvertedPendulum — default vs best tuned run."""
    print("\n=== Section 6: InvertedPendulum ===")

    # Plot 1: Clean comparison — default vs best tuned run
    # Change "pendulum_b500_lr02" to your best run if different
    best_prefix = "InvertedPendulum-v4_pendulum_b500_lr02_sd"
    default_prefix = "InvertedPendulum-v4_pendulum_default_sd"

    prefixes = [default_prefix, best_prefix]
    labels = ["default (b=5000)", "tuned (b=500, rtg, na, lr=0.02)"]

    plt.figure(figsize=(10, 6))
    for prefix, label in zip(prefixes, labels):
        x, y = load_run(prefix)
        if x is not None:
            plt.plot(x, y, label=label, linewidth=2)

    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='target (1000)')
    plt.axvline(x=100000, color='gray', linestyle=':', alpha=0.5, label='100K step budget')
    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval Average Return")
    plt.title("InvertedPendulum — Default vs Tuned")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "pendulum_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    # Plot 2: All tuning runs (for reference)
    pattern = os.path.join(EXP_DIR, "InvertedPendulum-v4_pendulum*", "log.csv")
    matches = sorted(glob.glob(pattern))

    if not matches:
        print("  WARNING: No InvertedPendulum runs found")
        return

    plt.figure(figsize=(10, 6))
    for path in matches:
        df = pd.read_csv(path)
        exp_name = os.path.basename(os.path.dirname(path))
        label = exp_name.split("InvertedPendulum-v4_")[1].rsplit("_sd", 1)[0]
        plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=label)

    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=100000, color='gray', linestyle=':', alpha=0.5)
    plt.xlim(0, 150000)  # zoom into the 100K region
    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval Average Return")
    plt.title("InvertedPendulum — All Tuning Runs (zoomed to 150K steps)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "pendulum_all_runs.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, default=None, help="Plot only a specific section (3, 4, 5, or 6)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    if args.section is None or args.section == 3:
        plot_section3()
    if args.section is None or args.section == 4:
        plot_section4()
    if args.section is None or args.section == 5:
        plot_section5()
    if args.section is None or args.section == 6:
        plot_section6()

    print("\nDone! Check the 'figures/' directory.")
