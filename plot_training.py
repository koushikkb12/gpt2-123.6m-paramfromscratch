"""
plot_training.py - Plot training curves from log files.

Usage:
    python plot_training.py                                      # Plot pre-training
    python plot_training.py --log_dir checkpoints/finetune       # Plot fine-tuning
    python plot_training.py --save                               # Save plots as PNG

Creates these plots:
    1. Training loss over steps
    2. Validation loss + perplexity over steps
    3. Learning rate schedule
    4. Throughput (tokens/sec) over time
    5. Combined overview

Great for documentation and HuggingFace model cards.
"""

import os
import csv
import argparse
import sys

def read_csv(path):
    """Read CSV file into list of dicts."""
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_training(log_dir: str, save: bool = False):
    """Generate training plots."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Falling back to text-based summary...\n")
        text_summary(log_dir)
        return

    # Read data
    train_data = read_csv(os.path.join(log_dir, "training_log.csv"))
    eval_data = read_csv(os.path.join(log_dir, "eval_log.csv"))

    if not train_data:
        print(f"No training data found in {log_dir}/")
        return

    # Parse values
    steps = [int(r["step"]) for r in train_data]
    losses = [float(r["loss"]) for r in train_data]
    smooth_losses = [float(r["smooth_loss"]) for r in train_data]
    lrs = [float(r["lr"]) for r in train_data]
    tok_per_sec = [float(r["tokens_per_sec"]) for r in train_data if r["tokens_per_sec"]]
    grad_norms = [float(r["grad_norm"]) for r in train_data if r.get("grad_norm")]

    eval_steps = [int(r["step"]) for r in eval_data] if eval_data else []
    eval_losses = [float(r["val_loss"]) for r in eval_data] if eval_data else []
    eval_ppls = [float(r["val_perplexity"]) for r in eval_data] if eval_data else []

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("GPT-2 Training Progress", fontsize=16, fontweight="bold")

    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(steps, losses, alpha=0.3, color="steelblue", linewidth=0.5, label="Raw loss")
    ax.plot(steps, smooth_losses, color="darkblue", linewidth=1.5, label="Smoothed loss")
    if eval_steps:
        ax.scatter(eval_steps, eval_losses, color="red", s=40, zorder=5, label="Val loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))

    # 2. Validation Perplexity
    ax = axes[0, 1]
    if eval_steps:
        ax.plot(eval_steps, eval_ppls, "ro-", markersize=4, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Validation Perplexity")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))
    else:
        ax.text(0.5, 0.5, "No eval data yet", ha="center", va="center", fontsize=14)
        ax.set_title("Validation Perplexity")

    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(steps, lrs, color="green", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))

    # 4. Gradient Norm / Throughput
    ax = axes[1, 1]
    if grad_norms and len(grad_norms) > 10:
        ax.plot(steps[:len(grad_norms)], grad_norms, alpha=0.5, color="orange", linewidth=0.5)
        # Smoothed
        window = min(50, len(grad_norms) // 4)
        if window > 1:
            smoothed_gn = [sum(grad_norms[max(0,i-window):i+1])/len(grad_norms[max(0,i-window):i+1])
                          for i in range(len(grad_norms))]
            ax.plot(steps[:len(smoothed_gn)], smoothed_gn, color="darkorange", linewidth=1.5, label="Smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
        ax.legend()
    elif tok_per_sec:
        ax.plot(steps[:len(tok_per_sec)], tok_per_sec, color="purple", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Training Throughput")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))

    plt.tight_layout()

    if save:
        save_path = os.path.join(log_dir, "training_plots.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plots saved: {save_path}")
    else:
        plt.show()

    plt.close()


def text_summary(log_dir: str):
    """Print a text-based summary when matplotlib is not available."""
    train_data = read_csv(os.path.join(log_dir, "training_log.csv"))
    eval_data = read_csv(os.path.join(log_dir, "eval_log.csv"))

    if not train_data:
        print("No training data found.")
        return

    losses = [float(r["loss"]) for r in train_data]
    steps = [int(r["step"]) for r in train_data]

    print(f"Training Progress ({len(train_data)} logged steps)")
    print(f"  Steps: {steps[0]} to {steps[-1]}")
    print(f"  Loss:  {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Min loss: {min(losses):.4f}")
    print(f"  Avg (first 10): {sum(losses[:10])/max(len(losses[:10]),1):.4f}")
    print(f"  Avg (last 10):  {sum(losses[-10:])/max(len(losses[-10:]),1):.4f}")

    if eval_data:
        eval_losses = [float(r["val_loss"]) for r in eval_data]
        print(f"\nValidation ({len(eval_data)} evals)")
        print(f"  Best val loss: {min(eval_losses):.4f}")
        print(f"  Last val loss: {eval_losses[-1]:.4f}")

    # ASCII loss chart
    print(f"\nLoss curve (text approximation):")
    n_points = min(50, len(losses))
    step_size = len(losses) // n_points
    sampled = [losses[i * step_size] for i in range(n_points)]
    max_loss = max(sampled)
    min_loss_val = min(sampled)
    chart_width = 40

    for i, loss in enumerate(sampled):
        if max_loss > min_loss_val:
            bar_len = int((loss - min_loss_val) / (max_loss - min_loss_val) * chart_width)
        else:
            bar_len = chart_width // 2
        bar = "#" * bar_len
        step_label = steps[i * step_size]
        print(f"  {step_label:>7d} |{bar} {loss:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot GPT-2 training curves")
    parser.add_argument("--log_dir", type=str, default="checkpoints/pretrain",
                        help="Directory containing log files")
    parser.add_argument("--save", action="store_true",
                        help="Save plots as PNG instead of displaying")
    args = parser.parse_args()

    print(f"\nPlotting training curves from: {args.log_dir}/")
    plot_training(args.log_dir, save=args.save)


if __name__ == "__main__":
    main()
