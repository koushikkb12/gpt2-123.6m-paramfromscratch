"""
utils.py - Shared utilities for training GPT-2 from scratch.

Includes:
  - Comprehensive logging (JSONL + CSV + console)
  - Training metrics tracking (smoothed loss, throughput, grad norms)
  - VRAM monitoring
  - Checkpoint save/load helpers
  - Text generation helper
  - Learning rate scheduler
  - Training summary report generator
  - Loss curve plotting
"""

import os
import csv
import time
import math
import json
import torch
import tiktoken
from datetime import datetime, timedelta


class TrainingLogger:
    """
    Comprehensive training logger that saves everything for documentation.
    
    Creates these files in the log directory:
      - training_log.jsonl   : Detailed per-step metrics (machine readable)
      - training_log.csv     : Same data in CSV (easy to plot in Excel/Sheets/pandas)
      - eval_log.csv         : Validation metrics
      - samples.txt          : Generated text samples during training
      - training_summary.md  : Human-readable summary (updated at end)
    """

    def __init__(self, log_dir: str):
        self.start_time = time.time()
        self.step_times = []
        self.losses = []
        self.eval_losses = []
        self.log_dir = log_dir
        self.samples = []

        os.makedirs(log_dir, exist_ok=True)

        # JSONL log (append mode, one JSON object per line)
        self.jsonl_path = os.path.join(log_dir, "training_log.jsonl")

        # CSV log for easy plotting
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self._init_csv(self.csv_path, [
            "step", "loss", "smooth_loss", "lr", "tokens_seen",
            "tokens_per_sec", "grad_norm", "elapsed_sec",
            "vram_used_gb", "vram_peak_gb", "timestamp"
        ])

        # Eval CSV
        self.eval_csv_path = os.path.join(log_dir, "eval_log.csv")
        self._init_csv(self.eval_csv_path, [
            "step", "val_loss", "val_perplexity", "best_val_loss",
            "tokens_seen", "elapsed_sec", "timestamp"
        ])

        # Samples file
        self.samples_path = os.path.join(log_dir, "samples.txt")

        print(f"  Logging to: {log_dir}/")
        print(f"    - training_log.jsonl  (detailed step logs)")
        print(f"    - training_log.csv    (for plotting)")
        print(f"    - eval_log.csv        (validation results)")
        print(f"    - samples.txt         (generated text samples)")

    def _init_csv(self, path, headers):
        """Initialize a CSV file with headers (only if file doesn't exist)."""
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, step: int, loss: float, lr: float, tokens_seen: int,
            total_steps: int, grad_norm: float = None, extra: dict = None):
        """Log a training step to console, JSONL, and CSV."""
        now = time.time()
        elapsed = now - self.start_time
        self.losses.append(loss)

        # Calculate throughput
        if len(self.step_times) > 0:
            dt = now - self.step_times[-1]
            tokens_per_step = extra.get("tokens_per_step", 0) if extra else 0
            tokens_per_sec = (tokens_per_step / dt) if dt > 0 else 0
        else:
            tokens_per_sec = 0
            dt = 0
        self.step_times.append(now)

        # ETA
        steps_done = step
        steps_remaining = total_steps - steps_done
        if steps_done > 0:
            avg_step_time = elapsed / steps_done
            eta_seconds = steps_remaining * avg_step_time
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "N/A"

        # Smoothed loss (last 100 logged steps)
        recent_losses = self.losses[-100:]
        smooth_loss = sum(recent_losses) / len(recent_losses)

        # VRAM info
        vram_used = 0.0
        vram_peak = 0.0
        vram_str = ""
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_str = f" | VRAM: {vram_used:.1f}/{vram_total:.1f}GB"

        # Grad norm string
        grad_str = ""
        if grad_norm is not None:
            grad_str = f" | gnorm {grad_norm:.2f}"

        # Console output
        msg = (
            f"step {step:>6d}/{total_steps} | "
            f"loss {loss:.4f} (avg {smooth_loss:.4f}) | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_seen/1e6:.1f}M | "
            f"{tokens_per_sec:.0f} tok/s | "
            f"ETA {eta_str}{grad_str}{vram_str}"
        )
        if extra:
            for k, v in extra.items():
                if k != "tokens_per_step":
                    msg += f" | {k}: {v}"
        print(msg)

        # JSONL log
        log_entry = {
            "step": step,
            "loss": round(loss, 6),
            "smooth_loss": round(smooth_loss, 6),
            "lr": lr,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
            "elapsed_sec": round(elapsed, 1),
            "vram_used_gb": round(vram_used, 2),
            "vram_peak_gb": round(vram_peak, 2),
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            for k, v in extra.items():
                if k != "tokens_per_step":
                    log_entry[k] = v
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # CSV log
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, round(loss, 6), round(smooth_loss, 6), lr, tokens_seen,
                round(tokens_per_sec, 1),
                round(grad_norm, 4) if grad_norm is not None else "",
                round(elapsed, 1),
                round(vram_used, 2), round(vram_peak, 2),
                datetime.now().isoformat()
            ])

    def log_eval(self, step: int, val_loss: float, val_perplexity: float,
                 best_val_loss: float = None, tokens_seen: int = 0):
        """Log evaluation results to console, CSV, and JSONL."""
        elapsed = time.time() - self.start_time
        self.eval_losses.append({"step": step, "val_loss": val_loss, "val_ppl": val_perplexity})

        print(f"  >>> EVAL step {step}: val_loss={val_loss:.4f}, "
              f"perplexity={val_perplexity:.2f}"
              + (f", best={best_val_loss:.4f}" if best_val_loss is not None else ""))

        # Eval CSV
        with open(self.eval_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, round(val_loss, 6), round(val_perplexity, 2),
                round(best_val_loss, 6) if best_val_loss is not None else "",
                tokens_seen, round(elapsed, 1),
                datetime.now().isoformat()
            ])

        # Also append to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({
                "type": "eval",
                "step": step,
                "val_loss": round(val_loss, 6),
                "val_perplexity": round(val_perplexity, 2),
                "best_val_loss": round(best_val_loss, 6) if best_val_loss else None,
                "tokens_seen": tokens_seen,
                "elapsed_sec": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

    def log_sample(self, step: int, prompt: str, generated: str):
        """Save a generated text sample."""
        self.samples.append({"step": step, "prompt": prompt, "text": generated})

        with open(self.samples_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Step {step} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"{'='*60}\n")
            f.write(generated[:500])
            f.write(f"\n{'='*60}\n")

    def save_summary(self, model_config, train_config, total_steps: int,
                     tokens_seen: int, best_val_loss: float, device_name: str = ""):
        """
        Generate a training_summary.md for documentation.
        This is the file you can reference when writing about your project.
        """
        elapsed = time.time() - self.start_time
        summary_path = os.path.join(self.log_dir, "training_summary.md")

        # Compute stats
        avg_loss_first = sum(l for l in self.losses[:100]) / max(len(self.losses[:100]), 1)
        avg_loss_last = sum(l for l in self.losses[-100:]) / max(len(self.losses[-100:]), 1)
        min_loss = min(self.losses) if self.losses else float("inf")

        avg_tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

        with open(summary_path, "w") as f:
            f.write("# GPT-2 Training Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration**: {format_time(elapsed)}\n")
            f.write(f"**Device**: {device_name}\n\n")

            f.write("## Model Architecture\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|---|---|\n")
            f.write(f"| Layers | {model_config.n_layer} |\n")
            f.write(f"| Attention Heads | {model_config.n_head} |\n")
            f.write(f"| Hidden Dim | {model_config.n_embd} |\n")
            f.write(f"| Vocab Size | {model_config.vocab_size:,} |\n")
            f.write(f"| Max Seq Length | {model_config.block_size} |\n")
            f.write(f"| Dropout | {model_config.dropout} |\n\n")

            f.write("## Training Hyperparameters\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|---|---|\n")
            f.write(f"| Total Steps | {total_steps:,} |\n")
            f.write(f"| Tokens Trained | {tokens_seen/1e9:.2f}B |\n")
            f.write(f"| Peak LR | {train_config.learning_rate} |\n")
            f.write(f"| Min LR | {train_config.min_lr} |\n")
            f.write(f"| Warmup Steps | {train_config.warmup_steps:,} |\n")
            f.write(f"| Weight Decay | {train_config.weight_decay} |\n")
            f.write(f"| Micro Batch Size | {train_config.micro_batch_size} |\n")
            f.write(f"| Grad Accumulation | {train_config.gradient_accumulation_steps} |\n")
            f.write(f"| Effective Batch | {train_config.effective_batch_size} |\n")
            f.write(f"| Gradient Clip | {train_config.grad_clip} |\n\n")

            f.write("## Results\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|---|---|\n")
            f.write(f"| Initial Loss (avg first 100) | {avg_loss_first:.4f} |\n")
            f.write(f"| Final Loss (avg last 100) | {avg_loss_last:.4f} |\n")
            f.write(f"| Min Training Loss | {min_loss:.4f} |\n")
            f.write(f"| Best Val Loss | {best_val_loss:.4f} |\n")
            f.write(f"| Best Val Perplexity | {math.exp(min(best_val_loss, 20)):.2f} |\n")
            f.write(f"| Avg Throughput | {avg_tok_per_sec:.0f} tokens/sec |\n")
            f.write(f"| Total Training Time | {format_time(elapsed)} |\n\n")

            if self.eval_losses:
                f.write("## Validation History\n\n")
                f.write("| Step | Val Loss | Perplexity |\n")
                f.write("|---|---|---|\n")
                for entry in self.eval_losses:
                    f.write(f"| {entry['step']:,} | {entry['val_loss']:.4f} | {entry['val_ppl']:.2f} |\n")
                f.write("\n")

            if self.samples:
                f.write("## Generated Samples\n\n")
                for s in self.samples[-5:]:  # Last 5 samples
                    f.write(f"### Step {s['step']:,}\n")
                    f.write(f"**Prompt**: {s['prompt']}\n\n")
                    f.write(f"```\n{s['text'][:300]}\n```\n\n")

            f.write("## Log Files\n\n")
            f.write(f"- `training_log.csv` - Per-step metrics (for plotting)\n")
            f.write(f"- `training_log.jsonl` - Detailed logs (machine readable)\n")
            f.write(f"- `eval_log.csv` - Validation results\n")
            f.write(f"- `samples.txt` - All generated text samples\n")

        print(f"\n  Training summary saved: {summary_path}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m}m"


def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """
    Learning rate schedule: linear warmup + cosine decay.
    
    1. Linear warmup from 0 to max_lr over warmup_steps
    2. Cosine decay from max_lr to min_lr over remaining steps
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After max_steps, return min_lr
    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(
    model, optimizer, scheduler, scaler, step, val_loss,
    config, path, is_best=False
):
    """Save a full training checkpoint for resumption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
        "val_loss": val_loss,
        "config": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "dropout": config.dropout,
            "bias": config.bias,
        }
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path} (step {step}, val_loss={val_loss:.4f})")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pt")
        torch.save(checkpoint, best_path)
        print(f"  Best model saved: {best_path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load a checkpoint and return the step number."""
    print(f"  Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"  Resumed from step {step}, val_loss={val_loss:.4f}")
    return step, val_loss


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"  Found latest checkpoint: {latest}")
    return latest


def get_tokenizer():
    """Get the GPT-2 BPE tokenizer via tiktoken."""
    return tiktoken.get_encoding("gpt2")


def generate_text(
    model, tokenizer, prompt: str, max_tokens: int = 200,
    temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model.generate(
            idx, max_new_tokens=max_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
    
    generated = output[0].tolist()
    text = tokenizer.decode(generated)
    model.train()
    return text
