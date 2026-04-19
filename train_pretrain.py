"""
train_pretrain.py - Pre-training GPT-2 on OpenWebText.

Usage:
    python train_pretrain.py                    # Start fresh
    python train_pretrain.py --resume           # Resume from latest checkpoint
    python train_pretrain.py --max_steps 1000   # Override max steps

Features:
    - Mixed precision (FP16) training for T4 efficiency
    - Gradient accumulation for large effective batch sizes
    - Cosine LR schedule with linear warmup
    - Periodic validation, checkpointing, and sample generation
    - Auto-resume from latest checkpoint
    - Comprehensive logging: JSONL, CSV, samples, and training summary
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F

from config import ModelConfig, PretrainConfig
from model import GPT2
from dataset import PretrainDataset, ValidationDataset, create_pretrain_dataloader
from utils import (
    TrainingLogger, get_lr, save_checkpoint, load_checkpoint,
    find_latest_checkpoint, generate_text, get_tokenizer
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train GPT-2 on OpenWebText")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Override micro batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---- Configs ----
    model_config = ModelConfig()
    train_config = PretrainConfig()
    
    # Apply overrides
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.micro_batch_size:
        train_config.micro_batch_size = args.micro_batch_size
    if args.grad_accum:
        train_config.gradient_accumulation_steps = args.grad_accum
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    
    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_amp = device == "cuda"
    device_name = ""
    print(f"\nDevice: {device}")
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU: {device_name}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        
        # CUDA performance optimizations
        torch.backends.cudnn.benchmark = True   # Auto-tune convolution algorithms (stable shapes)
        torch.backends.cuda.matmul.allow_tf32 = False  # T4 doesn't support TF32 (Ampere+)
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        print("  CUDA optimizations enabled: cudnn.benchmark=True")
    
    # ---- Model ----
    print(f"\nInitializing GPT-2 Small...")
    model = GPT2(model_config).to(device)
    
    # Compile disabled: causes massive throughput drop on T4 (compute 7.5) due to unoptimized kernels
    # if device == "cuda":
    #     print("  Compiling model with torch.compile (this may take ~30s on first run)...")
    #     model = torch.compile(model)
    
    # ---- Optimizer ----
    # Separate weight decay for different parameter types (GPT-2/3 convention)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:  # Weight matrices
                decay_params.append(param)
            else:  # Biases, LayerNorm parameters
                no_decay_params.append(param)
    
    print(f"  Decay params: {sum(p.numel() for p in decay_params)/1e6:.1f}M")
    print(f"  No-decay params: {sum(p.numel() for p in no_decay_params)/1e6:.1f}M")
    
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2),
       eps=train_config.eps)
    
    # ---- Mixed precision ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # ---- Checkpoint resume ----
    start_step = 0
    best_val_loss = float("inf")
    
    if args.resume:
        ckpt_path = find_latest_checkpoint(train_config.checkpoint_dir)
        if ckpt_path:
            start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scaler)
            start_step += 1  # Start from next step
        else:
            print("No checkpoint found, starting fresh.")
    
    # ---- Data ----
    print(f"\nSetting up data streaming...")
    print(f"  Dataset: {train_config.dataset_name}")
    print(f"  Effective batch size: {train_config.effective_batch_size} sequences")
    print(f"  Tokens per step: {train_config.tokens_per_step:,}")
    print(f"  Total steps: {train_config.max_steps:,}")
    print(f"  Total tokens: ~{train_config.max_steps * train_config.tokens_per_step / 1e9:.1f}B")
    
    train_loader = create_pretrain_dataloader(train_config)
    train_iter = iter(train_loader)
    
    # Validation dataset (pre-loaded into memory)
    print("Loading validation data...")
    val_dataset = ValidationDataset(
        dataset_name=train_config.dataset_name,
        block_size=model_config.block_size,
        num_tokens=500_000,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_config.micro_batch_size, shuffle=False
    )
    
    # ---- Tokenizer for generation ----
    tokenizer = get_tokenizer()
    
    # ---- Logger ----
    logger = TrainingLogger(log_dir=train_config.checkpoint_dir)
    
    # Sample prompts for periodic generation
    sample_prompts = [
        "The meaning of life is",
        "In a galaxy far far away",
        "The president of the United States",
        "Once upon a time there was",
        "The best way to learn programming is",
    ]
    
    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Starting pre-training from step {start_step}...")
    print(f"  Logging: {train_config.checkpoint_dir}/")
    print(f"  Checkpoints every {train_config.checkpoint_interval} steps")
    print(f"  Eval every {train_config.eval_interval} steps")
    print(f"  Samples every {train_config.sample_interval} steps")
    print(f"{'='*60}\n")
    
    model.train()
    tokens_seen = start_step * train_config.tokens_per_step
    
    for step in range(start_step, train_config.max_steps):
        step_start = time.time()
        
        # Update learning rate
        lr = get_lr(
            step, train_config.warmup_steps, train_config.max_steps,
            train_config.learning_rate, train_config.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # ---- Gradient accumulation loop ----
        optimizer.zero_grad()
        total_loss = 0.0
        
        for micro_step in range(train_config.gradient_accumulation_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                # Reset data iterator (new epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                logits, loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps
            
            total_loss += loss.item()
            
            # Backward pass
            scaler.scale(loss).backward()
        
        # Gradient clipping and norm tracking
        grad_norm = None
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            ).item()
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        tokens_seen += train_config.tokens_per_step
        
        # ---- Logging ----
        if step % train_config.log_interval == 0:
            logger.log(
                step=step, loss=total_loss, lr=lr,
                tokens_seen=tokens_seen,
                total_steps=train_config.max_steps,
                grad_norm=grad_norm,
                extra={"tokens_per_step": train_config.tokens_per_step}
            )
        
        # ---- Evaluation ----
        if step > 0 and step % train_config.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= train_config.eval_steps:
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                        _, vloss = model(vx, vy)
                    val_losses.append(vloss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(avg_val_loss, 20))  # Cap to avoid overflow
            logger.log_eval(step, avg_val_loss, val_ppl,
                          best_val_loss=best_val_loss, tokens_seen=tokens_seen)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    model, optimizer, None, scaler, step, avg_val_loss,
                    model_config,
                    os.path.join(train_config.checkpoint_dir, f"step_{step}.pt"),
                    is_best=True
                )
            
            model.train()
        
        # ---- Periodic checkpoint ----
        if step > 0 and step % train_config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, None, scaler, step, best_val_loss,
                model_config,
                os.path.join(train_config.checkpoint_dir, f"step_{step}.pt"),
            )
        
        # ---- Sample generation ----
        if step > 0 and step % train_config.sample_interval == 0:
            model.eval()
            prompt = sample_prompts[step // train_config.sample_interval % len(sample_prompts)]
            sample = generate_text(
                model, tokenizer,
                prompt=prompt,
                max_tokens=150, temperature=0.8, top_k=50,
                device=device,
            )
            print(f"  >>> Sample (step {step}):")
            print(f"  {sample[:300]}")
            print()
            
            # Save sample to log
            logger.log_sample(step, prompt, sample)
            model.train()
    
    # ---- Final save ----
    save_checkpoint(
        model, optimizer, None, scaler, train_config.max_steps, best_val_loss,
        model_config,
        os.path.join(train_config.checkpoint_dir, f"step_{train_config.max_steps}.pt"),
        is_best=True,
    )
    
    # ---- Training summary ----
    logger.save_summary(
        model_config=model_config,
        train_config=train_config,
        total_steps=train_config.max_steps,
        tokens_seen=tokens_seen,
        best_val_loss=best_val_loss,
        device_name=device_name,
    )
    
    print(f"\n{'='*60}")
    print(f"Pre-training complete!")
    print(f"  Total steps: {train_config.max_steps:,}")
    print(f"  Total tokens: {tokens_seen/1e9:.2f}B")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val perplexity: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"  Checkpoints: {train_config.checkpoint_dir}/")
    print(f"  Logs: training_log.csv, eval_log.csv, samples.txt")
    print(f"  Summary: training_summary.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
