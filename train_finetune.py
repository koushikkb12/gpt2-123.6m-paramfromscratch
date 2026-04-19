"""
train_finetune.py - Fine-tune pre-trained GPT-2 for human conversation.

=== OVERVIEW ===
This script takes our pre-trained GPT-2 (which can predict the next word)
and teaches it to have conversations. Think of it like this:

  Pre-training = teaching a child to read and understand language
  Fine-tuning  = teaching that child how to have a conversation

=== HOW IT WORKS ===

1. LOAD THE BASE MODEL
   We load the pre-trained GPT-2 checkpoint (best.pt from pre-training).
   This model already understands English grammar, facts, and patterns
   from training on 1.3 billion tokens of web text.

2. LOAD CONVERSATION DATA
   We use two datasets:
   - OpenAssistant/oasst1: Real human-AI conversations (20K English paths)
   - yahma/alpaca-cleaned: 52K instruction-response pairs
   Total: ~72K training examples

3. LOSS MASKING
   The crucial difference from pre-training: we ONLY compute loss on
   the assistant's response tokens. The model sees the user's message
   (it needs context) but isn't penalized for not predicting user text.
   This is like teaching someone to respond to questions, not to ask them.

4. MULTI-EPOCH TRAINING
   Unlike pre-training (which sees each web page once), fine-tuning
   cycles through the same conversations many times. This is fine because:
   - The dataset is smaller (72K vs billions of tokens)
   - We use a much lower learning rate (2e-5 vs 6e-4)
   - We want the model to deeply learn conversation patterns

5. SAMPLE GENERATION
   Periodically, we generate a test conversation to visually monitor
   how the model's chat ability improves during training.

=== USAGE ===
    python train_finetune.py                                            # defaults
    python train_finetune.py --pretrained checkpoints/pretrain/best.pt  # specify checkpoint
    python train_finetune.py --resume                                   # resume from latest

=== KEY HYPERPARAMETERS (see config.py FinetuneConfig) ===
    learning_rate:  2e-5  (30x smaller than pre-training!)
    weight_decay:   0.01  (less regularization - we want to learn)
    micro_batch:    8     (sequences per forward pass)
    grad_accum:     2     (effective batch = 16 sequences)
    num_epochs:     8     (cycle through data multiple times)
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F

from config import ModelConfig, FinetuneConfig
from model import GPT2
from dataset_finetune import (
    create_finetune_datasets,
    create_finetune_dataloaders,
    get_tokenizer,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN,
)
from utils import (
    TrainingLogger, get_lr, save_checkpoint, load_checkpoint,
    find_latest_checkpoint, format_time,
)


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 for conversation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_finetune.py                              # Use defaults
  python train_finetune.py --pretrained path/to/best.pt # Custom checkpoint
  python train_finetune.py --resume                     # Resume training
        """,
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pre-trained checkpoint (default: checkpoints/pretrain/best.pt)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume fine-tuning from latest fine-tune checkpoint"
    )
    return parser.parse_args()


# ============================================================
# CONVERSATION SAMPLE GENERATION
# ============================================================

def generate_chat_sample(model, tokenizer, device, temperature=0.8, top_k=50):
    """
    Generate a test conversation to monitor training progress.

    This is like a "unit test" for chat quality. We give the model a
    user message and see if it can generate a coherent response.
    During early training, responses will be gibberish. As training
    progresses, they should become more coherent and relevant.

    We test with multiple prompts to see different capabilities:
    1. A simple greeting (basic conversation)
    2. A knowledge question (factual recall from pre-training)
    3. A creative task (instruction following from Alpaca)
    """
    model.eval()

    test_prompts = [
        "Hello! How are you doing today?",
        "What is machine learning?",
        "Write a short poem about the ocean.",
    ]

    results = []
    for prompt in test_prompts:
        # Format the prompt in our chat template
        # USER_TOKEN + newline + prompt + newline + ASSISTANT_TOKEN + newline
        formatted = USER_TOKEN + "\n" + prompt + "\n" + ASSISTANT_TOKEN + "\n"
        input_ids = tokenizer.encode_ordinary(formatted)
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Encode our end-of-turn token so we know when to stop
        end_ids = tokenizer.encode_ordinary(END_TOKEN)

        generated_tokens = []
        max_new_tokens = 200

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to block_size if needed
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

                # Get prediction for next token
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Top-k filtering: only consider top k most likely tokens
                # This prevents the model from choosing very unlikely tokens
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                idx = torch.cat([idx, next_token], dim=1)
                token_id = next_token.item()
                generated_tokens.append(token_id)

                # Check if model generated the end-of-turn token
                if len(generated_tokens) >= len(end_ids):
                    if generated_tokens[-len(end_ids):] == end_ids:
                        generated_tokens = generated_tokens[:-len(end_ids)]
                        break

                # Also stop at the standard EOT token
                if token_id == tokenizer.eot_token:
                    generated_tokens = generated_tokens[:-1]
                    break

        response = tokenizer.decode(generated_tokens).strip()
        results.append((prompt, response))

    model.train()
    return results


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, val_loader, device, dtype, use_amp, max_batches=50):
    """
    Evaluate the model on the validation set.

    Returns average loss over `max_batches` batches.

    Why limit batches?
    - Full evaluation takes too long and wastes GPU time
    - 50 batches (50 * 8 = 400 conversations) is enough
      to get a stable loss estimate
    - We evaluate frequently, so each eval should be fast
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                _, loss = model(x, y)
            losses.append(loss.item())

    model.train()

    if losses:
        return sum(losses) / len(losses)
    return float("inf")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    args = parse_args()

    # ---- Load configs ----
    model_config = ModelConfig()
    ft_config = FinetuneConfig()

    pretrained_path = args.pretrained or ft_config.pretrained_checkpoint

    # ---- Device setup ----
    # We use mixed precision (float16) on GPU for 2x speed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_amp = device == "cuda"

    print(f"\n{'=' * 60}")
    print(f"  GPT-2 CONVERSATIONAL FINE-TUNING")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram:.1f} GB)")

    # ---- Load pre-trained model ----
    print(f"\n  Loading pre-trained model: {pretrained_path}")

    # Load the checkpoint saved during pre-training
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config", {})

    # Apply saved config (ensures model architecture matches the checkpoint)
    for key, value in saved_config.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    # Create model and load pre-trained weights
    model = GPT2(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # === ENABLE DROPOUT FOR FINE-TUNING ===
    # Pre-training used dropout=0.0 for maximum throughput, but during
    # fine-tuning we cycle through a smaller dataset many times, so we
    # need dropout to prevent overfitting.
    # We manually set all Dropout modules to our fine-tuning dropout rate.
    if ft_config.dropout > 0:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = ft_config.dropout
        print(f"  Dropout enabled: {ft_config.dropout}")

    param_count = model.get_num_params() / 1e6
    print(f"  Model: {param_count:.1f}M parameters")
    print(f"  Pre-train val_loss: {checkpoint.get('val_loss', 'N/A')}")

    # ---- Load datasets ----
    tokenizer = get_tokenizer()
    train_dataset, val_dataset = create_finetune_datasets(block_size=model_config.block_size)
    train_loader, val_loader = create_finetune_dataloaders(
        train_dataset, val_dataset,
        micro_batch_size=ft_config.micro_batch_size,
    )

    # ---- Optimizer setup ----
    # We separate parameters into two groups:
    #   1. Weight matrices (2D tensors) -> apply weight decay
    #   2. Biases and norms (1D tensors) -> no weight decay
    #
    # WHY? Weight decay prevents large weights (regularization), but
    # applying it to biases/norms can hurt performance.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"\n  Optimizer groups: {n_decay:,} decay + {n_nodecay:,} no-decay params")

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": ft_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=ft_config.learning_rate, betas=(ft_config.beta1, ft_config.beta2),
       eps=ft_config.eps)

    # GradScaler for mixed precision training
    # Scales loss up before backward pass to prevent float16 underflow,
    # then scales gradients back down before optimizer step
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Resume from checkpoint ----
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt_path = find_latest_checkpoint(ft_config.checkpoint_dir)
        if ckpt_path:
            start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scaler)
            start_step += 1
            # Figure out which epoch we're in
            steps_per_epoch = len(train_loader) // ft_config.gradient_accumulation_steps
            if steps_per_epoch > 0:
                start_epoch = start_step // steps_per_epoch
            print(f"  Resuming from step {start_step}, epoch {start_epoch + 1}")

    # ---- Training setup ----
    os.makedirs(ft_config.checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(
        log_dir=ft_config.checkpoint_dir,
    )

    # Calculate training schedule
    steps_per_epoch = len(train_loader) // ft_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * ft_config.num_epochs

    print(f"\n{'=' * 60}")
    print(f"  TRAINING PLAN")
    print(f"{'=' * 60}")
    print(f"  Epochs:             {ft_config.num_epochs}")
    print(f"  Steps per epoch:    {steps_per_epoch:,}")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Micro batch size:   {ft_config.micro_batch_size}")
    print(f"  Gradient accum:     {ft_config.gradient_accumulation_steps}")
    print(f"  Effective batch:    {ft_config.micro_batch_size * ft_config.gradient_accumulation_steps}")
    print(f"  Learning rate:      {ft_config.learning_rate} -> {ft_config.min_lr}")
    print(f"  Warmup steps:       {ft_config.warmup_steps}")
    print(f"  Eval every:         {ft_config.eval_interval} steps")
    print(f"  Checkpoint every:   {ft_config.checkpoint_interval} steps")
    print(f"  Sample every:       {ft_config.sample_interval} steps")
    print(f"{'=' * 60}\n")

    # ---- Training loop ----
    model.train()
    global_step = start_step
    train_start_time = time.time()

    for epoch in range(start_epoch, ft_config.num_epochs):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch + 1}/{ft_config.num_epochs}")
        print(f"{'='*60}")

        # Reset for gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # === FORWARD PASS ===
            # torch.amp.autocast automatically casts operations to float16
            # where safe, keeping critical ops in float32
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                logits, loss = model(x, y)
                # Scale loss by accumulation steps so the effective gradient
                # is the mean over the full effective batch
                loss = loss / ft_config.gradient_accumulation_steps

            # Track loss (before scaling for logging purposes)
            accum_loss += loss.item()
            accum_count += 1

            # === BACKWARD PASS ===
            # GradScaler scales the loss up, computes gradients, then
            # the optimizer step scales them back down
            scaler.scale(loss).backward()

            # === OPTIMIZER STEP (every gradient_accumulation_steps) ===
            if accum_count % ft_config.gradient_accumulation_steps == 0:
                # Gradient clipping prevents exploding gradients
                # Unscale first so we clip in float32 space
                if ft_config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), ft_config.grad_clip
                    )
                else:
                    grad_norm = 0.0

                # Actual optimizer step (updates weights)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # === LEARNING RATE SCHEDULE ===
                # Cosine decay with warmup: lr ramps up linearly during
                # warmup, then smoothly decreases following a cosine curve
                lr = get_lr(
                    global_step, ft_config.warmup_steps, total_steps,
                    ft_config.learning_rate, ft_config.min_lr
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # === LOGGING ===
                if global_step % ft_config.log_interval == 0 and global_step > 0:
                    elapsed = time.time() - train_start_time
                    tokens_per_step = (
                        ft_config.micro_batch_size
                        * ft_config.gradient_accumulation_steps
                        * model_config.block_size
                    )
                    logger.log(
                        step=global_step, loss=accum_loss, lr=lr,
                        tokens_seen=global_step * tokens_per_step,
                        total_steps=total_steps,
                        grad_norm=grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        extra={
                            "epoch": epoch + 1,
                            "tokens_per_step": tokens_per_step,
                        },
                    )

                # === EVALUATION ===
                if global_step % ft_config.eval_interval == 0 and global_step > 0:
                    val_loss = evaluate(model, val_loader, device, dtype, use_amp)
                    val_ppl = math.exp(min(val_loss, 20))

                    logger.log_eval(
                        global_step, val_loss, val_ppl,
                        best_val_loss=best_val_loss,
                        tokens_seen=global_step * tokens_per_step,
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, None, scaler, global_step,
                            val_loss, model_config,
                            os.path.join(ft_config.checkpoint_dir, f"step_{global_step}.pt"),
                            is_best=True,
                        )

                # === GENERATE SAMPLE CONVERSATIONS ===
                if global_step % ft_config.sample_interval == 0 and global_step > 0:
                    print(f"\n  --- Sample conversations (step {global_step}) ---")
                    samples = generate_chat_sample(model, tokenizer, device)
                    for prompt, response in samples:
                        print(f"  User: {prompt}")
                        print(f"  GPT-2: {response[:200]}")
                        print()
                        # Log the sample
                        logger.log_sample(global_step, prompt, response)

                # === PERIODIC CHECKPOINT ===
                if global_step % ft_config.checkpoint_interval == 0 and global_step > 0:
                    save_checkpoint(
                        model, optimizer, None, scaler, global_step,
                        best_val_loss, model_config,
                        os.path.join(ft_config.checkpoint_dir, f"step_{global_step}.pt"),
                    )

                # Reset accumulator
                accum_loss = 0.0
                global_step += 1

    # ---- Final save ----
    save_checkpoint(
        model, optimizer, None, scaler, global_step, best_val_loss,
        model_config,
        os.path.join(ft_config.checkpoint_dir, "final.pt"),
        is_best=True,
    )

    # ---- Generate final samples ----
    print(f"\n{'=' * 60}")
    print("  FINAL CONVERSATION SAMPLES")
    print(f"{'=' * 60}")
    samples = generate_chat_sample(model, tokenizer, device)
    for prompt, response in samples:
        print(f"\n  User: {prompt}")
        print(f"  GPT-2: {response[:300]}")
    logger.log_sample(global_step, "FINAL", "\n\n".join(
        f"User: {p}\nGPT-2: {r}" for p, r in samples
    ))

    # ---- Training summary ----
    elapsed = time.time() - train_start_time
    tokens_per_step = (
        ft_config.micro_batch_size
        * ft_config.gradient_accumulation_steps
        * model_config.block_size
    )

    # Generate training_summary.md for documentation
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.save_summary(
        model_config, ft_config,
        total_steps=global_step,
        tokens_seen=global_step * tokens_per_step,
        best_val_loss=best_val_loss,
        device_name=device_name,
    )

    print(f"\n{'=' * 60}")
    print(f"  FINE-TUNING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total steps:    {global_step:,}")
    print(f"  Total time:     {format_time(elapsed)}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best perplexity: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"  Checkpoints:    {ft_config.checkpoint_dir}/")
    print(f"  Best model:     {ft_config.checkpoint_dir}/best.pt")
    print(f"\n  Documentation:")
    print(f"    {ft_config.checkpoint_dir}/training_summary.md")
    print(f"    {ft_config.checkpoint_dir}/training_log.csv")
    print(f"    {ft_config.checkpoint_dir}/training_log.jsonl")
    print(f"    {ft_config.checkpoint_dir}/eval_log.csv")
    print(f"    {ft_config.checkpoint_dir}/samples.txt")
    print(f"\n  To chat with your model:")
    print(f"    python chat.py --checkpoint {ft_config.checkpoint_dir}/best.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
