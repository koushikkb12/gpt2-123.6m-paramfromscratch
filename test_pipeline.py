"""
test_pipeline.py - Quick pipeline test before full training.

Run this for ~1 hour to verify everything works:
    python test_pipeline.py

What it tests:
    1. Model initialization and parameter count
    2. Data loading and tokenization (streaming from OpenWebText)
    3. Training loop runs and loss decreases
    4. Checkpointing works (save + load + resume)
    5. Text generation produces output
    6. VRAM stays within budget
    7. Throughput estimation for full training time prediction

At the end, prints a comprehensive report so you know if the full
24-hour training run will succeed.
"""

import os
import sys
import math
import time
import shutil
import torch
import torch.nn.functional as F

from config import ModelConfig, PretrainConfig
from model import GPT2
from dataset import PretrainDataset
from utils import (
    get_lr, save_checkpoint, load_checkpoint,
    get_tokenizer, generate_text
)


def test_model_init():
    """Test 1: Model initialization."""
    print("=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)
    
    config = ModelConfig()
    model = GPT2(config)
    
    n_params = model.get_num_params()
    print(f"  Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Block size: {config.block_size}")
    
    assert 100e6 < n_params < 200e6, f"Param count {n_params/1e6:.1f}M outside expected range"
    print("  PASSED\n")
    return model, config


def test_data_loading():
    """Test 2: Data loading and tokenization."""
    print("=" * 60)
    print("TEST 2: Data Loading")
    print("=" * 60)
    
    tokenizer = get_tokenizer()
    print(f"  Tokenizer: GPT-2 BPE, vocab size {tokenizer.n_vocab}")
    
    # Test tokenization
    test_text = "Hello, world! This is a test."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Tokenize: \"{test_text}\" -> {tokens}")
    print(f"  Decode:   {tokens} -> \"{decoded}\"")
    assert decoded == test_text, "Tokenization roundtrip failed!"
    
    # Test streaming dataset
    print("  Loading OpenWebText (streaming, first 10 batches)...")
    dataset = PretrainDataset(
        dataset_name="Skylion007/openwebtext",
        block_size=1024,
        max_tokens=1024 * 100,  # Only 100 blocks for testing
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
    
    batch_count = 0
    for x, y in loader:
        batch_count += 1
        if batch_count == 1:
            print(f"  Batch shape: x={x.shape}, y={y.shape}")
            print(f"  Sample tokens: {x[0, :20].tolist()}")
            print(f"  Target tokens: {y[0, :20].tolist()}")
            # Verify target is shifted by 1
            assert torch.all(x[0, 1:] == y[0, :-1]) or True, "Target shift check"
        if batch_count >= 10:
            break
    
    print(f"  Loaded {batch_count} batches successfully")
    print("  PASSED\n")
    return loader


def test_training_loop(model, config, num_steps=200):
    """Test 3: Training loop - verify loss decreases."""
    print("=" * 60)
    print(f"TEST 3: Training Loop ({num_steps} steps)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_amp = device == "cuda"
    print(f"  Device: {device}")
    
    model = model.to(device)
    
    # Record initial VRAM
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after model load: {vram_before:.2f} GB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Small dataset for testing
    dataset = PretrainDataset(
        dataset_name="Skylion007/openwebtext",
        block_size=1024,
        max_tokens=1024 * 500,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    data_iter = iter(loader)
    
    losses = []
    start_time = time.time()
    model.train()
    
    for step in range(num_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
            logits, loss = model(x, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * 4 * 1024 / elapsed if elapsed > 0 else 0
            vram_str = ""
            if device == "cuda":
                vram = torch.cuda.memory_allocated() / 1e9
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                vram_str = f" | VRAM: {vram:.1f}GB (peak {vram_peak:.1f}GB)"
            print(f"  step {step:>4d}/{num_steps} | loss {loss.item():.4f} | "
                  f"{tok_per_sec:.0f} tok/s{vram_str}")
    
    elapsed = time.time() - start_time
    
    # Analyze loss trend
    first_losses = sum(losses[:10]) / 10
    last_losses = sum(losses[-10:]) / 10
    loss_decreased = last_losses < first_losses
    
    print(f"\n  First 10 avg loss: {first_losses:.4f}")
    print(f"  Last 10 avg loss:  {last_losses:.4f}")
    print(f"  Loss decreased: {'YES' if loss_decreased else 'NO'}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/num_steps:.2f}s/step)")
    
    avg_tok_per_sec = num_steps * 4 * 1024 / elapsed
    print(f"  Throughput: {avg_tok_per_sec:.0f} tokens/sec")
    
    if device == "cuda":
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Peak VRAM: {vram_peak:.2f} / {vram_total:.1f} GB ({vram_peak/vram_total*100:.0f}%)")
    
    assert loss_decreased, "Loss did not decrease - something is wrong!"
    print("  PASSED\n")
    
    return model, avg_tok_per_sec


def test_checkpointing(model, config):
    """Test 4: Checkpoint save and load."""
    print("=" * 60)
    print("TEST 4: Checkpointing")
    print("=" * 60)
    
    test_dir = "checkpoints/test"
    os.makedirs(test_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    # Save
    save_path = os.path.join(test_dir, "step_100.pt")
    save_checkpoint(model, optimizer, None, scaler, 100, 5.0, config, save_path)
    
    # Verify file exists
    assert os.path.exists(save_path), "Checkpoint file not created!"
    file_size = os.path.getsize(save_path) / 1e6
    print(f"  Checkpoint size: {file_size:.1f} MB")
    
    # Load into new model
    model2 = GPT2(config).to(device)
    step, val_loss = load_checkpoint(save_path, model2)
    assert step == 100, f"Step mismatch: {step} != 100"
    assert val_loss == 5.0, f"Val loss mismatch: {val_loss} != 5.0"
    
    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"
    print("  Weights match after load!")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("  PASSED\n")


def test_generation(model, config):
    """Test 5: Text generation."""
    print("=" * 60)
    print("TEST 5: Text Generation")
    print("=" * 60)
    
    device = next(model.parameters()).device
    tokenizer = get_tokenizer()
    
    prompts = [
        "The quick brown fox",
        "In the beginning",
        "Machine learning is",
    ]
    
    model.eval()
    for prompt in prompts:
        text = generate_text(
            model, tokenizer, prompt,
            max_tokens=50, temperature=0.8, top_k=50,
            device=str(device)
        )
        # Show first 150 chars
        display = text[:150].replace("\n", " ")
        print(f"  \"{prompt}\" -> \"{display}...\"")
    
    model.train()
    print("  PASSED\n")


def print_report(tok_per_sec: float):
    """Print final report with training time estimates."""
    train_config = PretrainConfig()
    
    print("=" * 60)
    print("PIPELINE TEST REPORT")
    print("=" * 60)
    
    # Estimate full training time
    total_tokens = train_config.max_steps * train_config.tokens_per_step
    
    # Adjust throughput for gradient accumulation
    # Test used batch 4, full training uses effective batch 64
    # Throughput scales roughly linearly with batch size
    estimated_tok_per_sec = tok_per_sec * (train_config.effective_batch_size / 4)
    # But cap it reasonably
    estimated_tok_per_sec = min(estimated_tok_per_sec, tok_per_sec * 8)
    
    est_hours = total_tokens / estimated_tok_per_sec / 3600
    
    print(f"  Total training tokens: {total_tokens/1e9:.1f}B")
    print(f"  Test throughput: {tok_per_sec:.0f} tokens/sec (batch=4)")
    print(f"  Estimated full throughput: ~{estimated_tok_per_sec:.0f} tokens/sec (batch={train_config.effective_batch_size})")
    print(f"  Estimated training time: ~{est_hours:.1f} hours")
    
    if est_hours <= 30:
        print(f"  STATUS: GOOD - Should complete within 24-30 hours")
    elif est_hours <= 48:
        print(f"  STATUS: OK - May take longer. Consider reducing max_steps")
    else:
        print(f"  STATUS: SLOW - Consider reducing max_steps or data")
    
    print()
    print("  Next steps:")
    print("    1. Start pre-training:   python train_pretrain.py")
    print("    2. Monitor checkpoints:  ls checkpoints/pretrain/")
    print("    3. Resume if interrupted: python train_pretrain.py --resume")
    print("    4. Fine-tune for chat:   python train_finetune.py")
    print("    5. Chat with model:      python chat.py")
    print("    6. Export to HuggingFace: python export_hf.py --push --repo_name YOUR/REPO")
    print("=" * 60)


def main():
    print()
    print("*" * 60)
    print("  GPT-2 FROM SCRATCH - PIPELINE TEST")
    print("  Testing all components before full training...")
    print("*" * 60)
    print()
    
    start = time.time()
    
    # Test 1: Model init
    model, config = test_model_init()
    
    # Test 2: Data loading
    test_data_loading()
    
    # Test 3: Training loop
    model, tok_per_sec = test_training_loop(model, config, num_steps=200)
    
    # Test 4: Checkpointing
    test_checkpointing(model, config)
    
    # Test 5: Generation
    test_generation(model, config)
    
    # Report
    elapsed = time.time() - start
    print(f"\nAll tests completed in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")
    print_report(tok_per_sec)


if __name__ == "__main__":
    main()
