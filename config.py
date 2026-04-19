"""
config.py - All hyperparameters for GPT-2 Small (124M) training from scratch.

This is the single source of truth for model architecture and training settings.
GPT-2 Small: 12 layers, 12 heads, 768 embedding dim, 50257 vocab (BPE).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """GPT-2 Small architecture configuration."""
    n_layer: int = 12               # Number of transformer blocks
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Embedding / hidden dimension
    block_size: int = 1024          # Maximum sequence length
    vocab_size: int = 50257         # GPT-2 BPE vocabulary size
    dropout: float = 0.0            # Dropout rate (0 for pre-training)
    bias: bool = False              # Use bias in Linear/LayerNorm (False = slightly better)


@dataclass
class PretrainConfig:
    """Pre-training hyperparameters for OpenWebText."""
    # Dataset
    dataset_name: str = "Skylion007/openwebtext"
    val_split_ratio: float = 0.0005

    # Optimization
    max_steps: int = 28_000              # ~23hr on T4 (buffer for eval/checkpoint overhead)
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Batching — optimized for T4 16GB VRAM
    # micro_batch=10 uses ~12GB (77% VRAM), verified as largest safe fit
    # grad_accum=5 gives effective batch of 50 sequences (51K tokens/step)
    micro_batch_size: int = 10
    gradient_accumulation_steps: int = 5

    # Logging and Checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50
    checkpoint_interval: int = 1_000
    sample_interval: int = 500

    # Paths
    checkpoint_dir: str = "checkpoints/pretrain"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * 1024


@dataclass
class FinetuneConfig:
    """
    Fine-tuning hyperparameters for conversational training.

    Optimized for 24 hours on T4 with ~72K training examples:
      - OASST1 (all tree paths, English): ~20K conversations
      - Alpaca-cleaned: ~52K instruction-response pairs
      - Total: ~72K examples

    With micro_batch=8, grad_accum=2, effective_batch=16:
      - ~4,500 steps per epoch
      - 8 epochs = ~36,000 total steps
      - ~24 hours on T4 at ~1,700 tokens/sec

    Key differences from pre-training:
      - 30x smaller learning rate (2e-5 vs 6e-4) to preserve pre-trained knowledge
      - Less weight decay (0.01 vs 0.1) to allow fine-grained learning
      - Dropout enabled (0.1) to prevent overfitting on the smaller dataset
      - Multiple epochs (vs single-pass in pre-training)
    """
    # Optimization
    num_epochs: int = 6                   # Multiple passes through the data
    learning_rate: float = 2e-5           # 30x smaller than pre-training!
    min_lr: float = 2e-6                  # Cosine decay floor
    warmup_steps: int = 200               # Gentle warmup to avoid destabilizing
    weight_decay: float = 0.01            # Light regularization
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    dropout: float = 0.1                  # Regularization (pre-training used 0.0)

    # Batching — optimized for T4 16GB
    micro_batch_size: int = 10            # Increased to maximize GPU utilization
    gradient_accumulation_steps: int = 2  # Effective batch = 20 sequences

    # Logging and Checkpointing
    log_interval: int = 10                # Print every 10 steps
    eval_interval: int = 500              # Evaluate every 500 steps
    eval_steps: int = 50                  # Batches per evaluation
    checkpoint_interval: int = 1_000      # Save checkpoint every 1K steps
    sample_interval: int = 500            # Generate test conversations every 500 steps

    # Paths
    pretrained_checkpoint: str = "checkpoints/pretrain/best.pt"
    checkpoint_dir: str = "checkpoints/finetune"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps
