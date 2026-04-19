"""
model.py - GPT-2 Small (124M) implemented from scratch in PyTorch.

Architecture:
  - Token + Positional Embeddings
  - 12x Transformer Blocks (Pre-LayerNorm):
      - Multi-Head Causal Self-Attention (12 heads, 768 dim)
      - Feed-Forward Network (GELU, 4x expansion)
  - Final LayerNorm + Linear Head (weight-tied with token embeddings)

This follows the GPT-2 paper (Radford et al., 2019) with Pre-LN ordering
(more stable training) instead of Post-LN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    
    Each head independently computes scaled dot-product attention,
    then outputs are concatenated and projected. A causal mask ensures
    each position can only attend to previous positions (autoregressive).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined Q, K, V projection for efficiency (one matmul instead of three)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask: lower triangular matrix
        # Registered as buffer (not a parameter, but moves with .to(device))
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        head_dim = C // self.n_head

        # Project to Q, K, V and split
        qkv = self.c_attn(x)                           # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)        # each (B, T, C)

        # Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Use PyTorch's optimized fused attention kernel
        # This auto-selects the best implementation for the GPU:
        #   - Flash Attention on Ampere+ (compute >= 8.0)
        #   - Memory-efficient attention on older GPUs like T4 (compute 7.5)
        # Much faster than manual attention: avoids materializing the full T×T matrix
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,     # We use is_causal=True instead (more efficient)
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,     # Automatically applies causal mask
        )

        # Concatenate heads: (B, nh, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    
    Expands to 4x the embedding dimension, applies GELU, then projects back.
    This is where most of the model's "knowledge" is stored.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)       # (B, T, 4*C)
        x = self.gelu(x)       # GELU activation
        x = self.c_proj(x)     # (B, T, C)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-LayerNorm ordering.
    
    Pre-LN (used here):  x = x + Attn(LN(x));  x = x + MLP(LN(x))
    Post-LN (original):  x = LN(x + Attn(x));  x = LN(x + MLP(x))
    
    Pre-LN is more stable during training (gradients flow better through residuals).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections around attention and MLP
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    Full GPT-2 Language Model.
    
    Token embeddings + positional embeddings -> N transformer blocks -> 
    LayerNorm -> Linear head (logits over vocabulary).
    
    The LM head shares weights with the token embedding (weight tying),
    which saves parameters and improves performance.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: maps token IDs to vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings: learned position encoding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout on embeddings
            drop = nn.Dropout(config.dropout),
            # Stack of transformer blocks
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            # Final layer norm
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language model head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        # This means the same matrix is used to go from token->hidden and hidden->token
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled initialization for residual projections (GPT-2 convention)
        # Scale by 1/sqrt(2*n_layer) to prevent residual stream from growing
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameter count
        n_params = self.get_num_params()
        print(f"GPT-2 model initialized: {n_params/1e6:.1f}M parameters")

    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count parameters. By default excludes position embeddings
        (since they dont contribute to transformer computation per se).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            idx: Token indices, shape (B, T)
            targets: Target token indices for loss computation, shape (B, T).
                     If None, only compute logits for the last position (inference).
        
        Returns:
            logits: Vocabulary logits
            loss: Cross-entropy loss (if targets provided)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Embeddings
        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd) - broadcast over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # ignore padding tokens marked as -1
            )
        else:
            # Inference: only compute logits for the last position (efficient)
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            idx: Starting token indices, shape (B, T)
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more focused, >1 = more random)
            top_k: If set, only sample from the top k most likely tokens
            top_p: If set, nucleus sampling - sample from smallest set with cumulative prob >= top_p
        
        Returns:
            idx: Extended sequence with generated tokens, shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop sequence to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass (inference mode - only last position)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so the first token above threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embeddings for fine-tuning with additional special tokens.
        Preserves existing weights and initializes new ones randomly.
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return

        # Create new embedding
        old_wte = self.transformer.wte
        self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd)
        self.transformer.wte.weight.data[:old_vocab_size] = old_wte.weight.data
        # Initialize new tokens
        nn.init.normal_(self.transformer.wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)

        # Create new LM head
        old_lm = self.lm_head
        self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False)
        self.lm_head.weight.data[:old_vocab_size] = old_lm.weight.data

        # Re-tie weights
        self.transformer.wte.weight = self.lm_head.weight

        # Update config
        self.config.vocab_size = new_vocab_size
        print(f"Resized embeddings: {old_vocab_size} -> {new_vocab_size}")
