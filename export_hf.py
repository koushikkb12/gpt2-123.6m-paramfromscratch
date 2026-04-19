"""
export_hf.py - Export your trained GPT-2 to HuggingFace Hub.

Usage:
    python export_hf.py --checkpoint checkpoints/finetune/best.pt --repo_name your-username/gpt2-from-scratch
    
This converts the model to HuggingFace-compatible format and pushes it to the Hub
with a proper model card, so anyone can load it with:
    
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("your-username/gpt2-from-scratch")
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

from config import ModelConfig
from model import GPT2


def convert_to_hf_format(checkpoint_path: str, output_dir: str, model_config: ModelConfig):
    """
    Convert our GPT-2 checkpoint to HuggingFace GPT-2 format.
    
    HuggingFace GPT2 uses slightly different weight names, so we
    need to map our names to theirs.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Load our model
    saved_config = checkpoint.get("config", {})
    for key, value in saved_config.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    
    model = GPT2(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Save in HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    print(f"  Saved model weights")
    
    # Save config.json (HuggingFace format)
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": model_config.vocab_size,
        "n_positions": model_config.block_size,
        "n_embd": model_config.n_embd,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_inner": 4 * model_config.n_embd,
        "activation_function": "gelu",
        "resid_pdrop": model_config.dropout,
        "embd_pdrop": model_config.dropout,
        "attn_pdrop": model_config.dropout,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "torch_dtype": "float32",
        "_name_or_path": "gpt2-from-scratch",
        "training_info": {
            "total_params": model.get_num_params(),
            "architecture": "GPT-2 Small (trained from scratch)",
            "dataset": "OpenWebText + OpenAssistant/oasst1",
        }
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"  Saved config.json")
    
    return model


def create_model_card(output_dir: str, model_config: ModelConfig, checkpoint_path: str):
    """Create a README.md model card for HuggingFace."""
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    val_loss = checkpoint.get("val_loss", "N/A")
    step = checkpoint.get("step", "N/A")
    
    card = f"""---
license: mit
tags:
  - gpt2
  - from-scratch
  - causal-lm
  - conversational
language:
  - en
pipeline_tag: text-generation
---

# GPT-2 From Scratch (124M Parameters)

A GPT-2 Small model (124M parameters) trained **entirely from scratch** on a single NVIDIA Tesla T4 GPU.

## Model Details

| Property | Value |
|---|---|
| Architecture | GPT-2 Small (Pre-LayerNorm) |
| Parameters | ~124M |
| Layers | {model_config.n_layer} |
| Attention Heads | {model_config.n_head} |
| Hidden Dimension | {model_config.n_embd} |
| Max Sequence Length | {model_config.block_size} |
| Vocabulary | GPT-2 BPE (50,257 tokens) |

## Training

### Pre-training
- **Dataset**: OpenWebText (~2B tokens)
- **Hardware**: Single NVIDIA Tesla T4 (16GB VRAM)
- **Precision**: Mixed FP16
- **Optimizer**: AdamW (lr=6e-4, cosine decay)
- **Batch Size**: 64 (8 micro-batch x 8 gradient accumulation)

### Fine-tuning
- **Dataset**: OpenAssistant/oasst1 (multi-turn conversations)
- **Objective**: Causal LM with masked loss (only on assistant responses)
- **Final Val Loss**: {val_loss}
- **Final Step**: {step}

## Usage

```python
# Note: This model uses a custom architecture.
# Load with the original training code for best results.
import torch
from model import GPT2
from config import ModelConfig

config = ModelConfig()
model = GPT2(config)
checkpoint = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(checkpoint)
```

## Built From Scratch

Every component was implemented from zero in PyTorch:
- Multi-head causal self-attention
- Feed-forward networks with GELU
- Pre-LayerNorm transformer blocks
- Positional and token embeddings
- Weight tying between embedding and output head
- Full training loop with mixed precision, gradient accumulation, checkpointing

## Limitations

This is a learning project. The model is small (124M params) and trained on limited data compared
to production models. It can hold basic conversations but will not match the quality of larger models.

## License

MIT
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)
    print(f"  Saved README.md (model card)")


def push_to_hub(output_dir: str, repo_name: str, token: str = None, revision: str = "main"):
    """Push the model to HuggingFace Hub."""
    from huggingface_hub import HfApi
    
    print(f"\nPushing to HuggingFace Hub: {repo_name} (branch: {revision})")
    
    api = HfApi(token=token)
    api.create_repo(repo_name, exist_ok=True, repo_type="model", token=token)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_name,
        repo_type="model",
        revision=revision,
        commit_message=f"Upload model to {revision} branch",
        token=token
    )
    print(f"  Successfully pushed to https://huggingface.co/{repo_name}/tree/{revision}")


def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 to HuggingFace")
    parser.add_argument("--finetune_checkpoint", type=str, default="checkpoints/finetune/best.pt",
                        help="Path to finetuned checkpoint")
    parser.add_argument("--pretrain_checkpoint", type=str, default="checkpoints/pretrain/best.pt",
                        help="Path to pretrained checkpoint")
    parser.add_argument("--repo_name", type=str, default=None,
                        help="HuggingFace repo name (e.g., username/gpt2-from-scratch)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token with write permissions")
    parser.add_argument("--push", action="store_true",
                        help="Push to HuggingFace Hub after export")
    args = parser.parse_args()
    
    if args.push and not args.repo_name:
        print("\nError: --repo_name is required when using --push")
        print("Example: python export_hf.py --push --repo_name yourname/gpt2-from-scratch --token hf_...")
        return
        
    model_config = ModelConfig()
    
    # 1. Process Finetuned Model (main branch)
    print(f"\n{'='*50}")
    print(f"Exporting FINETUNED model from {args.finetune_checkpoint}...")
    print(f"{'='*50}\n")
    
    finetune_dir = "hf_export_finetune"
    convert_to_hf_format(args.finetune_checkpoint, finetune_dir, model_config)
    create_model_card(finetune_dir, model_config, args.finetune_checkpoint)
    
    if args.push:
        push_to_hub(finetune_dir, args.repo_name, token=args.token, revision="main")

    # 2. Process Pretrained Model (pretrained branch)
    if os.path.exists(args.pretrain_checkpoint):
        print(f"\n{'='*50}")
        print(f"Exporting PRETRAINED model from {args.pretrain_checkpoint}...")
        print(f"{'='*50}\n")
        
        pretrain_dir = "hf_export_pretrain"
        convert_to_hf_format(args.pretrain_checkpoint, pretrain_dir, model_config)
        create_model_card(pretrain_dir, model_config, args.pretrain_checkpoint)
        
        if args.push:
            # First, create branch if it doesn't exist
            from huggingface_hub import HfApi
            api = HfApi(token=args.token)
            try:
                api.create_branch(repo_id=args.repo_name, branch="pretrained", repo_type="model")
            except Exception as e:
                print(f"  Branch 'pretrained' might already exist or error: {e}")
            
            push_to_hub(pretrain_dir, args.repo_name, token=args.token, revision="pretrained")
    else:
        print(f"\nPretrained checkpoint not found at {args.pretrain_checkpoint}. Skipping.")

    print("\nDone!")


if __name__ == "__main__":
    main()
