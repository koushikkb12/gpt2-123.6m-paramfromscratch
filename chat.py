"""
chat.py - Interactive chat interface for your fine-tuned GPT-2.

Usage:
    python chat.py                                          # Use default checkpoint
    python chat.py --checkpoint checkpoints/finetune/best.pt  # Specify checkpoint
    python chat.py --temperature 0.9 --top_k 40             # Adjust sampling

Commands during chat:
    /quit or /exit  - Exit chat
    /reset          - Clear conversation history
    /temp <float>   - Change temperature
    /topk <int>     - Change top-k
"""

import sys
import argparse
import torch
import tiktoken
import time

from config import ModelConfig, FinetuneConfig
from model import GPT2


USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endofturn|>"


def load_model(checkpoint_path: str, device: str):
    """Load a fine-tuned model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Reconstruct config
    config = ModelConfig()
    saved_config = checkpoint.get("config", {})
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    model = GPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.get_num_params()/1e6:.1f}M parameters")
    return model, config


def generate_response(
    model, tokenizer, conversation_tokens: list,
    max_tokens: int = 300, temperature: float = 0.8,
    top_k: int = 50, top_p: float = 0.9, device: str = "cpu"
) -> str:
    """Generate an assistant response given conversation history."""
    
    # Truncate to fit block_size
    max_context = model.config.block_size - max_tokens
    if len(conversation_tokens) > max_context:
        conversation_tokens = conversation_tokens[-max_context:]
    
    idx = torch.tensor([conversation_tokens], dtype=torch.long, device=device)
    
    # Encode end token to detect when to stop
    end_ids = tokenizer.encode_ordinary(END_TOKEN)
    
    start_time = time.time()
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Check for end token
            if len(generated_tokens) >= len(end_ids):
                if generated_tokens[-len(end_ids):] == end_ids:
                    generated_tokens = generated_tokens[:-len(end_ids)]
                    break
            
            # Also stop at EOT
            if token_id == tokenizer.eot_token:
                generated_tokens = generated_tokens[:-1]
                break
    
    elapsed = time.time() - start_time
    tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
    
    response = tokenizer.decode(generated_tokens).strip()
    return response, tokens_per_sec, len(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="Chat with your GPT-2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/finetune/best.pt",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=300)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print()
    print("=" * 50)
    print("  GPT-2 Chat - Trained From Scratch!")
    print("  Type /quit to exit, /reset to clear history")
    print("=" * 50)
    print()
    
    # Conversation state
    conversation_tokens = []
    temperature = args.temperature
    top_k = args.top_k
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit"):
                print("Goodbye!")
                break
            elif cmd[0] == "/reset":
                conversation_tokens = []
                print("Conversation cleared.\n")
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                temperature = float(cmd[1])
                print(f"Temperature set to {temperature}\n")
                continue
            elif cmd[0] == "/topk" and len(cmd) > 1:
                top_k = int(cmd[1])
                print(f"Top-k set to {top_k}\n")
                continue
            else:
                print("Unknown command. Available: /quit, /reset, /temp, /topk\n")
                continue
        
        # Format user message
        user_formatted = USER_TOKEN + "\n" + user_input + "\n" + ASSISTANT_TOKEN + "\n"
        user_tokens = tokenizer.encode_ordinary(user_formatted)
        conversation_tokens.extend(user_tokens)
        
        # Generate response
        response, tok_per_sec, n_tokens = generate_response(
            model, tokenizer, conversation_tokens,
            max_tokens=args.max_tokens, temperature=temperature,
            top_k=top_k, top_p=args.top_p, device=device
        )
        
        # Add response to history
        response_tokens = tokenizer.encode_ordinary(response + "\n" + END_TOKEN)
        conversation_tokens.extend(response_tokens)
        
        print(f"\nGPT-2: {response}")
        print(f"  [{n_tokens} tokens, {tok_per_sec:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
