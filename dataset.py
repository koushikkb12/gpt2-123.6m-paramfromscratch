"""
dataset.py - Data loading for GPT-2 pre-training and conversational fine-tuning.

Pre-training:
  - Streams OpenWebText from HuggingFace (no full download needed)
  - Tokenizes with tiktoken (GPT-2 BPE)
  - Packs sequences to block_size (1024) with no padding

Fine-tuning:
  - Loads OpenAssistant/oasst1 conversations
  - Formats as multi-turn chat with special tokens
  - Creates loss mask (only train on assistant responses)
"""

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import tiktoken
from datasets import load_dataset
from typing import List, Dict
import numpy as np


def get_tokenizer():
    """Get GPT-2 BPE tokenizer."""
    return tiktoken.get_encoding("gpt2")


# ============================================================
# PRE-TRAINING DATASET
# ============================================================

class PretrainDataset(IterableDataset):
    """
    Streaming dataset for pre-training on OpenWebText.
    
    Streams documents, tokenizes them, and packs tokens into
    fixed-length sequences of block_size. No padding waste.
    Documents are separated by an end-of-text token.
    """

    def __init__(
        self,
        dataset_name: str = "Skylion007/openwebtext",
        block_size: int = 1024,
        split: str = "train",
        max_tokens: int = None,
    ):
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.split = split
        self.max_tokens = max_tokens
        self.tokenizer = get_tokenizer()
        self.eot_token = self.tokenizer.eot_token  # End of text token (50256)

    def __iter__(self):
        """
        Yield (input, target) pairs of shape (block_size,).
        Target is input shifted by 1 position (next-token prediction).
        """
        # Load dataset in streaming mode
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True, trust_remote_code=True)

        buffer = []
        tokens_yielded = 0

        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue

            # Tokenize and append end-of-text token
            tokens = self.tokenizer.encode_ordinary(text)
            tokens.append(self.eot_token)
            buffer.extend(tokens)

            # Yield complete blocks
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)   # input
                y = torch.tensor(chunk[1:], dtype=torch.long)     # target (shifted by 1)
                yield x, y

                tokens_yielded += self.block_size
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValidationDataset(Dataset):
    """
    Small validation dataset for periodic evaluation during training.
    Pre-tokenizes a subset of data into memory.
    """

    def __init__(
        self,
        dataset_name: str = "Skylion007/openwebtext",
        block_size: int = 1024,
        num_tokens: int = 500_000,
    ):
        self.block_size = block_size
        tokenizer = get_tokenizer()

        # Load a small validation split
        print(f"Loading validation data ({num_tokens/1e3:.0f}K tokens)...")
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

        # Collect tokens
        all_tokens = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            tokens = tokenizer.encode_ordinary(text)
            tokens.append(tokenizer.eot_token)
            all_tokens.extend(tokens)
            if len(all_tokens) >= num_tokens:
                break

        all_tokens = all_tokens[:num_tokens]
        self.tokens = np.array(all_tokens, dtype=np.int64)
        self.n_chunks = (len(self.tokens) - 1) // self.block_size
        print(f"  Validation: {self.n_chunks} chunks, {len(self.tokens)/1e3:.0f}K tokens")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


# ============================================================
# CHAT FINE-TUNING DATASET
# ============================================================

class ChatDataset(Dataset):
    """
    Dataset for conversational fine-tuning using OpenAssistant/oasst1.
    
    Formats conversations with special tokens and creates
    loss masks so we only train on assistant responses.
    """

    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|endofturn|>"

    def __init__(
        self,
        dataset_name: str = "OpenAssistant/oasst1",
        block_size: int = 1024,
        split: str = "train",
    ):
        self.block_size = block_size
        self.tokenizer = get_tokenizer()
        
        # We encode special tokens as regular text
        # The model learns these patterns during fine-tuning
        self.user_ids = self.tokenizer.encode_ordinary(self.USER_TOKEN)
        self.assistant_ids = self.tokenizer.encode_ordinary(self.ASSISTANT_TOKEN)
        self.end_ids = self.tokenizer.encode_ordinary(self.END_TOKEN)

        # Load and process conversations
        print(f"Loading {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Build conversation trees
        self.conversations = self._build_conversations(ds)
        print(f"  Built {len(self.conversations)} conversations")

    def _build_conversations(self, ds) -> list:
        """
        Build linear conversations from the tree structure in oasst1.
        Each conversation = list of (role, text) tuples.
        """
        # Group messages by conversation
        from collections import defaultdict
        msg_by_id = {}
        children = defaultdict(list)
        roots = []
        
        for row in ds:
            msg_id = row["message_id"]
            parent_id = row["parent_id"]
            msg_by_id[msg_id] = row
            if parent_id is None:
                roots.append(msg_id)
            else:
                children[parent_id].append(msg_id)
        
        # Extract linear conversation paths (follow first child)
        conversations = []
        for root_id in roots:
            conv = []
            current_id = root_id
            while current_id is not None:
                msg = msg_by_id[current_id]
                role = msg["role"]  # "prompter" or "assistant"
                text = msg["text"]
                conv.append((role, text))
                # Follow first child
                kids = children.get(current_id, [])
                current_id = kids[0] if kids else None
            
            if len(conv) >= 2:  # Need at least user + assistant
                conversations.append(conv)
        
        return conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: token ids, shape (block_size,)
            labels: target ids with -1 for non-assistant tokens, shape (block_size,)
        """
        conv = self.conversations[idx]
        
        # Build token sequence with loss mask
        all_tokens = []
        all_mask = []  # 1 = compute loss, 0 = ignore
        
        for role, text in conv:
            if role == "prompter":
                # User turn: add user token + text (no loss)
                turn_tokens = self.user_ids + self.tokenizer.encode_ordinary("\n" + text + "\n")
                all_tokens.extend(turn_tokens)
                all_mask.extend([0] * len(turn_tokens))
            else:
                # Assistant turn: add assistant token + text + end (compute loss)
                prefix = self.assistant_ids + self.tokenizer.encode_ordinary("\n")
                response = self.tokenizer.encode_ordinary(text)
                suffix = self.tokenizer.encode_ordinary("\n") + self.end_ids
                
                # No loss on the prefix tokens
                all_tokens.extend(prefix)
                all_mask.extend([0] * len(prefix))
                # Loss on response + end token
                all_tokens.extend(response + suffix)
                all_mask.extend([1] * len(response + suffix))
        
        # Truncate or pad to block_size + 1 (need one extra for target shift)
        total_len = self.block_size + 1
        if len(all_tokens) > total_len:
            all_tokens = all_tokens[:total_len]
            all_mask = all_mask[:total_len]
        else:
            # Pad with EOT tokens (masked)
            pad_len = total_len - len(all_tokens)
            all_tokens.extend([self.tokenizer.eot_token] * pad_len)
            all_mask.extend([0] * pad_len)
        
        # Create input/target pairs
        x = torch.tensor(all_tokens[:-1], dtype=torch.long)
        y = torch.tensor(all_tokens[1:], dtype=torch.long)
        mask = torch.tensor(all_mask[1:], dtype=torch.long)
        
        # Set non-assistant targets to -1 (ignored by cross_entropy)
        y[mask == 0] = -1
        
        return x, y


def create_pretrain_dataloader(config, split="train", max_tokens=None):
    """Create a DataLoader for pre-training."""
    dataset = PretrainDataset(
        dataset_name=config.dataset_name,
        block_size=1024,
        split=split,
        max_tokens=max_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
    )


def create_chat_dataloader(config, split="train"):
    """Create a DataLoader for chat fine-tuning."""
    dataset = ChatDataset(
        dataset_name=config.dataset_name,
        block_size=1024,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
    )
