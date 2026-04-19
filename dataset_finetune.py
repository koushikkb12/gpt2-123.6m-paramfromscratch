"""
dataset_finetune.py - Conversation datasets for fine-tuning GPT-2 into a chatbot.

=== WHAT THIS FILE DOES ===
Our pre-trained GPT-2 can predict the next word, but it doesn't understand
conversations. This file teaches it the FORMAT of human-AI dialogue by
preparing training data from two sources:

  1. OpenAssistant/oasst1 - Real human-AI conversations (tree structure)
  2. yahma/alpaca-cleaned - 52K instruction-response pairs from Stanford

=== KEY CONCEPTS ===

1. CHAT FORMAT - We wrap every conversation in special tokens so the model
   learns WHEN to speak and WHEN to listen. See USER_TOKEN, ASSISTANT_TOKEN,
   and END_TOKEN below.

2. LOSS MASKING - We only compute loss on the assistant's response tokens.
   The user's message tokens have their labels set to -1 (ignored by
   PyTorch's cross_entropy). This teaches the model to GENERATE responses,
   not to parrot back user messages.

3. TREE EXPLORATION (OASST1) - OASST1 stores conversations as TREES where
   each message can have multiple replies. The original code only followed
   the first child, giving about 3,500 English conversations. By exploring ALL
   paths through the tree, we get about 20,000 conversations - 6x more data!

4. MULTI-SOURCE - We combine OASST1 (multi-turn conversations) with
   Alpaca (single-turn instructions) so the model learns both styles.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import tiktoken
from datasets import load_dataset
from collections import defaultdict
import numpy as np


def get_tokenizer():
    """Get GPT-2 BPE tokenizer (same one used during pre-training)."""
    return tiktoken.get_encoding("gpt2")


# ============================================================
# SPECIAL TOKEN STRINGS
# ============================================================
# We don't add these to the tokenizer's vocabulary. Instead, we encode
# them as regular text. The model learns during fine-tuning that these
# character sequences mark conversation boundaries.
#
# Why not add real special tokens?
#   - Adding tokens changes the embedding matrix size
#   - The pre-trained weights wouldn't have embeddings for new tokens
#   - Encoding as text works well enough for a 124M model
# ============================================================

USER_TOKEN = "<" + "|user|" + ">"
ASSISTANT_TOKEN = "<" + "|assistant|" + ">"
END_TOKEN = "<" + "|endofturn|" + ">"


def _format_conversation(conv, tokenizer, user_ids, assistant_ids, end_ids, block_size):
    """
    Format a conversation into (input_ids, labels) for training.

    This is the core formatting function shared by both OasstDataset and
    AlpacaDataset. It converts a list of (role, text) tuples into:
      - x: input token IDs, shape (block_size,)
      - y: target token IDs with -1 for masked positions, shape (block_size,)

    The key trick: we set labels to -1 for all tokens that are NOT
    part of the assistant's response. PyTorch's cross_entropy loss
    automatically ignores positions where label == -1.

    This means the model only learns to predict the assistant's words,
    not the user's words or the special tokens.

    Input format for each turn:
        User turn:      USER_TOKEN + newline + user message + newline
        Assistant turn: ASSISTANT_TOKEN + newline + response + newline + END_TOKEN

    Loss mask:
        User turns:     all 0s (don't learn to predict user text)
        Assistant text:  all 1s (learn to predict the response)
        Special tokens: all 0s (don't learn to predict format tokens)
    """
    all_tokens = []     # The actual token IDs
    all_mask = []       # 1 = compute loss here, 0 = ignore

    for role, text in conv:
        if role == "prompter":
            # === USER TURN ===
            # Format: USER_TOKEN + newline + user_message + newline
            # Mask: all zeros (we don't train on user text)
            turn_tokens = (
                user_ids
                + tokenizer.encode_ordinary("\n" + text + "\n")
            )
            all_tokens.extend(turn_tokens)
            all_mask.extend([0] * len(turn_tokens))

        else:
            # === ASSISTANT TURN ===
            # Format: ASSISTANT_TOKEN + newline + response + newline + END_TOKEN
            # Mask: 0 for prefix, 1 for response + end token

            # Prefix (no loss) - the model sees these but isn't
            # penalized for not predicting them
            prefix = (
                assistant_ids
                + tokenizer.encode_ordinary("\n")
            )
            all_tokens.extend(prefix)
            all_mask.extend([0] * len(prefix))

            # Response + end token (loss computed here!)
            # This is what the model actually learns to generate
            response = tokenizer.encode_ordinary(text)
            suffix = (
                tokenizer.encode_ordinary("\n")
                + end_ids
            )
            all_tokens.extend(response + suffix)
            all_mask.extend([1] * len(response + suffix))

    # === TRUNCATION / PADDING ===
    # We need block_size + 1 tokens because we shift by 1 to create
    # the (input, target) pair: input = tokens[:-1], target = tokens[1:]
    total_len = block_size + 1

    if len(all_tokens) > total_len:
        # Truncate long conversations (lose the end)
        all_tokens = all_tokens[:total_len]
        all_mask = all_mask[:total_len]
    else:
        # Pad short conversations with EOT tokens (masked out)
        pad_len = total_len - len(all_tokens)
        all_tokens.extend([tokenizer.eot_token] * pad_len)
        all_mask.extend([0] * pad_len)

    # === CREATE INPUT/TARGET PAIRS ===
    # Standard language model setup:
    #   input:  [tok_0, tok_1, ..., tok_N-1]
    #   target: [tok_1, tok_2, ..., tok_N  ]
    # The model predicts the next token at each position
    x = torch.tensor(all_tokens[:-1], dtype=torch.long)
    y = torch.tensor(all_tokens[1:], dtype=torch.long)
    mask = torch.tensor(all_mask[1:], dtype=torch.long)

    # Set ignored positions to -1
    # cross_entropy(ignore_index=-1) will skip these positions
    y[mask == 0] = -1

    return x, y


# ============================================================
# OASST1 DATASET - Multi-turn conversations
# ============================================================

class OasstDataset(Dataset):
    """
    Loads OpenAssistant/oasst1 conversations and formats them for training.

    OASST1 is a tree-structured dataset where:
    - Each conversation starts with a "root" message from a user (prompter)
    - Each message can have multiple reply branches (like Reddit threads)
    - We extract ALL paths from root to leaf, giving us many more
      training conversations than just following the first reply

    Example tree:
        User: "What is Python?"
        +-- Assistant: "Python is a programming language..."
        |   +-- User: "What are its main features?"
        |       +-- Assistant: "Key features include..."   <- Path 1
        |       +-- Assistant: "Python is known for..."    <- Path 2
        +-- Assistant: "Python is a snake species..."      <- Path 3

    Instead of getting 1 conversation, we get 3!
    """

    def __init__(self, block_size=1024, split="train", english_only=True):
        self.block_size = block_size
        self.tokenizer = get_tokenizer()

        # Pre-encode our special token strings
        # These become sequences of regular token IDs
        self.user_ids = self.tokenizer.encode_ordinary(USER_TOKEN)
        self.assistant_ids = self.tokenizer.encode_ordinary(ASSISTANT_TOKEN)
        self.end_ids = self.tokenizer.encode_ordinary(END_TOKEN)

        # Load the raw dataset from HuggingFace
        print(f"Loading OpenAssistant/oasst1 ({split})...")
        ds = load_dataset("OpenAssistant/oasst1", split=split)

        # Build all conversation paths from the tree structure
        self.conversations = self._build_all_paths(ds, english_only)
        print(f"  Built {len(self.conversations)} conversation paths")

    def _build_all_paths(self, ds, english_only):
        """
        Extract ALL linear paths through the OASST1 conversation trees.

        The algorithm:
        1. Index all messages by ID and build parent->children mapping
        2. Find root messages (no parent = start of conversation)
        3. DFS from each root to enumerate every root-to-leaf path
        4. Each path becomes one training conversation

        This is a KEY optimization: the original code followed only the
        first child at each node, producing about 3,500 English conversations.
        By following ALL children, we produce about 20,000!
        """
        # Step 1: Build lookup structures
        msg_by_id = {}                    # message_id -> full message dict
        children = defaultdict(list)      # parent_id -> [child_ids]
        roots = []                        # message_ids with no parent

        for row in ds:
            msg_id = row["message_id"]
            parent_id = row["parent_id"]
            msg_by_id[msg_id] = row

            if parent_id is None:
                roots.append(msg_id)
            else:
                children[parent_id].append(msg_id)

        # Step 2: DFS to extract all root-to-leaf paths
        def get_all_paths(node_id, current_path):
            """
            Recursively explore tree, collecting complete paths.

            current_path: list of (role, text) tuples built so far
            Returns: list of complete paths (each path = list of tuples)
            """
            msg = msg_by_id[node_id]
            # Extend the current path with this message
            new_path = current_path + [(msg["role"], msg["text"])]

            kids = children.get(node_id, [])
            if not kids:
                # Leaf node - this path is complete
                return [new_path]

            # Internal node - explore each child branch
            all_paths = []
            for child_id in kids:
                all_paths.extend(get_all_paths(child_id, new_path))
            return all_paths

        # Step 3: Collect paths from all root messages
        conversations = []
        for root_id in roots:
            root_msg = msg_by_id[root_id]

            # Filter to English only (oasst1 has many languages)
            if english_only and root_msg.get("lang", "en") != "en":
                continue

            for path in get_all_paths(root_id, []):
                # Need at least user + assistant (1 exchange)
                if len(path) >= 2:
                    conversations.append(path)

        return conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        """Convert a conversation into (input_ids, labels) for training."""
        conv = self.conversations[idx]
        return _format_conversation(
            conv, self.tokenizer,
            self.user_ids, self.assistant_ids, self.end_ids,
            self.block_size,
        )


# ============================================================
# ALPACA DATASET - Single-turn instructions
# ============================================================

class AlpacaDataset(Dataset):
    """
    Loads Stanford Alpaca (cleaned) as single-turn conversations.

    Alpaca format:
        instruction: "Give three tips for staying healthy."
        input: ""
        output: "1. Eat a balanced diet..."

    We convert each to our chat format:
        USER_TOKEN
        Give three tips for staying healthy.
        ASSISTANT_TOKEN
        1. Eat a balanced diet...
        END_TOKEN

    Why include Alpaca?
    - It adds 52K instruction-following examples
    - Single-turn but diverse (coding, writing, reasoning, etc.)
    - Complements OASST1's multi-turn but smaller dataset
    - Helps the model learn to follow instructions precisely
    """

    def __init__(self, block_size=1024, split="train"):
        self.block_size = block_size
        self.tokenizer = get_tokenizer()

        # Pre-encode special tokens
        self.user_ids = self.tokenizer.encode_ordinary(USER_TOKEN)
        self.assistant_ids = self.tokenizer.encode_ordinary(ASSISTANT_TOKEN)
        self.end_ids = self.tokenizer.encode_ordinary(END_TOKEN)

        # Load dataset
        print(f"Loading yahma/alpaca-cleaned...")
        ds = load_dataset("yahma/alpaca-cleaned", split=split)

        # Convert to conversation format
        self.conversations = []
        for row in ds:
            instruction = row["instruction"]
            inp = row.get("input", "")
            output = row["output"]

            # Skip empty outputs
            if not output.strip():
                continue

            # Combine instruction + input into the user message
            if inp.strip():
                user_text = instruction + "\n\nInput: " + inp
            else:
                user_text = instruction

            # Store as a 2-turn conversation (user + assistant)
            self.conversations.append([
                ("prompter", user_text),
                ("assistant", output),
            ])

        print(f"  Loaded {len(self.conversations)} instruction pairs")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        """Same format as OasstDataset - reuse the shared formatting function."""
        conv = self.conversations[idx]
        return _format_conversation(
            conv, self.tokenizer,
            self.user_ids, self.assistant_ids, self.end_ids,
            self.block_size,
        )


# ============================================================
# COMBINED DATASET + DATALOADER
# ============================================================

def create_finetune_datasets(block_size=1024):
    """
    Create combined training and validation datasets from all sources.

    Returns:
        train_dataset: ConcatDataset of OASST1 (train) + Alpaca
        val_dataset:   OASST1 validation set only (held-out conversations)

    Why ConcatDataset?
    - PyTorch's ConcatDataset lets us merge multiple datasets into one
    - The DataLoader then shuffles across all sources automatically
    - Each epoch sees examples from both OASST1 and Alpaca
    """
    print("\n" + "=" * 60)
    print("LOADING FINE-TUNING DATASETS")
    print("=" * 60)

    # --- Training data ---
    oasst_train = OasstDataset(block_size=block_size, split="train", english_only=True)
    alpaca_train = AlpacaDataset(block_size=block_size, split="train")

    # Combine into one dataset
    train_dataset = ConcatDataset([oasst_train, alpaca_train])

    # --- Validation data ---
    # Only use OASST1 validation (Alpaca has no val split)
    val_dataset = OasstDataset(block_size=block_size, split="validation", english_only=True)

    total_train = len(train_dataset)
    print(f"\n  TOTAL TRAINING EXAMPLES: {total_train:,}")
    print(f"    - OASST1 (all paths): {len(oasst_train):,}")
    print(f"    - Alpaca-cleaned:     {len(alpaca_train):,}")
    print(f"  VALIDATION EXAMPLES:    {len(val_dataset):,}")
    print("=" * 60 + "\n")

    return train_dataset, val_dataset


def create_finetune_dataloaders(train_dataset, val_dataset, micro_batch_size=8):
    """
    Create DataLoaders with proper shuffling and batching.

    Key settings:
    - shuffle=True for training (see different order each epoch)
    - shuffle=False for validation (reproducible eval)
    - num_workers=2 for async data loading (keeps GPU fed)
    - pin_memory=True for faster CPU->GPU transfer
    - drop_last=True for training to avoid small final batches
      that could cause gradient accumulation issues
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,       # Random order each epoch
        num_workers=2,       # Parallel data loading
        pin_memory=True,     # Faster GPU transfer
        drop_last=True,      # Avoid tiny final batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,       # Same order every eval
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    """
    Run this file directly to test the datasets:
        python dataset_finetune.py
    """
    tokenizer = get_tokenizer()

    # Load datasets
    train_ds, val_ds = create_finetune_datasets(block_size=1024)

    # Show a sample
    print("\n=== SAMPLE CONVERSATION ===")
    x, y = train_ds[0]
    print(f"Input shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Non-masked tokens: {(y != -1).sum().item()} / {y.shape[0]}")

    # Decode the input to see the conversation
    text = tokenizer.decode(x.tolist())
    print(f"\nDecoded (first 500 chars):")
    print(text[:500])

    # Show label coverage stats
    print("\n=== DATASET STATISTICS ===")
    total_tokens = 0
    masked_tokens = 0
    for i in range(min(100, len(train_ds))):
        _, yi = train_ds[i]
        total_tokens += yi.shape[0]
        masked_tokens += (yi != -1).sum().item()
    pct = masked_tokens / total_tokens * 100
    print(f"Label coverage (first 100 samples): {pct:.1f}% of tokens have loss")
    print(f"  (The rest are user turns / padding / special tokens)")