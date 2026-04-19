# GPT-2 Training Summary

**Date**: 2026-04-19 11:15:27
**Duration**: 7h27m
**Device**: Tesla T4

## Model Architecture

| Parameter | Value |
|---|---|
| Layers | 12 |
| Attention Heads | 12 |
| Hidden Dim | 768 |
| Vocab Size | 50,257 |
| Max Seq Length | 1024 |
| Dropout | 0.0 |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Total Steps | 21,576 |
| Tokens Trained | 0.44B |
| Peak LR | 2e-05 |
| Min LR | 2e-06 |
| Warmup Steps | 200 |
| Weight Decay | 0.01 |
| Micro Batch Size | 10 |
| Grad Accumulation | 2 |
| Effective Batch | 20 |
| Gradient Clip | 1.0 |

## Results

| Metric | Value |
|---|---|
| Initial Loss (avg first 100) | 3.2005 |
| Final Loss (avg last 100) | 2.5032 |
| Min Training Loss | 1.8899 |
| Best Val Loss | 2.7300 |
| Best Val Perplexity | 15.33 |
| Avg Throughput | 16447 tokens/sec |
| Total Training Time | 7h27m |

## Validation History

| Step | Val Loss | Perplexity |
|---|---|---|
| 500 | 2.7815 | 16.14 |
| 1,000 | 2.7412 | 15.51 |
| 1,500 | 2.7320 | 15.36 |
| 2,000 | 2.7300 | 15.33 |
| 2,500 | 2.7318 | 15.36 |
| 3,000 | 2.7390 | 15.47 |
| 3,500 | 2.7365 | 15.43 |
| 4,000 | 2.7481 | 15.61 |
| 4,500 | 2.7561 | 15.74 |
| 5,000 | 2.7625 | 15.84 |
| 5,500 | 2.7674 | 15.92 |
| 6,000 | 2.7672 | 15.91 |
| 6,500 | 2.7718 | 15.99 |
| 7,000 | 2.7767 | 16.07 |
| 7,500 | 2.7822 | 16.16 |
| 8,000 | 2.7886 | 16.26 |
| 8,500 | 2.7861 | 16.22 |
| 9,000 | 2.7891 | 16.27 |
| 9,500 | 2.7913 | 16.30 |
| 10,000 | 2.7956 | 16.37 |
| 10,500 | 2.7958 | 16.37 |
| 11,000 | 2.8054 | 16.53 |
| 11,500 | 2.8077 | 16.57 |
| 12,000 | 2.8033 | 16.50 |
| 12,500 | 2.8029 | 16.49 |
| 13,000 | 2.8035 | 16.50 |
| 13,500 | 2.8093 | 16.60 |
| 14,000 | 2.8052 | 16.53 |
| 14,500 | 2.8098 | 16.61 |
| 15,000 | 2.8107 | 16.62 |
| 15,500 | 2.8120 | 16.64 |
| 16,000 | 2.8083 | 16.58 |
| 16,500 | 2.8100 | 16.61 |
| 17,000 | 2.8117 | 16.64 |
| 17,500 | 2.8114 | 16.63 |
| 18,000 | 2.8120 | 16.64 |
| 18,500 | 2.8118 | 16.64 |
| 19,000 | 2.8121 | 16.64 |
| 19,500 | 2.8127 | 16.66 |
| 20,000 | 2.8135 | 16.67 |
| 20,500 | 2.8142 | 16.68 |
| 21,000 | 2.8134 | 16.67 |
| 21,500 | 2.8144 | 16.68 |

## Generated Samples

### Step 21,000
**Prompt**: Write a short poem about the ocean.

```
The ocean flows in a gentle breeze,
As the sun sets, the ocean turns to wilt 
In the sky, a gentle breeze,
The ocean views all the stars in its path,
A rustling sight, a sight to behold 

The ocean's calmness is like a light breeze,
A peaceful breeze, a gentle breeze,
A serene place, a gentle breeze
```

### Step 21,500
**Prompt**: Hello! How are you doing today?

```
I'm doing everything right now. Is there anything you need help with today?
```

### Step 21,500
**Prompt**: What is machine learning?

```
Machine learning is a field of machine learning that involves the development of artificial intelligence and machine learning algorithms. It is a subset of machine learning that involves the training of algorithms to learn from data and make predictions and decisions. Machine learning is designed to
```

### Step 21,500
**Prompt**: Write a short poem about the ocean.

```
Verse 1:
The ocean is full of love,
With its gentle waves,
A peaceful sea that's free from the rain.

Chorus:
Oce of love,
A warm breeze,
The waves, the symphony of love,
In the center of the world.

Verse 2:
The ocean is full of love,
A warm breeze,
The waves, and the symphony of love,
The waves, t
```

### Step 21,576
**Prompt**: FINAL

```
User: Hello! How are you doing today?
GPT-2: I am writing toHi everyone, how may I help you today?

User: What is machine learning?
GPT-2: Machine learning is a type of artificial intelligence that involves the use of artificial intelligence, such as the development of algorithms that can take advan
```

## Log Files

- `training_log.csv` - Per-step metrics (for plotting)
- `training_log.jsonl` - Detailed logs (machine readable)
- `eval_log.csv` - Validation results
- `samples.txt` - All generated text samples
