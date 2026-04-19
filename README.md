# GPT-2 Conversational Model (Trained from Scratch)

This repository contains the training pipeline, configuration, and insights from building and training a 124M parameter GPT-2 model completely from scratch using PyTorch. 

The project encompasses two main phases:
1. **Pre-training**: Training the base causal language model on the OpenWebText dataset.
2. **Fine-tuning**: Adapting the model for conversational interactions using the OpenAssistant (OASST1) dataset.

## System Specifications & Total Training Time

The full system pipeline was executed on a cloud instance with the following specifications:
- **GPU**: 1x NVIDIA Tesla T4 (15GB VRAM)
- **CPU**: Intel Xeon Platinum 8259CL @ 2.50GHz (4 vCPUs)
- **RAM**: 15 GB 
- **Total Training Time**: ~29.1 hours 
  - *Pre-training*: 21.67 hours (78,027 seconds)
  - *Fine-tuning*: 7.43 hours (26,758 seconds)

## Datasets & Final Metrics

### Phase 1: Pre-training
- **Dataset**: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)
  - *Details*: An open-source recreation of OpenAI's WebText dataset. It consists of text web-scraped from outbound URLs from Reddit submissions that received at least 3 upvotes, ensuring a baseline of quality. 
  - *Scale Used*: Processed approximately 1.3 Billion tokens during the pre-training run.
- **Final Metrics**: Completed 25,500 steps, achieving a final validation loss of **3.362** and a perplexity of **28.86**.
- **Logs**: Training and evaluation logs can be found locally at `checkpoints/pretrain/training_log.csv` and `checkpoints/pretrain/eval_log.csv`.

### Phase 2: Fine-tuning
- **Dataset**: [OpenAssistant / OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - *Details*: A high-quality, human-generated and human-annotated assistant-style conversation corpus. The dataset focuses on typical assistant interactions (multi-turn conversations, instruction-following) created via crowdsourcing.
- **Final Metrics**: The fine-tuning phase achieved its optimal validation loss of **2.730** (perplexity **15.33**) at step 2,000.
- **Logs**: Fine-tuning logs are tracked locally at `checkpoints/finetune/training_log.csv` and `checkpoints/finetune/eval_log.csv`.

## Outcomes and Learnings

This project served as a deep dive into Large Language Model (LLM) architecture, training pipelines, and dataset processing. Some of the key learnings and outcomes include:

1. **Architecture Implementation**: 
   - Successfully implemented the full GPT-2 transformer architecture from scratch, including multi-head causal self-attention, feed-forward networks with GELU, pre-LayerNorm blocks, and weight tying between embeddings and the output head.
2. **Training Optimization**: 
   - Maximized hardware utilization on a single NVIDIA Tesla T4 GPU (16GB VRAM) using mixed-precision training (FP16), gradient accumulation, and learning rate scheduling (cosine decay with warmup). 
   - Managed limitations of hardware by identifying the optimal micro-batch size to prevent Out of Memory (OOM) errors while maintaining efficient throughput.
3. **Data Pipelines**:
   - Learned how to efficiently construct pipelines for both continuous pretraining data (OpenWebText) and turn-based conversational formatting (OASST1 with masked assistant loss).
4. **Hugging Face Hub Integration**:
   - Developed scripts to port the custom PyTorch model into a Hugging Face Hub compatible format, generating the necessary `config.json` and weight maps.

## Limitations: A Toy Model

⚠️ **Important Disclaimer**: This model is fundamentally a **toy model** intended for educational and research purposes. 

While the model has learned structural language patterns and can technically engage in conversational turns, **its outputs are often not very meaningful, coherent, or factually accurate**. 

The main limitations are:
- **Small Scale**: At only 124 million parameters, it lacks the capacity to store substantial world knowledge or exhibit complex reasoning capabilities found in modern LLMs (which are typically billions of parameters).
- **Compute constraints**: It was trained on limited data and for a short period of time (a single Tesla T4 GPU for ~24 hours is barely a fraction of the compute required for production-level models).
- **Hallucinations**: The model will frequently make up facts, contradict itself, and lose conversational context over multiple turns.
- **Safety**: There are no safety guardrails, RLHF, or alignment techniques applied to this model. It may produce unfiltered or inappropriate text depending on the prompt.

**Bottom-line**: It is highly recommended to view this model as an exercise in "how to build and train an LLM" rather than a functional conversational assistant.

## Hugging Face Models

Both the pre-trained base model and the conversational fine-tuned model have been uploaded to the Hugging Face Hub. 
- *The `main` branch contains the conversational fine-tuned model.*
- *The `pretrained` branch contains the base pre-trained model.*

## Repository Structure

```
.
├── config.py             # Model hyperparameters and configurations
├── model.py              # GPT-2 architecture implemented in PyTorch
├── dataset.py            # Pre-training dataset loader
├── dataset_finetune.py   # Conversational fine-tuning dataset loader
├── train_pretrain.py     # Pre-training loop
├── train_finetune.py     # Fine-tuning loop
├── export_hf.py          # Script to export weights to Hugging Face format
├── chat.py               # Interactive CLI chat script
├── plot_training.py      # Utility for visualizing training loss/metrics
└── checkpoints/          # Local directory for model weights and logs
```

## Setup & Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Acknowledgments

This project relied on datasets from [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) and [OpenAssistant (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1).
