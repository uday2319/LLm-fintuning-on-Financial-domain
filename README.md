# Financial-Domain LLM Fine-Tuning with LoRA & QLoRA

Fine-tuning **TinyLlama-1.1B-Chat** for financial question answering with two parameter-efficient methods, **LoRA** and **QLoRA**. Both adapt a general-purpose model to financial concepts while using a fraction of the compute and memory that full fine-tuning needs.

## Overview

General-purpose LLMs understand everyday text well but often need domain adaptation for specialised topics. This project fine-tunes TinyLlama on financial question–answer data covering:

| Investing | Tax & savings schemes | Personal finance |
|---|---|---|
| Stocks, Bonds, Mutual Funds, SIP | Income Tax, GST, PPF, NPS | Inflation, Budgeting, Insurance, Financial Planning |

**QLoRA was the primary approach used for the final model.**

## Approach

### LoRA

LoRA freezes the pretrained weights and trains small low-rank adapter matrices inserted into selected layers, so only a tiny fraction of parameters is updated.

```text
TinyLlama 1.1B  →  frozen pretrained weights  →  + LoRA adapters  →  fine-tune  →  financial model
```

### QLoRA

QLoRA loads the frozen base model in **4-bit NF4** precision before adding LoRA adapters, which greatly reduces GPU memory during training.

```text
TinyLlama 1.1B  →  4-bit NF4 quantization  →  prepare for k-bit training  →  + LoRA adapters  →  fine-tune  →  financial model
```

### LoRA vs QLoRA

| | LoRA | QLoRA |
|---|---|---|
| Base model precision | FP16 | 4-bit NF4 (double quantization) |
| Base weights | Frozen | Frozen |
| Trained parameters | LoRA adapters | LoRA adapters |
| GPU memory | Lower than full fine-tuning | Lower than LoRA |
| Trade-off | Faster per step | Uses less memory but is slower per step (weights are dequantized on the fly) |
| Best for | GPUs with enough memory for the FP16 model | Limited GPUs such as a free Colab T4 |

## Dataset

Financial-domain question–answer data based on the **IBM FinQA** dataset. FinQA is built from financial reports and contains questions that require reasoning over financial information, including numerical calculations.

## Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Base model | TinyLlama-1.1B-Chat |
| Method | QLoRA |
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | FP16 |
| LoRA rank (`r`) / alpha | 8 / 16 |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.1 |
| Batch size | 2 |
| Epochs | 4 |
| Learning rate | 1e-4 |
| Max sequence length | 512 |

## Training Results

Training loss fell steadily over 4 epochs:

| Step | Training loss |
|---|---|
| 20 | 13.29 |
| 40 | 5.38 |
| 60 | 1.73 |

This shows the model is learning the training examples. Because the dataset is small, training loss alone does **not** show that the model generalises. A rigorous comparison would need a larger held-out evaluation set and task-specific metrics, such as answer accuracy.

## Key Learnings

- **LoRA** makes fine-tuning cheap by training low-rank adapters while the original weights stay frozen.
- **QLoRA** adds 4-bit NF4 quantization and k-bit training preparation on top of LoRA. This makes it possible to fine-tune LLMs on limited GPU memory, such as a free Colab GPU.
- Training loss is a useful sanity check but not a measure of answer quality, so real evaluation needs held-out data and task metrics.

## How to Run

The notebook needs a GPU. Use **Google Colab** with a **T4 GPU** runtime.

```bash
git clone https://github.com/uday2319/Fintuned-LLm-on-Financial-domain-Lora-QLora.git
cd Fintuned-LLm-on-Financial-domain-Lora-QLora
pip install -r requirements.txt
```

Then open `Financial_Lora_Qlora_fintuned.ipynb` in Colab and run the cells in order. The notebook:

1. Loads TinyLlama and configures 4-bit quantization
2. Prepares the model for k-bit training and attaches LoRA adapters
3. Loads, formats and tokenizes the financial dataset
4. Fine-tunes the model
5. Generates and reviews financial-domain answers

## Project Structure

```text
Fintuned-LLm-on-Financial-domain-Lora-QLora/
├── Financial_Lora_Qlora_fintuned.ipynb   fine-tuning notebook
├── requirements.txt                       dependencies
├── README.md
└── .gitignore
```

## Tech Stack

| Area | Tools |
|---|---|
| Fine-tuning | LoRA, QLoRA, 4-bit NF4 quantization, instruction tuning |
| Libraries | PyTorch, Hugging Face Transformers, Datasets, PEFT, BitsAndBytes, Accelerate |
| NLP | Tokenization, causal language modelling, financial question answering |
| Environment | Python, Jupyter, Google Colab (GPU) |

## Takeaway

LoRA and QLoRA can adapt a pretrained LLM to a specialised financial domain without full-model fine-tuning. QLoRA in particular, with 4-bit quantization plus LoRA adapters, makes domain-specific LLM experiments practical on limited GPU resources.
