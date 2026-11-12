# **Financial Domain LLM Fine-Tuning — LoRA & QLoRA**

Fine-tuning **TinyLlama 1.1B Chat** for financial-domain question answering using **LoRA and QLoRA** parameter-efficient fine-tuning techniques.

The project explores how parameter-efficient fine-tuning can adapt a general-purpose language model to financial concepts while reducing the computational and memory requirements of full-model fine-tuning.

---

## 📌 Project Overview

Large Language Models are capable of understanding general-purpose text but may require domain adaptation to perform better on specialized tasks.

This project fine-tunes **TinyLlama 1.1B Chat** on financial-domain data to improve its ability to answer questions related to:

* Inflation
* SIP
* Mutual Funds
* Stocks
* Bonds
* Budgeting
* Income Tax
* GST
* Insurance
* PPF
* NPS
* Financial Planning

Two parameter-efficient fine-tuning approaches were explored:
**LoRA** and **QLoRA**.

---

##  **Model & Fine-Tuning Approach**

### Base Model

**TinyLlama 1.1B Chat**

Instead of updating all model parameters, the project uses **Low-Rank Adaptation (LoRA)** to train a small number of additional parameters.

### LoRA

LoRA freezes the pretrained model weights and introduces trainable low-rank adapter matrices into selected layers.

```text
TinyLlama 1.1B
      ↓
Frozen Pretrained Weights
      ↓
LoRA Adapters
      ↓
Fine-Tuning
      ↓
Financial Domain Model
```

### QLoRA

QLoRA extends LoRA by loading the pretrained model using **4-bit quantization**, significantly reducing GPU memory usage during fine-tuning.

```text
TinyLlama 1.1B
      ↓
4-bit NF4 Quantization
      ↓
Prepare for k-bit Training
      ↓
LoRA Adapters
      ↓
Fine-Tuning
      ↓
Financial Domain Model
```

### LoRA vs QLoRA

| Feature              | LoRA                        | QLoRA                                |
| -------------------- | --------------------------- | ------------------------------------ |
| Base Model           | FP16 / standard precision   | 4-bit quantized                      |
| Base Weights         | Frozen                      | Frozen                               |
| Trainable Parameters | LoRA adapters               | LoRA adapters                        |
| Memory Usage         | Lower than full fine-tuning | Lower than LoRA                      |
| Quantization         | Not required                | 4-bit NF4                            |
| Training Efficiency  | High                        | Very High                            |
| Main Advantage       | Efficient parameter tuning  | Efficient tuning with reduced memory |

**QLoRA was the primary approach used for the final financial-domain model.**

---

## 📊 Dataset

The project uses financial-domain question-answering data based on the **IBM FinQA dataset**.

**FinQA** is a financial question-answering dataset designed around financial reports and numerical reasoning. It contains questions that require reasoning over financial information, including numerical calculations.


##**Fine-Tuning Configuration**

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Base Model              | TinyLlama 1.1B Chat |
| Fine-Tuning Method      | QLoRA               |
| Quantization            | 4-bit               |
| Quantization Type       | NF4                 |
| Double Quantization     | Enabled             |
| Compute Type            | FP16                |
| LoRA Rank (`r`)         | 8                   |
| LoRA Alpha              | 16                  |
| Target Modules          | `q_proj`, `v_proj`  |
| LoRA Dropout            | 0.1                 |
| Batch Size              | 2                   |
| Epochs                  | 4                   |
| Learning Rate           | `1e-4`              |
| Maximum Sequence Length | 512                 |

---

## 📉 Training Results

The model was trained for **4 epochs** on the curated financial instruction-response dataset.

The training loss decreased during the training process:

```text
Step 20 → 13.29
Step 40 →  5.38
Step 60 →  1.73
```

This indicates that the model was learning the training examples during fine-tuning.

However, because the dataset is relatively small, **training loss alone is not sufficient to establish strong generalization**. A larger evaluation dataset and task-specific metrics would be required for a rigorous performance comparison.

---

## 🔬 Key Learning: LoRA & QLoRA

This project demonstrates the practical difference between two parameter-efficient fine-tuning techniques.

### LoRA

LoRA reduces the number of trainable parameters by adding low-rank trainable matrices to selected model layers while keeping the original model weights frozen.

### QLoRA

QLoRA combines:

* 4-bit quantization
* NF4 quantization
* LoRA adapters
* k-bit training preparation

This makes it possible to fine-tune relatively large language models with significantly lower GPU memory requirements.

The experiment helped demonstrate how **quantization and parameter-efficient fine-tuning can make LLM customization more accessible on limited GPU resources.**

---

## 🛠️ **Technologies**

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **Hugging Face Datasets**
* **PEFT**
* **BitsAndBytes**
* **Accelerate**
* **Google Colab**
* **QLoRA**
* **LoRA**

---

## 🚀 **How to Run**

### 1. Clone the Repository

```bash
git clone <https://github.com/uday2319/Fintuned-LLm-on-Financial-domain-Lora-QLora>
cd financial-llm-finetuning
```

### 2. Install Dependencies

```bash
pip install torchao
pip install "bitsandbytes>=0.46.1"
pip install transformers datasets peft accelerate
```

Alternatively:

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

Open:

```text
Financial_LLM_Finetuning.ipynb
```

using **Google Colab**.

### 4. **Run the Notebook**

Execute the notebook cells in order to:

1. Load TinyLlama
2. Configure 4-bit quantization
3. Prepare the model for k-bit training
4. Configure LoRA
5. Load and prepare the financial dataset
6. Tokenize the training examples
7. Fine-tune the model
8. Generate financial-domain responses
9. Evaluate sample outputs

---

## 📁 **Project Structure**

```text
financial-llm-finetuning/
│
├── Financial_Lora_Qlora_fintuned.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 🎯 Project Objectives

The main objectives of this project were to:

* Understand parameter-efficient LLM fine-tuning
* Implement **LoRA** and **QLoRA**
* Apply 4-bit NF4 quantization
* Fine-tune a pretrained LLM for a financial domain
* Work with financial question-answering data
* Understand GPU memory-efficient model training
* Experiment with instruction-response fine-tuning
* Analyze training loss and generated responses
---
## 👨‍💻 **Skills Demonstrated**

**Generative AI:**
LLM Fine-Tuning • LoRA • QLoRA • Instruction Tuning • Quantization

**Deep Learning:**
PyTorch • Transformers • GPU Training

**NLP:**
Tokenization • Language Modeling • Financial Question Answering

**LLM Ecosystem:**
Hugging Face Transformers • PEFT • BitsAndBytes • Datasets

**Development Environment:**
Google Colab • Python • Jupyter Notebook

---

## ⭐ Key Takeaway

This project demonstrates how **LoRA and QLoRA can efficiently adapt a pretrained language model to a specialized financial domain without performing full-model fine-tuning**.

The project particularly focuses on **memory-efficient LLM fine-tuning using 4-bit quantization and LoRA adapters**, making experimentation with domain-specific LLMs feasible on limited GPU resources.
