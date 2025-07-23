
# ✨ Challenge: Build Your Own Transformer from Scratch

This project showcases a **minimal yet complete implementation** of a decoder-only Transformer, trained on a small poetic dataset for character-level text generation.

---

## 🧠 Overview

- Implements core components of a Transformer **from scratch** (without high-level libraries)
- Learns to generate poetic text **character by character**
- Predicts the next token based on context using **self-attention**
- Demonstrates creative text generation from prompt words like: `life`, `night`, `journey`

---

## 📁 Project Structure

```
mini-transformer/
├── poem.txt          # Training dataset
├── train.py          # Model training script
├── generate.py       # Script to generate text from a prompt
├── model.pt          # Trained model checkpoint
├── transformer.py    # Core model architecture
└── .gitignore        # Ignore cache and weights in version control
```

---

## 🧱 Key Components

- **Token + Positional Embeddings**: Encode input character and position
- **Masked Multi-Head Self-Attention**: Enables autoregressive learning
- **Feedforward Layers**: Nonlinear transformations on hidden states
- **Residual Connections + LayerNorm**: Improve gradient flow and training stability

---

## 📊 Training Summary

- **Dataset**: Small poetic lines (100–500 characters)
- **Context Size**: 64
- **Epochs**: 300
- **Final Loss**: ~0.15
- **Optimizer**: Adam with learning rate 1e-3

---

## ✨ Sample Generation

Prompt → Output:

```
life:
The sun rises beyond the hills with quiet grace.
Each morning brings a promise not yet broken.

journey:
Each step forward rewrites the past behind.
Footprints fade but meaning stays.
```

> 💡 The model adapts style and theme based on input prompts.

---

## 🚀 How to Use

### 1. Train the Model

```bash
python train.py
```

### 2. Generate Text from Prompt

```bash
python generate.py
```

---

## 🛠 Challenges & Future Directions

### Key Learnings
- Properly implementing **masked attention**
- Managing overfitting on small datasets
- Fine-tuning residual + normalization behavior

### Improvements to Explore
- Move to **word-level tokenization**
- Train on **larger corpora** (e.g., Shakespeare, Wikipedia)
- Add an **encoder** for seq-to-seq tasks

---

This project is a great foundation to understand modern NLP models and experiment with creative text generation. 🚀
