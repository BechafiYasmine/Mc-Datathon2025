
# ✨ Challenge: Build Your Own Transformer from Scratch

This is a minimal implementation of a decoder-only Transformer trained on a small poetic dataset.

## 🧠 What It Does

- Learns from a small paragraph of text  
- Predicts the next character one-by-one  
- Generates creative continuations from a prompt  

## 📁 Files

- `transformer.py` - Model & training logic  
- `train.py` - Training script  
- `generate.py` - Sampling/generation script  
- `poem.txt` - Your training data  
- `model.pt` - Trained model checkpoint  

## 🚀 How to Run

### 1. Train the model

    python train.py

### 2. Generate text

    python generate.py

---

✨ Challenge: Build Your Own Transformer from Scratch

🔥 Minimal Transformer — Character-Level Text Generator

This project implements a decoder-only Transformer from scratch (like GPT) that learns to generate poetic text character by character.

🧠 What It Does
- Learns patterns from a small poetic dataset
- Predicts the next character given a prompt
- Produces fluent poetic continuations (e.g., for words like life, journey, night)

🧱 Core Components
- Token + Positional Embeddings
- Multi-head Self-Attention
- Feedforward layers
- Masked attention for autoregressive behavior
- Residual connections + LayerNorm

📈 Training Summary
- Dataset: Small poetic lines (100–500 characters)
- Context size: 64 tokens
- Epochs: 300 — final loss ≈ 0.15
- Optimizer: Adam (lr = 1e-3)

✨ Results
- Prompts like life, journey, night produce creative, structured poetic text
- Clear improvement in fluency and theme control
- Shows how Transformers handle sequence generation even with minimal data

🛠 Challenges & Ideas
- Difficulties: Implementing masked attention, overfitting on small data
- Future: Move to word-level tokens, add an encoder, use a larger corpus
