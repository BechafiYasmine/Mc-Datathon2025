
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# Vocabulary setup (you can edit based on your dataset)
with open("poem.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

block_size = 64

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, embed_size = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        energy = torch.einsum("nqe,nke->nqk", [queries, keys])
        attention = F.softmax(energy / math.sqrt(self.embed_size), dim=2)
        out = torch.einsum("nqk,nke->nqe", [attention, values])
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.attention(x)
        x = self.norm1(x + self.dropout(attn))
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.fc_out(x)
