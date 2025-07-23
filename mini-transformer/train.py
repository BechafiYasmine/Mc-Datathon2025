
import torch
from torch.utils.data import DataLoader
from transformer import TransformerLM, CharDataset, encode

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and encode text
with open("/content/drive/MyDrive/mini-transformer/poem.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
dataset = CharDataset(data, block_size=64)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model setup
model = TransformerLM(vocab_size=sorted(list(set(text))).__len__(), embed_size=128, heads=4, num_layers=4, dropout=0.1)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(300):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0 or epoch == 299:
        print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "/content/drive/MyDrive/mini-transformer/model.pt")
print("✅ Model saved.")
