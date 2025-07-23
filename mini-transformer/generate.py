import torch
from transformer import TransformerLM
from transformer import encode, decode, vocab_size, block_size  # importe depuis transformer.py
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres du modèle (doivent correspondre à ceux utilisés pour l'entraînement)
embed_size = 128
heads = 4
num_layers = 4
dropout = 0.1

# Charger le modèle
model = TransformerLM(vocab_size, embed_size, heads, num_layers, dropout).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/mini-transformer/model.pt", map_location=device))
model.eval()

# Fonction de génération de texte
def generate(prompt, max_new_tokens=200):
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        context_cond = context[:, -block_size:]  # tronquer à block_size
        logits = model(context_cond)
        logits = logits[:, -1, :]  # Prendre le dernier token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

    generated = decode(context[0].tolist())
    return generated

# Utilisation depuis la ligne de commande (ou modification ici du prompt directement)
if __name__ == "__main__":
    prompt = "life:"  # Modifie ici pour tester d'autres départs
    print(generate(prompt))
