import torch
import torch.nn as nn
from datasets import load_dataset

print("Loading WikiText-2 dataset …")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text = " ".join(dataset["train"]["text"][:1000])
print(f"  Total characters in training slice: {len(text):,}")

# Vocab and Tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"  Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(lst: list[int]) -> str:
    return "".join([itos[i] for i in lst])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"  Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# Hyperparameters
batch_size = 32
block_size = 64
max_iters  = 5000
lr         = 3e-4
device     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x  = torch.stack([src[i      : i + block_size    ] for i in ix])
    y  = torch.stack([src[i + 1  : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding(idx)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits  = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss    = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits    = logits[:, -1, :] / temperature
            probs     = torch.softmax(logits, dim=-1)
            next_idx  = torch.argmax(probs, dim=-1, keepdim=True)
            idx       = torch.cat((idx, next_idx), dim=1)
        return idx

model     = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}\n")

print("=" * 50)
print("Starting training …")
print("=" * 50)

for iteration in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iteration % 500 == 0:
        xv, yv = get_batch("val")
        _, val_loss = model(xv, yv)
        print(f"  Step {iteration:>5}  |  Train Loss: {loss.item():.4f}  |  Val Loss: {val_loss.item():.4f}")

print("=" * 50)
print("Training complete!")
print("=" * 50)

torch.save(model.state_dict(), "mini_llm_weights.pt")
print("\nModel weights saved to: mini_llm_weights.pt")

# Text Generation
model.eval()
start     = torch.zeros((1, 1), dtype=torch.long).to(device) # null token se start kiya generation
generated = model.generate(start, max_new_tokens=300, temperature=0.8)

print("\n" + "=" * 50)
print("Generated Text:")
print("=" * 50)
print(decode(generated[0].tolist()))
print("=" * 50)
