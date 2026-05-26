import os, json, threading, time, math
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from datasets import load_dataset

app = Flask(__name__)

state = {
    "status"      : "idle",
    "step"        : 0,
    "max_iters"   : 5000,
    "train_loss"  : [],
    "val_loss"    : [],
    "steps"       : [],
    "generated"   : "",
    "vocab_size"  : 0,
    "message"     : "",
    "device"      : "cpu",
    "params"      : 0,
}

model_ref   = {"model": None}
enc_ref     = {"stoi": None, "itos": None, "train_data": None, "val_data": None}
lock        = threading.Lock()
stop_flag   = threading.Event()

DEFAULT_HP = dict(
    batch_size  = 32,
    block_size  = 64,
    max_iters   = 5000,
    lr          = 3e-4,
    subset_rows = 1000,
    temperature = 0.8,
    embed_dim   = 128,
)

def build_vocab(text):
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos

def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]

def decode(lst, itos):
    return "".join(itos.get(i, "?") for i in lst)

# --- Transformer Model (GPT-lite) ---

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x);   q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ff  = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size      = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding   = nn.Embedding(block_size, n_embd)
        self.blocks          = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f            = nn.LayerNorm(n_embd)
        self.lm_head         = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok  = self.token_embedding(idx)
        pos  = self.pos_embedding(torch.arange(T, device=idx.device))
        x    = self.blocks(self.ln_f(tok + pos))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs    = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, next_idx), dim=1)
        return idx

def get_batch(split, block_size, batch_size, device):
    data = enc_ref["train_data"] if split == "train" else enc_ref["val_data"]
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i     : i+block_size  ] for i in ix])
    y    = torch.stack([data[i+1   : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Training Worker Thread ---

def train_worker(hp: dict):
    global state
    stop_flag.clear()

    try:
        with lock: state["status"] = "loading"; state["message"] = "Loading WikiText-2 dataset…"
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text    = " ".join(dataset["train"]["text"][:hp["subset_rows"]])

        chars, stoi, itos = build_vocab(text)
        vocab_size = len(chars)
        data = torch.tensor(encode(text, stoi), dtype=torch.long)
        n    = int(0.9 * len(data))

        enc_ref["stoi"]       = stoi
        enc_ref["itos"]       = itos
        enc_ref["train_data"] = data[:n]
        enc_ref["val_data"]   = data[n:]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with lock:
            state["vocab_size"] = vocab_size
            state["device"]     = device
            state["message"]    = f"Dataset loaded · {len(text):,} chars · vocab {vocab_size}"

        n_embd = hp["embed_dim"]
        mdl    = MiniGPT(vocab_size, n_embd=n_embd, n_head=4, n_layer=4,
                         block_size=hp["block_size"]).to(device)
        model_ref["model"] = mdl
        params = sum(p.numel() for p in mdl.parameters())

        opt = torch.optim.AdamW(mdl.parameters(), lr=hp["lr"])

        with lock:
            state["params"]    = params
            state["max_iters"] = hp["max_iters"]
            state["status"]    = "training"
            state["step"]      = 0
            state["train_loss"] = []
            state["val_loss"]   = []
            state["steps"]      = []
            state["message"]   = f"Training started · {params:,} parameters"

        eval_every = max(50, hp["max_iters"] // 100)
        for it in range(hp["max_iters"]):
            if stop_flag.is_set():
                with lock: state["status"] = "idle"; state["message"] = "Training stopped by user."
                return

            xb, yb = get_batch("train", hp["block_size"], hp["batch_size"], device)
            _, loss = mdl(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()

            if it % eval_every == 0 or it == hp["max_iters"] - 1:
                mdl.eval()
                with torch.no_grad():
                    xv, yv    = get_batch("val", hp["block_size"], hp["batch_size"], device)
                    _, vloss  = mdl(xv, yv)
                mdl.train()
                with lock:
                    state["step"]       = it
                    state["steps"].append(it)
                    state["train_loss"].append(round(loss.item(), 4))
                    state["val_loss"].append(round(vloss.item(), 4))
                    state["message"] = f"Step {it} · loss {loss.item():.4f}"

        torch.save(mdl.state_dict(), "mini_llm_weights.pt")
        with lock:
            state["status"]  = "done"
            state["step"]    = hp["max_iters"]
            state["message"] = "Training complete! Weights saved."

    except Exception as e:
        with lock:
            state["status"]  = "error"
            state["message"] = str(e)

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/train", methods=["POST"])
def api_train():
    if state["status"] == "training":
        return jsonify({"ok": False, "msg": "Already training"})
    body = request.get_json(force=True, silent=True)
    if body is None:
        body = {}
    hp = dict(DEFAULT_HP)
    hp.update(body)
    t  = threading.Thread(target=train_worker, args=(hp,), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_flag.set()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    with lock:
        return jsonify(dict(state))

@app.route("/api/stream")
def api_stream():
    def event_gen():
        last_step = -1
        while True:
            time.sleep(0.8)
            with lock:
                s = dict(state)
            if s["step"] != last_step or s["status"] in ("done", "error", "idle"):
                last_step = s["step"]
                yield f"data: {json.dumps(s)}\n\n"
            if s["status"] in ("done", "error"):
                yield "data: {\"_close\": true}\n\n"
                break
    return Response(stream_with_context(event_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    mdl  = model_ref["model"]
    stoi = enc_ref["stoi"]
    itos = enc_ref["itos"]
    if mdl is None or stoi is None:
        return jsonify({"ok": False, "msg": "Model not trained yet."})

    body   = request.get_json(force=True, silent=True) or {}
    prompt = body.get("prompt", "")
    n_tok  = int(body.get("tokens", 200))
    temp   = float(body.get("temperature", 0.8))
    top_k  = int(body.get("top_k", 40))
    device = state["device"]

    if prompt:
        encoded = encode(prompt, stoi)
        if not encoded:
            return jsonify({"ok": False, "msg": "Prompt contains unknown characters."})
        start = torch.tensor([encoded], dtype=torch.long, device=device)
    else:
        start = torch.zeros((1, 1), dtype=torch.long, device=device) # null token se start kiya generation

    mdl.eval()
    with torch.no_grad():
        out = mdl.generate(start, max_new_tokens=n_tok, temperature=temp, top_k=top_k)
    text = decode(out[0].tolist(), itos)
    return jsonify({"ok": True, "text": text})

@app.route("/api/save", methods=["POST"])
def api_save():
    mdl = model_ref["model"]
    if mdl is None:
        return jsonify({"ok": False, "msg": "No model to save."})
    torch.save(mdl.state_dict(), "mini_llm_weights.pt")
    return jsonify({"ok": True, "msg": "Saved to mini_llm_weights.pt"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, threaded=True, port=port)
