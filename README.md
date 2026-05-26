# 🧠 Mini LLM — From Scratch

A **character-level GPT-lite transformer** built entirely from scratch with **PyTorch**, served through a **Flask** web application. Train a real transformer model in your browser — configure hyperparameters, watch live loss curves update via Server-Sent Events, and auto-regressively generate text from a trained model — all without leaving the UI.

🔗 **Live Demo:** [mini-llm-from-scratch-2.onrender.com](https://mini-llm-from-scratch-2.onrender.com)
📁 **Repository:** [github.com/dwivediadarsh496-commits/Mini-LLM-From-Scratch](https://github.com/dwivediadarsh496-commits/Mini-LLM-From-Scratch)

---

## ✨ Features

- **Train a real GPT-style transformer in the browser** — no setup, no CLI required
- **Live loss curves** — training and validation loss stream in real-time via SSE (Server-Sent Events)
- **Fully configurable hyperparameters** — batch size, context length, learning rate, embedding dim, training steps, dataset size
- **Text generation with controls** — set prompt, token count, temperature, and Top-K sampling from the UI
- **Save & load model weights** — weights auto-saved to `mini_llm_weights.pt` after training
- **WikiText-2 dataset** — automatically downloaded and preprocessed at training time via Hugging Face `datasets`
- **Stop training mid-run** — graceful stop with a flag-based interrupt mechanism
- **CPU & CUDA support** — automatically uses GPU if available
- **Standalone CLI mode** — run training headlessly via `mini_llm_from_scratch.py`

---

## 🏗️ Model Architecture — MiniGPT

The model is a decoder-only transformer (GPT-style) built from scratch in PyTorch, operating at the **character level**.

```
Input Characters
      │
      ▼
Token Embedding  +  Positional Embedding
      │
      ▼
┌─────────────────────────────┐
│   Transformer Block × 4     │
│  ┌───────────────────────┐  │
│  │ LayerNorm              │  │
│  │ Multi-Head Attention   │  │  ← 4 heads, causal masking
│  │  (4 × Head)            │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │ LayerNorm              │  │
│  │ FeedForward (GELU)     │  │  ← 4× expansion, dropout
│  └───────────────────────┘  │
└─────────────────────────────┘
      │
      ▼
   LayerNorm
      │
      ▼
  Linear (LM Head)
      │
      ▼
Character Probabilities (Vocab)
```

### Default Architecture

| Component | Value |
|---|---|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Context length (block size) | 64 tokens |
| Dropout | 0.1 |
| Feed-forward expansion | 4× |
| Activation | GELU |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |

---

## ⚙️ Configurable Hyperparameters

All of these can be set from the web UI before starting training:

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 32 | Number of sequences per training step |
| `block_size` | 64 | Context length (tokens per sequence) |
| `max_iters` | 5000 | Total training steps |
| `lr` | 3e-4 | AdamW learning rate |
| `subset_rows` | 1000 | Rows of WikiText-2 to use (controls dataset size) |
| `embed_dim` | 128 | Embedding & hidden dimension |
| `temperature` | 0.8 | Generation temperature (higher = more random) |

---

## 🔄 How It Works

### Training Flow

```
Browser  ──[POST /api/train + hyperparams]──►  train_worker() [background thread]
                                                      │
                                              Load WikiText-2 dataset
                                              Build character vocab
                                              Instantiate MiniGPT
                                              AdamW training loop
                                                      │
Browser  ◄──[GET  /api/stream  SSE]──────────  yield step, loss, val_loss
                                                      │
                                              Save mini_llm_weights.pt
```

### Generation Flow

```
Browser  ──[POST /api/generate + prompt + params]──►  MiniGPT.generate()
                                                           │
                                              encode prompt → token ids
                                              auto-regressive loop
                                              temperature scaling + Top-K sampling
                                              decode ids → characters
                                                           │
Browser  ◄────────────────────────────────  { "text": "generated output..." }
```

---

## 📁 Project Structure

```
Mini-LLM-From-Scratch/
├── app.py                    # Flask backend — training worker, SSE stream, API routes
├── mini_llm_from_scratch.py  # Standalone CLI training script (Bigram backbone)
├── mini_llm_weights.pt       # Pre-trained MiniGPT weights (committed to repo)
├── requirements.txt          # Python dependencies
├── Procfile                  # Gunicorn process definition for Render/Heroku
├── render.yaml               # Render deployment configuration
├── .gitignore
└── templates/
    └── index.html            # Interactive dashboard UI (Chart.js loss curves)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/dwivediadarsh496-commits/Mini-LLM-From-Scratch.git
cd Mini-LLM-From-Scratch

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

From the dashboard you can:
1. Adjust hyperparameters in the form
2. Click **Start Training** — loss curves update live
3. Click **Stop** to interrupt training at any time
4. Enter a prompt and click **Generate** to sample text
5. Click **Save Weights** to persist the model to disk

### Run CLI Training (Standalone)

```bash
python mini_llm_from_scratch.py
```

This runs the simpler Bigram model variant headlessly in the terminal.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve the web UI |
| POST | `/api/train` | Start training with custom hyperparameters (JSON body) |
| POST | `/api/stop` | Stop an in-progress training run |
| GET | `/api/status` | Get current training state (step, losses, status, message) |
| GET | `/api/stream` | SSE stream of live training updates |
| POST | `/api/generate` | Generate text from the trained model |
| POST | `/api/save` | Save model weights to `mini_llm_weights.pt` |

### Example: Start Training

```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 32,
    "block_size": 64,
    "max_iters": 3000,
    "lr": 3e-4,
    "subset_rows": 800,
    "embed_dim": 128
  }'
```

### Example: Generate Text

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The history of",
    "tokens": 300,
    "temperature": 0.8,
    "top_k": 40
  }'
```

**Response:**
```json
{
  "ok": true,
  "text": "The history of the world was a great deal of the most important..."
}
```

### SSE Stream Format

The `/api/stream` endpoint pushes newline-delimited JSON events:

```
data: {"status": "training", "step": 150, "train_loss": [2.31, 2.18, ...], "val_loss": [...], "message": "Step 150 · loss 2.1823"}

data: {"status": "done", "message": "Training complete! Weights saved."}

data: {"_close": true}
```

---

## 📊 Training State Object

The `/api/status` response and SSE stream both return a full state object:

```json
{
  "status": "training",       // idle | loading | training | done | error
  "step": 450,
  "max_iters": 5000,
  "train_loss": [2.45, 2.31, 2.18],
  "val_loss": [2.51, 2.38, 2.22],
  "steps": [0, 50, 100],
  "generated": "",
  "vocab_size": 67,
  "message": "Step 450 · loss 2.1823",
  "device": "cpu",
  "params": 1345600
}
```

---

## ☁️ Deployment (Render)

The repo is pre-configured for Render with `render.yaml` and a `Procfile`:

```
web: gunicorn app:app --timeout 120 --workers 1
```

> **Note:** Training is CPU-bound on Render's free tier. Use a smaller `subset_rows` (e.g. 300–500) and fewer `max_iters` (e.g. 1000–2000) for reasonable training times on the live demo.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch (transformer built from scratch) |
| Dataset | Hugging Face `datasets` — WikiText-2 |
| Backend | Python, Flask |
| Streaming | Server-Sent Events (SSE) |
| Frontend Charts | Chart.js |
| Frontend UI | HTML, CSS, JavaScript |
| Deployment | Render (Gunicorn) |

---

## 📊 Languages

- HTML — 56.2%
- Python — 43.7%
- Procfile — 0.1%

---

## 🤝 Contributing

Contributions welcome! Ideas for extending the project:

- Add support for custom training text (file upload)
- Implement word-level or BPE tokenization
- Add perplexity metric to the dashboard
- Support multi-layer depth control from the UI
- Add beam search generation mode

```bash
git checkout -b feature/your-idea
# Make changes
git commit -m "Add: ..."
git push origin feature/your-idea
# Open a pull request
```

---

## 📄 License

Open source. See the repository for license details.

---

> *Built to demystify transformers — every layer, every weight, from scratch.* 🔬
