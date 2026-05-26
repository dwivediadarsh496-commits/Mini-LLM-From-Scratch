# Mini LLM — From Scratch 🧠

An interactive character-level language model built with **PyTorch** and **Flask**. The web application allows users to configure hyperparameters, train a small transformer (GPT-lite) in real-time, view live training/validation loss curves, and auto-regressively generate text.

---

## 📁 Repository Structure

The workspace has been organized into a clean and standard layout:

```text
MINI LLM/
├── app.py                     # Flask Web App Backend (API & WebSocket-like SSE stream)
├── mini_llm_from_scratch.py   # Standalone CLI training script for a Bigram backbone
├── mini_llm_weights.pt        # Trained weights for the MiniGPT / Bigram model
├── requirements.txt           # Python package dependencies
├── Procfile                   # Process file for production web servers (e.g., Gunicorn)
├── render.yaml                # Infrastructure configuration for deployment on Render
│
├── templates/
│   └── index.html             # Interactive frontend UI (Dashboard with Chart.js)
│
└── mini_llm/                  # Local python virtual environment (virtualenv)
```

---

## 🚀 How to Run Locally

### 1. Setup Virtual Environment
If you are using the pre-existing virtual environment:
```powershell
# In PowerShell:
.\mini_llm\Scripts\activate
```

Or create a new one:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```
After starting the server, open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

### 4. Run Standalone CLI Script
To run the standalone CLI model training (using the Bigram model):
```bash
python mini_llm_from_scratch.py
```

---

## 🛠️ Features
- **Real-time Training Metrics**: Watch loss curves update line-by-line via Server-Sent Events (SSE).
- **Hyperparameter Customization**: Adjust learning rate, training steps, batch size, context length, embedding dimensions, etc.
- **Save & Load**: Easily save model weights to `mini_llm_weights.pt` and load them for text generation.
- **Top-K and Temperature Control**: Fine-tune text generation randomness and token choices directly from the UI.
