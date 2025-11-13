<div align="center">

# 🎬 **Sentiment Classification with RNN Architectures**
### _Comparative Analysis • Homework 3 — NLP • University of Maryland (DATA 643)_

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange?logo=pytorch)
![NLP](https://img.shields.io/badge/NLP-Sentiment_Analysis-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

</div>

---

# 🌟 **Overview**

This repository contains a **complete, reproducible NLP pipeline** for sentiment classification using the **IMDb 50k Movie Review Dataset**.  
We perform a systematic comparison of:

- 🧠 **Architectures:** `RNN`, `LSTM`, `BiLSTM`  
- ⚡ **Activations:** `ReLU`, `Tanh`, `Sigmoid`  
- 🔧 **Optimizers:** `Adam`, `SGD`, `RMSProp`  
- 📏 **Sequence Lengths:** `25`, `50`, `100`  
- 🛡 **Gradient Clipping:** On / Off  

Our goal is to understand how these choices affect:
- Accuracy
- F1 Score
- Training Time
- Training Stability  
on **CPU-only hardware**.

---

# 📁 **Repository Structure**

```plaintext
homework3-NLP/
├── data/
│   ├── raw/                     # Original IMDb CSV
│   └── processed/               # Tokenized + padded datasets
│
├── results/
│   ├── metrics.csv              # Logged experiments
│   ├── model_*.pt               # Model checkpoints
│   └── plots/                   # Auto-generated bar plots
│
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
│
├── README.md                    # (This file)
├── requirements.txt
└── report.pdf
```

---

# 📦 **Dataset**

**IMDb Movie Review Dataset (50,000 labeled reviews)**  
📌 Source: <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>

### Preprocessing includes:
- Lowercasing  
- Removing punctuation  
- Tokenization (custom `LiteTokenizer`)  
- Keeping **Top 10,000** most frequent words  
- Padding/truncating sequences to: `25`, `50`, `100`  

Outputs saved to `data/processed/`.

---

# ⚙️ **Setup Instructions**

### 1️⃣ Create & activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

# 🧼 **Run Preprocessing**

Place the IMDb CSV in `data/raw/IMDB Dataset.csv`, then run:

```bash
python -m src.preprocess   --raw_csv "data/raw/IMDB Dataset.csv"   --output_dir data/processed   --top_k 10000   --seq_lengths 25 50 100   --seed 42
```

---

# 🚀 **Training Models**

### General syntax:
```bash
python -m src.train   --arch <rnn|lstm|bilstm>   --activation <relu|tanh|sigmoid>   --optimizer <adam|sgd|rmsprop>   --seq_len <25|50|100>   [--clip_grad]   [--epochs N]
```

### Example:
```bash
python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 50
```

Every run logs a row in `results/metrics.csv` with:

- Architecture  
- Activation  
- Optimizer  
- Sequence Length  
- Gradient Clipping  
- **Accuracy**  
- **Training Time**  
- Checkpoint Path  

---

# 🧪 **Evaluate a Saved Model**

```bash
python -m src.evaluate   --arch lstm   --activation relu   --seq_len 100   --checkpoint results/model_lstm_relu_adam_len100_clip0.pt
```

Outputs:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

# 📈 **Generate Plots**

```bash
python -m src.plot_results
```

This produces GitHub-friendly bar charts:

- Accuracy by Architecture  
- Accuracy by Activation  
- Accuracy by Optimizer  
- Accuracy by Sequence Length  
- Accuracy by Gradient Clipping  

All saved to `results/plots/`.

---

# 🔬 **Recommended Experiment Matrix**

### 🧠 Architecture Comparison
```
rnn, lstm, bilstm   (activation=relu, optimizer=adam, seq_len=50)
```

### ⚡ Activation Comparison
```
relu, tanh, sigmoid (arch=lstm, optimizer=adam, seq_len=50)
```

### 🔧 Optimizer Comparison
```
adam, rmsprop, sgd  (arch=lstm, activation=relu, seq_len=50)
```

### 📏 Sequence Length Comparison
```
25, 50, 100         (arch=lstm, activation=relu, optimizer=adam)
```

### 🛡 Gradient Clipping Comparison
```
clip_grad = False vs True
```

This satisfies ALL experimental requirements for the grade.

---

# 🔁 **Reproducibility**

This project is **fully reproducible** due to:

- Fixed seeds (`torch`, `numpy`, `random`)  
- Deterministic preprocessing  
- Logged metrics  
- Saved model checkpoints  
- Exact experiment commands included  
- Clear train/eval scripts  

---

# 🛠 **Troubleshooting**

| Issue | Fix |
|-------|------|
| `FileNotFoundError: imdb_len*.npz` | Run preprocessing first |
| `ModuleNotFoundError` | Reinstall dependencies |
| Plots missing | Re-run `plot_results.py` |
| CSV corrupted | Delete `results/metrics.csv` and re-run experiments |

---

# 🏁 **Conclusion**

This repository provides a full NLP experiment pipeline that is:

- Clean  
- Modular  
- Reproducible  
- Research-friendly  
- GitHub-ready  

Ideal for academic submission and future extension.

---

