# src/evaluate.py

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import TextRNNClassifier


# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_tokenizer(path: str):
    with open(path, "r") as f:
        tok = json.load(f)
    vocab_size = len(tok["index_word"])
    pad_idx = tok["PAD_IDX"]
    return vocab_size, pad_idx


def load_data(seq_len: int, batch_size: int = 32):
    npz_path = f"data/processed/imdb_len{seq_len}.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Processed data file not found: {npz_path}")

    data = np.load(npz_path)

    x_test = torch.tensor(data["x_test"], dtype=torch.long)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return test_loader


def compute_metrics(y_true, y_pred_bin):
    """
    y_true, y_pred_bin: 1D numpy arrays of 0/1 labels.
    Returns accuracy, precision, recall, f1, and confusion matrix entries.
    """
    assert y_true.shape == y_pred_bin.shape

    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return accuracy, precision, recall, f1, (tp, fp, tn, fn)


def evaluate_model(
    arch: str,
    activation: str,
    seq_len: int,
    checkpoint: str,
    batch_size: int = 32,
):
    # 1) Load tokenizer (for vocab size and PAD index)
    vocab_size, pad_idx = load_tokenizer("data/processed/tokenizer_top10k.json")

    # 2) Build model with same hyperparameters as in train.py
    model = TextRNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=100,
        architecture=arch,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        activation=activation,
        pad_idx=pad_idx,
    )

    # 3) Load checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    # 4) Load test data
    test_loader = load_data(seq_len=seq_len, batch_size=batch_size)

    # 5) Run inference
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            probs = model(xb)  # shape (batch,)
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.tolist())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    preds_bin = (probs >= 0.5).astype(np.int32)

    # 6) Compute metrics
    acc, prec, rec, f1, (tp, fp, tn, fn) = compute_metrics(labels, preds_bin)

    print("=== Evaluation Results ===")
    print(f"Architecture : {arch}")
    print(f"Activation   : {activation}")
    print(f"Seq length   : {seq_len}")
    print(f"Checkpoint   : {checkpoint}")
    print("------------------------------")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print("------------------------------")
    print(f"Confusion Matrix (TP, FP, TN, FN): {tp}, {fp}, {tn}, {fn}")

    return acc, prec, rec, f1, (tp, fp, tn, fn)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved sentiment model on the IMDb test set."
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["rnn", "lstm", "bilstm"],
        help="Architecture type (must match training).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        required=True,
        choices=["relu", "tanh", "sigmoid"],
        help="Head activation (must match training).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        choices=[25, 50, 100],
        help="Sequence length corresponding to the processed dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )

    args = parser.parse_args()

    evaluate_model(
        arch=args.arch,
        activation=args.activation,
        seq_len=args.seq_len,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
