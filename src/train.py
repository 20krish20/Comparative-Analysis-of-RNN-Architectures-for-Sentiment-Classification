import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .models import TextRNNClassifier
from tqdm import tqdm
import os
import csv
import time



import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



def load_tokenizer(path):
    with open(path, "r") as f:
        tok = json.load(f)
    vocab_size = len(tok["index_word"])
    pad_idx = tok["PAD_IDX"]
    return vocab_size, pad_idx


def get_optimizer(name, params, lr=1e-3):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, loader, criterion, optimizer, clip_grad):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            preds_class = (preds >= 0.5).float()

            correct += (preds_class == yb).sum().item()
            total += len(yb)

            preds_all.extend(preds.tolist())
            labels_all.extend(yb.tolist())

    accuracy = correct / total
    return accuracy


def save_metrics(row, path="results/metrics.csv"):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, default="lstm",
                        choices=["rnn", "lstm", "bilstm"])
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--seq_len", type=int, default=50,
                        choices=[25, 50, 100])
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()

    # 1. ---- Load Dataset ----
    npz_path = f"data/processed/imdb_len{args.seq_len}.npz"
    data = np.load(npz_path)

    x_train = torch.tensor(data["x_train"], dtype=torch.long)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    x_test = torch.tensor(data["x_test"], dtype=torch.long)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test),
                             batch_size=32, shuffle=False)

    # 2. ---- Load Tokenizer ----
    vocab_size, pad_idx = load_tokenizer("data/processed/tokenizer_top10k.json")

    # 3. ---- Build Model ----
    model = TextRNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=100,
        architecture=args.arch,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        activation=args.activation,
        pad_idx=pad_idx
    )

    # 4. ---- Optimizer + Loss ----
    criterion = nn.BCELoss()
    optimizer = get_optimizer(args.optimizer, model.parameters())

    # 5. ---- Training Loop (timed) ----
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            clip_grad=args.clip_grad,
        )
    end_time = time.time()
    train_time = end_time - start_time

    # 6. ---- Evaluation ----
    acc = evaluate(model, test_loader)

    # NEW: save checkpoint so evaluate.py can use it
    ckpt_name = f"model_{args.arch}_{args.activation}_{args.optimizer}_len{args.seq_len}_clip{int(args.clip_grad)}.pt"
    ckpt_path = os.path.join("results", ckpt_name)
    torch.save(model.state_dict(), ckpt_path)

    # 7. ---- Save Metrics ----
    row = {
        "arch": args.arch,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "seq_len": args.seq_len,
        "clip_grad": args.clip_grad,
        "accuracy": acc,
        "train_time": train_time,
        "checkpoint": ckpt_path,
    }

    save_metrics(row)
    print("Experiment complete:", row)


if __name__ == "__main__":
    main()
