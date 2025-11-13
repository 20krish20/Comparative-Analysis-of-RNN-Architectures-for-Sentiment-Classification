import argparse
import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


from .utils import clean_text, LiteTokenizer, pad_sequences


# ----------------------------
# Helpers
# ----------------------------

def stratified_half_split(
    df: pd.DataFrame,
    label_col: str = "label",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a deterministic 50/50 train-test split, preserving class balance.

    For each label:
      - Shuffle indices with a fixed random seed.
      - Take the first half as train, second half as test.
    Then concatenate and shuffle within train/test.

    This approximates the "predefined 25k/25k" split for the original IMDb dataset,
    but uses the Kaggle CSV (which has 50k rows without an explicit split).
    """
    rng = np.random.RandomState(seed)
    train_idx_all: List[int] = []
    test_idx_all: List[int] = []

    # Group by class label; perform per-class splitting
    for cls, group in df.groupby(label_col, group_keys=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        cut = len(idx) // 2
        train_idx_all.append(idx[:cut])
        test_idx_all.append(idx[cut:])

    train_idx = np.concatenate(train_idx_all)
    test_idx = np.concatenate(test_idx_all)

    # Shuffle train and test indices to mix classes
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, test_df


def compute_len_stats(name: str, arr: np.ndarray) -> Dict[str, Any]:
    """
    Compute simple sequence-length statistics for a padded array.

    `arr` is 2D: (num_examples, maxlen).
    We compute lengths as "number of non-PAD tokens" (assumes PAD=0).
    """
    lengths = (arr != 0).sum(axis=1)

    return {
        "name": name,
        "mean_nonpad_tokens": float(np.mean(lengths)),
        "median_nonpad_tokens": float(np.median(lengths)),
        "pct_empty": float(np.mean(lengths == 0) * 100.0),
        "num_examples": int(arr.shape[0]),
    }


# ----------------------------
# Main preprocessing pipeline
# ----------------------------

def run_preprocessing(
    raw_csv_path: str,
    output_dir: str,
    top_k: int = 10_000,
    seq_lengths: List[int] = None,
    seed: int = 42,
) -> None:
    if seq_lengths is None:
        seq_lengths = [25, 50, 100]

    # 1. Load raw dataset
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    if not {"review", "sentiment"}.issubset(df_raw.columns):
        raise ValueError(
            "Expected columns 'review' and 'sentiment' in the CSV. "
            f"Columns found: {list(df_raw.columns)}"
        )

    # 2. Clean text
    df = df_raw[["review", "sentiment"]].copy()
    df["clean"] = df["review"].map(clean_text)

    # 3. Map sentiment to integer labels
    label_map = {"negative": 0, "positive": 1}
    df["label"] = df["sentiment"].map(lambda s: label_map.get(str(s).lower(), np.nan))
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # 4. Stratified 50/50 train-test split
    train_df, test_df = stratified_half_split(df, label_col="label", seed=seed)

    # 5. Fit tokenizer on TRAIN ONLY
    tokenizer = LiteTokenizer(num_words=top_k, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean"].tolist())

    # 6. Convert texts to sequences of IDs
    x_train_seq = tokenizer.texts_to_sequences(train_df["clean"].tolist())
    x_test_seq = tokenizer.texts_to_sequences(test_df["clean"].tolist())
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # 7. Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # 8. Save padded versions for each sequence length
    bundles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    npz_paths: Dict[int, str] = {}

    for L in seq_lengths:
        x_train_pad = pad_sequences(
            x_train_seq,
            maxlen=L,
            padding="post",
            truncating="post",
            value=tokenizer.PAD_IDX,
        )
        x_test_pad = pad_sequences(
            x_test_seq,
            maxlen=L,
            padding="post",
            truncating="post",
            value=tokenizer.PAD_IDX,
        )

        out_path = os.path.join(output_dir, f"imdb_len{L}.npz")
        np.savez_compressed(
            out_path,
            x_train=x_train_pad,
            y_train=y_train,
            x_test=x_test_pad,
            y_test=y_test,
        )

        bundles[L] = (x_train_pad, x_test_pad)
        npz_paths[L] = out_path

    # 9. Save tokenizer and vocabulary
    tok_json_path = os.path.join(output_dir, "tokenizer_top10k.json")
    with open(tok_json_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    vocab_txt_path = os.path.join(output_dir, "vocab_top10k.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        # We want to write in index order (excluding PAD/OOV, which are 0/1).
        for idx in sorted(tokenizer.index_word.keys()):
            if idx <= 1:  # skip PAD and OOV in this file
                continue
            token = tokenizer.index_word[idx]
            f.write(f"{idx}\t{token}\n")

    # 10. Build summary statistics for report
    summary_rows: List[Dict[str, Any]] = []

    # Label balance (useful for Dataset Summary in report.pdf)
    label_counts_train = train_df["label"].value_counts().to_dict()
    label_counts_test = test_df["label"].value_counts().to_dict()
    summary_rows.append(
        {"name": "label_balance_train", **{f"class_{k}": int(v) for k, v in label_counts_train.items()}}
    )
    summary_rows.append(
        {"name": "label_balance_test", **{f"class_{k}": int(v) for k, v in label_counts_test.items()}}
    )

    # Sequence length stats for each padded variant
    for L in seq_lengths:
        x_train_pad, x_test_pad = bundles[L]
        summary_rows.append(compute_len_stats(f"train_len{L}", x_train_pad))
        summary_rows.append(compute_len_stats(f"test_len{L}", x_test_pad))

    # Vocab size actually used
    vocab_including_special = len(tokenizer.index_word)
    summary_rows.append(
        {
            "name": "vocab",
            "TOP_K_requested": top_k,
            "vocab_size_including_PAD_OOV": vocab_including_special,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "imdb_prep_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # 11. Print a concise log for the console
    print("=== Preprocessing complete ===")
    print(f"Raw dataset:     {raw_csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Train size:      {len(train_df)}")
    print(f"Test size:       {len(test_df)}")
    print(f"Tokenizer JSON:  {tok_json_path}")
    print(f"Vocab file:      {vocab_txt_path}")
    print(f"Summary CSV:     {summary_csv_path}")
    for L, p in npz_paths.items():
        print(f"NPZ (maxlen={L}): {p}")


# ----------------------------
# CLI entry point
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess IMDb (Kaggle) dataset for RNN-based sentiment classification."
    )

    parser.add_argument(
        "--raw_csv",
        type=str,
        default=os.path.join("data", "raw", "IMDB Dataset.csv"),
        help="Path to the raw IMDb CSV file from Kaggle.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Directory where processed artifacts will be stored.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10_000,
        help="Top-K most frequent words to keep in the vocabulary.",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="List of sequence lengths for padding/truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        raw_csv_path=args.raw_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        seq_lengths=args.seq_lengths,
        seed=args.seed,
    )
