import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def ensure_plots_dir(path: str = "results/plots"):
    os.makedirs(path, exist_ok=True)
    return path


def load_metrics(path: str = "results/metrics.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv not found at {path}")
    df = pd.read_csv(path)
    if "accuracy" not in df.columns:
        raise ValueError("metrics.csv must contain an 'accuracy' column.")
    return df


def plot_bar(df: pd.DataFrame, group_col: str, out_dir: str):
    """
    Group by `group_col`, compute mean accuracy, and save a bar plot.
    """
    grouped = df.groupby(group_col)["accuracy"].mean().reset_index()
    grouped = grouped.sort_values("accuracy", ascending=False)

    plt.figure()
    plt.bar(grouped[group_col].astype(str), grouped["accuracy"])
    plt.xlabel(group_col)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {group_col}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"accuracy_by_{group_col}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bar plots from results/metrics.csv"
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/metrics.csv",
        help="Path to metrics CSV file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/plots",
        help="Directory to store plots.",
    )
    args = parser.parse_args()

    plots_dir = ensure_plots_dir(args.out_dir)
    df = load_metrics(args.metrics_path)

    # Only keep the columns we know about
    expected_cols = ["arch", "activation", "optimizer", "seq_len", "clip_grad", "accuracy"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in metrics.csv, skipping its plot.")
            continue

        if col == "accuracy":
            continue  # not a grouping column

        plot_bar(df, col, plots_dir)


if __name__ == "__main__":
    main()
