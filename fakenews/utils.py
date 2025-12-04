import pandas as pd
import json
import os

def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".txt":
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    else:
        raise ValueError("Unsupported file format. Use CSV, JSON, or TXT.")

    possible_text_cols = ["text", "article", "content", "body"]
    text_col = next((c for c in df.columns if c.lower() in possible_text_cols), None)

    if text_col is None:
        raise ValueError("Dataset must contain a text column (text/content/article).")

    possible_label_cols = ["label", "category", "target", "class"]
    label_col = next((c for c in df.columns if c.lower() in possible_label_cols), None)

    if label_col is None:
        raise ValueError("Dataset must contain a label column (label/class/category).")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    return df


def preview(df, rows=5):
    return df.head(rows)
