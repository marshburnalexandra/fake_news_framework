from pathlib import Path
import pandas as pd
import warnings

POSSIBLE_TEXT_COLS = ["text", "article", "content", "body", "title"]
POSSIBLE_LABEL_COLS = ["label", "category", "target", "type", "class"]

def load_dataset(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p}")
    try:
        df = pd.read_csv(p)
    except UnicodeDecodeError:
        warnings.warn("UTF-8 decode failed, trying latin1.")
        df = pd.read_csv(p, encoding="latin1")

    text_col = next((c for c in df.columns if c.lower() in POSSIBLE_TEXT_COLS), None)
    label_col = next((c for c in df.columns if c.lower() in POSSIBLE_LABEL_COLS), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Dataset missing required columns. Found columns: {list(df.columns)}")

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df

def preview(df, n=5):
    return df.head(n)
