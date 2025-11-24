import pandas as pd
import os

POSSIBLE_TEXT_COLS = ["text", "article", "content", "body"]
POSSIBLE_LABEL_COLS = ["label", "category", "target", "type"]

def load_dataset(path: str):
    """Load a CSV dataset and normalize column names to 'text' and 'label'."""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print("Warning: UTF-8 failed, trying 'latin1' encoding...")
        df = pd.read_csv(path, encoding="latin1")

    text_col = next((c for c in df.columns if c.lower() in POSSIBLE_TEXT_COLS), None)
    label_col = next((c for c in df.columns if c.lower() in POSSIBLE_LABEL_COLS), None)
    
    if text_col is None or label_col is None:
        raise ValueError(f"Dataset missing required columns. Found: {list(df.columns)}")

    df = df.rename(columns={text_col: "text", label_col: "label"})
    
    return df

def preview(df: pd.DataFrame, n=None):
    """Preview the dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("preview() expects a pandas DataFrame.")
    if n is None:
        return df
    return df.head(n)

