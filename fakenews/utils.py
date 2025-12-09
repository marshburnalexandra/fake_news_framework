import pandas as pd
import json
import os

def load_dataset(path):
    """
    Load a text-classification dataset from a CSV, JSON, or TXT file.

    The function attempts to automatically identify the text and label columns
    based on common naming conventions. It supports:
      - CSV files (.csv)
      - JSON files (.json)
      - Tab-separated text files (.txt) with two columns: text and label

    Parameters
    ----------
    path : str
        File path to the dataset.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing two columns:
        - "text": the input text
        - "label": the associated label

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file format is unsupported or if required columns
        (text and label) cannot be found.

    Notes
    -----
    Recognized text column names:
        ["text", "article", "content", "body"]
    Recognized label column names:
        ["label", "category", "target", "class"]
    """
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
    """
    Return the first few rows of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to preview.
    rows : int, optional (default=5)
        Number of rows to return.

    Returns
    -------
    pandas.DataFrame
        The top `rows` entries of the DataFrame.
    """
    return df.head(rows)
