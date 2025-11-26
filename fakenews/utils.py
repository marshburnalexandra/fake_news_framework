"""
Dataset Loader Utility

This module provides helper functions to load and preview text classification datasets.
It automatically detects the text and label columns from common column names,
standardizes them, and returns a clean Pandas DataFrame.

Expected behavior:
- Reads a CSV dataset from a given file path.
- Automatically finds the text and label columns.
- Renames detected columns to 'text' and 'label'.
- Cleans the label column by converting it to lowercase and stripping whitespace.
- Handles encoding issues by falling back to latin1 if UTF-8 fails.
"""

from pathlib import Path
import pandas as pd
import warnings

POSSIBLE_TEXT_COLS = ["text", "article", "content", "body", "title"]
POSSIBLE_LABEL_COLS = ["label", "category", "target", "type", "class"]


def load_dataset(path):
    """
    Load and preprocess a CSV dataset for text classification tasks.

    This function reads a dataset from a CSV file, attempts to identify the
    text and label columns based on common column names, renames them to
    'text' and 'label', and cleans the label values.

    If UTF-8 decoding fails, it retries using latin1 encoding.

    Parameters
    ----------
    path : str or Path
        Path to the dataset CSV file.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame containing:
        - 'text': the main text content
        - 'label': the corresponding category or class

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no text or label column is found in the dataset.
    """

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
    """
    Display the first few rows of a DataFrame.

    This is useful for quickly inspecting the dataset contents after loading.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to preview.
    n : int, optional
        Number of rows to display (default is 5).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the first n rows.
    """

    return df.head(n)
