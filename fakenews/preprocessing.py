"""
Text Preprocessing Utility using NLTK

This module provides a reusable text preprocessing pipeline for Natural Language
Processing (NLP) tasks. It includes basic cleaning operations such as lowercasing,
URL removal, punctuation removal, stopword filtering, and stemming.

Features:
- Automatically downloads NLTK stopwords if available
- Removes URLs and non-alphabetical characters
- Applies optional stopword removal
- Applies optional stemming using PorterStemmer
"""

from typing import List
import re
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    warnings.warn("NLTK stopwords unavailable; stopword removal disabled.")
    _STOP_WORDS = set()

_STEMMER = PorterStemmer()


class Preprocessor:
    """
    A configurable text preprocessing pipeline.

    This class provides a reusable pipeline for cleaning and normalizing raw text
    before applying machine learning or NLP models.

    Processing steps include:
    - Converting text to lowercase
    - Removing URLs
    - Removing punctuation and numbers
    - Removing stopwords (optional)
    - Applying Porter stemming (optional)

    Instances of this class can be used like a function.
    """

    def __init__(self, remove_stopwords: bool = True, stem: bool = True):
        """
        Initialize the text preprocessor.

        Parameters
        ----------
        remove_stopwords : bool, optional
            Whether to remove English stopwords (default is True).
        stem : bool, optional
            Whether to apply word stemming using PorterStemmer (default is True).
        """
        self._remove_stopwords = remove_stopwords
        self._stem = stem

    def clean(self, text: str) -> str:
        """
        Clean and normalize a string.

        This method applies the full preprocessing pipeline:
        - Converts to lowercase
        - Removes URLs
        - Keeps alphabetic characters only
        - Splits text into tokens
        - Removes stopwords if enabled
        - Applies stemming if enabled

        Parameters
        ----------
        text : str
            Input string to clean.

        Returns
        -------
        str
            The cleaned and preprocessed text.

        Raises
        ------
        TypeError
            If input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = [t for t in text.split() if t]

        if self._remove_stopwords and _STOP_WORDS:
            tokens = [t for t in tokens if t not in _STOP_WORDS]

        if self._stem:
            tokens = [_STEMMER.stem(t) for t in tokens]

        return " ".join(tokens)

    def __call__(self, text: str) -> str:
        """
        Call the instance as a function.

        This is equivalent to calling clean(text).

        Parameters
        ----------
        text : str
            Input string to preprocess.

        Returns
        -------
        str
            Cleaned text.
        """
        return self.clean(text)

    def __repr__(self):
        """
        Return a string representation of the Preprocessor instance.

        Returns
        -------
        str
            A readable description of the current configuration.
        """
        return f"Preprocessor(remove_stopwords={self._remove_stopwords}, stem={self._stem})"
