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
    """Encapsulates text cleaning and preprocessing pipeline."""

    def __init__(self, remove_stopwords: bool = True, stem: bool = True):
        self._remove_stopwords = remove_stopwords
        self._stem = stem

    def clean(self, text: str) -> str:
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
        """Allow object to be called like a function."""
        return self.clean(text)

    def __repr__(self):
        return f"Preprocessor(remove_stopwords={self._remove_stopwords}, stem={self._stem})"
