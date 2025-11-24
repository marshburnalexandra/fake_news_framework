from typing import List
import re
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
except LookupError:
    warnings.warn("NLTK stopwords not found. Stopword removal disabled.")
    stop_words = set()

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Basic cleaning with validation.
    Steps:
    - Ensures text is a string
    - Lowercase
    - Remove URLs
    - Remove punctuation
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text() expected str, got {type(text)} instead.")
    
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[a^zA-Z\s]", "", text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Split cleaned text into tokens."""
    if not isinstance(text, str):
        raise TypeError("tokenize() only accepts strings.")
    return text.split()

def remove_stopwords(tokens: List[str]) -> List[str]:
    if not isinstance(tokens, list):
        raise TypeError("remove_stopwords() expects a list of tokens.")
    if not stop_words:
        warnings.warn("Stopwords unavailable - tokens not filtered.")
        return tokens
    return [t for t in tokens if t not in stop_words]

def stem_tokens(tokens: List[str]) -> List[str]:
    if not isinstance(tokens, list):
        raise TypeError("stem_token() expects a list of tokens.")
    return [stemmer.stem(t) for t in tokens]

def preprocess_pipeline(text: str) -> str:
    """
    Full pipeline:
    clean -> tokenize -> remove stopwords -> stem -> join words
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)