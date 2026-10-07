"""Text normalisation shared by training and inference (prevents train/serve skew)."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_NON_ESSENTIAL = re.compile(r"[^a-z\s$!]")  # keep '$' and '!' — strong spam signals
_NLTK_RESOURCES = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab",
                   "stopwords": "corpora/stopwords"}


def ensure_nltk_resources() -> None:
    """Download NLTK data only if missing."""
    for pkg, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    ensure_nltk_resources()
    return frozenset(stopwords.words("english"))


_stemmer = PorterStemmer()


def preprocess_message(message: str) -> str:
    """lowercase -> strip punctuation/digits -> tokenize -> drop stopwords -> stem -> join."""
    text = _NON_ESSENTIAL.sub("", str(message).lower())
    stops = _stop_words()
    tokens = (t for t in word_tokenize(text) if t not in stops)
    return " ".join(_stemmer.stem(t) for t in tokens)


def preprocess_corpus(messages: Iterable[str]) -> list[str]:
    return [preprocess_message(m) for m in messages]
