"""Local text normalization and NLP feature extraction."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import nltk
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_stemmer = PorterStemmer()
_wordnet = WordNetLemmatizer()


@lru_cache(maxsize=1)
def _spacy_pipeline() -> Any:
    """Load a local English model when available, else use spaCy's tokenizer."""
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except (OSError, ImportError):
        return spacy.blank("en")


def clean_text(text: str) -> str:
    """Remove markup and replace URLs/user handles before tokenization."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = _MENTION_RE.sub(" USER ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _lemma(token: str) -> str:
    try:
        return _wordnet.lemmatize(token)
    except LookupError:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("sses"):
            return token[:-2]
        if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        return token


def preprocess_tokens(text: str, *, use_stemming: bool = False) -> list[str]:
    """Tokenize with spaCy; normalize with WordNet lemma and optional stemming."""
    normalized = clean_text(text)
    doc = _spacy_pipeline()(normalized)
    tokens = [token.text.lower() for token in doc if _TOKEN_RE.fullmatch(token.text)]
    tokens = [_lemma(token) for token in tokens]
    if use_stemming:
        tokens = [_stemmer.stem(token) for token in tokens]
    return tokens


def preprocess_text(text: str) -> str:
    """Return normalized tokens for vectorizers."""
    return " ".join(preprocess_tokens(text))


def nlp_features(text: str) -> dict[str, Any]:
    """Return visible preprocessing variants and lightweight sentiment features."""
    tokens = preprocess_tokens(text)
    stemmed = preprocess_tokens(text, use_stemming=True)
    sentiment = TextBlob(clean_text(text)).sentiment
    nlp = _spacy_pipeline()
    tokenizer_tokens = [token.text for token in nlp(clean_text(text)) if not token.is_space]
    return {
        "cleaned": clean_text(text),
        "tokens": tokens,
        "stems": stemmed,
        "spacy_tokens": tokenizer_tokens,
        "polarity": round(float(sentiment.polarity), 3),
        "subjectivity": round(float(sentiment.subjectivity), 3),
        "lemma_source": "WordNet" if _wordnet_available() else "token fallback",
        "spacy_model": "en_core_web_sm" if _has_spacy_model() else "spaCy blank English",
    }


@lru_cache(maxsize=1)
def _wordnet_available() -> bool:
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        return False


@lru_cache(maxsize=1)
def _has_spacy_model() -> bool:
    try:
        spacy.load("en_core_web_sm", disable=["ner", "parser"])
        return True
    except (OSError, ImportError):
        return False
