"""Classic text baselines and an additive-attention LSTM encoder-decoder."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .preprocessing import preprocess_text, preprocess_tokens

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def _build_classic_models() -> dict[str, Pipeline]:
    return {
        "BoW · Naive Bayes": Pipeline(
            [
                ("vectorizer", CountVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2))),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
        "TF-IDF · Logistic": Pipeline(
            [
                ("vectorizer", TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2))),
                ("classifier", LogisticRegression(C=3.0, max_iter=500, class_weight="balanced")),
            ]
        ),
    }


@dataclass
class AttentionLSTM:
    """Compact CPU-friendly seq2seq classifier for the dashboard demo."""

    epochs: int = 14
    max_length: int = 32
    embedding_size: int = 48
    hidden_size: int = 48

    def fit(self, comments: list[str], labels: list[str]) -> "AttentionLSTM":
        import torch
        from torch import nn

        torch.manual_seed(17)
        np.random.seed(17)
        self.torch = torch
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2}
        counts = Counter(token for comment in comments for token in preprocess_tokens(comment))
        for word in counts:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.label_encoder = LabelEncoder().fit(labels)
        x = torch.tensor([self._encode(comment) for comment in comments], dtype=torch.long)
        y = torch.tensor(self.label_encoder.transform(labels), dtype=torch.float32)
        vocab_size = len(self.vocab)
        embedding_size, hidden_size = self.embedding_size, self.hidden_size
        pad_id, bos_id = self.vocab["<pad>"], self.vocab["<bos>"]

        class EncoderDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
                self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
                self.query = nn.Linear(hidden_size, hidden_size, bias=False)
                self.key = nn.Linear(hidden_size, hidden_size, bias=False)
                self.score = nn.Linear(hidden_size, 1, bias=False)
                self.decoder = nn.LSTM(embedding_size + hidden_size, hidden_size, batch_first=True)
                self.output = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 1))

            def forward(self, tokens):
                mask = tokens.ne(pad_id)
                encoded, (hidden, cell) = self.encoder(self.embedding(tokens))
                query = self.query(hidden[-1]).unsqueeze(1)
                energy = self.score(torch.tanh(self.key(encoded) + query)).squeeze(-1)
                energy = energy.masked_fill(~mask, -1e9)
                attention = torch.softmax(energy, dim=1)
                context = torch.bmm(attention.unsqueeze(1), encoded)
                start = torch.full((tokens.shape[0], 1), bos_id, dtype=torch.long)
                start = start.to(tokens.device)
                decoded, _ = self.decoder(
                    torch.cat((self.embedding(start), context), dim=2), (hidden, cell)
                )
                return self.output(decoded[:, 0, :]).squeeze(-1), attention

        self.network = EncoderDecoder()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.012, weight_decay=0.01)
        loss_fn = nn.BCEWithLogitsLoss()
        self.network.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(x))
            for start in range(0, len(x), 16):
                indices = order[start : start + 16]
                optimizer.zero_grad()
                logits, _ = self.network(x[indices])
                loss = loss_fn(logits, y[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
        self.network.eval()
        return self

    def _encode(self, comment: str) -> list[int]:
        token_ids = [self.vocab.get(token, 1) for token in preprocess_tokens(comment)]
        token_ids = token_ids[: self.max_length]
        return token_ids + [0] * (self.max_length - len(token_ids))

    def predict(self, comment: str) -> tuple[float, list[tuple[str, float]]]:
        torch = self.torch
        with torch.no_grad():
            encoded = torch.tensor([self._encode(comment)], dtype=torch.long)
            logits, weights = self.network(encoded)
            probability = float(torch.sigmoid(logits)[0].item())
            tokens = preprocess_tokens(comment)[: self.max_length]
            attention = weights[0, : len(tokens)].tolist()
        ranked = sorted(zip(tokens, attention), key=lambda item: item[1], reverse=True)[:6]
        return probability, ranked


@dataclass
class ModelBundle:
    classic: dict[str, Pipeline]
    neural: AttentionLSTM
    label_encoder: LabelEncoder
    row_count: int


def train_models(frame: pd.DataFrame) -> ModelBundle:
    comments = frame["comment"].astype(str).tolist()
    labels = frame["label"].astype(str).tolist()
    if len(set(labels)) != 2:
        raise ValueError("Training data must contain both safe and toxic labels.")
    classic = _build_classic_models()
    for model in classic.values():
        model.fit(comments, labels)
    neural = AttentionLSTM().fit(comments, labels)
    encoder = LabelEncoder().fit(labels)
    return ModelBundle(classic, neural, encoder, len(frame))


def predict_all(bundle: ModelBundle, comment: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, model in bundle.classic.items():
        probabilities = model.predict_proba([comment])[0]
        toxic_index = list(model.classes_).index("toxic")
        probability = float(probabilities[toxic_index])
        results[name] = {
            "probability": probability,
            "label": "toxic" if probability >= 0.5 else "safe",
        }
    probability, attention = bundle.neural.predict(comment)
    results["LSTM · Attention"] = {
        "probability": probability,
        "label": "toxic" if probability >= 0.5 else "safe",
        "attention": attention,
    }
    return results


def dataset_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "samples": len(frame),
        "toxic": int((frame["label"] == "toxic").sum()),
        "safe": int((frame["label"] == "safe").sum()),
        "balance": round(
            min((frame["label"] == "toxic").sum(), (frame["label"] == "safe").sum())
            / max((frame["label"] == "toxic").sum(), (frame["label"] == "safe").sum()),
            2,
        ),
    }
