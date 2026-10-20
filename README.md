# CyberShield AI

A polished, local-first cyberbullying and toxic-comment detection dashboard.
It combines regex-based text cleaning, NLTK stemming and lemmatization,
spaCy tokenization, TextBlob sentiment, Bag-of-Words and TF-IDF baselines,
and a PyTorch LSTM encoder-decoder with additive attention.

> **Responsible-use note:** This is an educational demonstration, not a
> production moderation or safety system. The included examples are synthetic,
> the model is not a substitute for human review, and predictions may be wrong.
> Do not use it to make consequential decisions about people.

## Run locally

Use Python 3.11 or newer, then from this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The first launch trains the demo
models locally; no API keys or remote services are used.

## Project layout

```text
.
├── app.py                      # Streamlit dashboard and user interactions
├── data/
│   └── demo_comments.csv       # Synthetic starter training examples
├── src/
│   └── cyberguard/
│       ├── data.py             # Dataset loading and label normalization
│       ├── modeling.py         # BoW, TF-IDF and LSTM-attention models
│       └── preprocessing.py    # Regex, spaCy, NLTK and TextBlob utilities
├── tests/
│   └── test_preprocessing.py   # Focused preprocessing and data tests
└── requirements.txt
```

## Use your own data

Upload a CSV in the sidebar containing `comment` and `label` columns. Labels
can be `toxic`/`safe` or `1`/`0`; the app accepts at least four examples per
class and retrains its models locally. Keep sensitive data local and use
properly consented, representative datasets.

## Model notes

The dashboard compares CountVectorizer + Multinomial Naive Bayes, TF-IDF +
Logistic Regression, and a PyTorch sequence model. The neural model encodes
tokens with an LSTM, computes additive attention over encoder outputs, then
decodes the attended context into a binary toxicity prediction. The small
demo corpus is for an interface walkthrough only; evaluate on a separate,
representative test set before making claims about model quality.

spaCy's English tokenizer and NLTK's Porter stemmer work without downloads.
When NLTK WordNet data and a spaCy English model are installed, their
lemmatizers are used; otherwise the app reports and uses a local fallback.
