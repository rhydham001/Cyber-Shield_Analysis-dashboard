"""Demo corpus and CSV dataset validation."""

from __future__ import annotations

from io import StringIO

import pandas as pd

SAFE_EXAMPLES = [
    "Thanks for sharing your idea, I learned something new.",
    "I disagree with this point, but I respect your opinion.",
    "Could you explain how you reached that conclusion?",
    "Great work on the project, the design looks thoughtful.",
    "Let's keep the conversation kind and constructive.",
    "Your feedback helped me improve this draft.",
    "I see it differently, and here is another perspective.",
    "That was a helpful question, thank you for asking.",
    "Welcome to the group, happy to have you here.",
    "We can discuss the topic without attacking each other.",
    "This is an interesting article with useful examples.",
    "I appreciate the time you spent explaining this.",
    "Your creativity really shines in this illustration.",
    "Let's focus on the idea and find a solution together.",
    "Thank you for correcting that detail politely.",
    "I enjoyed reading your post and learning your view.",
    "Everyone deserves to feel safe in this community.",
    "That suggestion makes sense, I will give it a try.",
    "You made a thoughtful contribution to the discussion.",
    "I understand your concern and appreciate you explaining it.",
    "Good luck with your presentation, you have prepared well.",
    "Could we compare the evidence before drawing a conclusion?",
    "I value your perspective even though we disagree.",
    "This community is better when we help one another.",
    "Thanks for the kind words, that means a lot.",
    "Let's take a break and return to this conversation calmly.",
    "Your question is welcome here.",
    "I like how you explained the different options.",
    "We all make mistakes, let's learn and move forward.",
    "That is a fair point to consider.",
    "I hope your day gets better, take care.",
    "The team did a nice job collaborating on this.",
]

TOXIC_EXAMPLES = [
    "You are worthless and nobody wants you here.",
    "Everyone should laugh at this pathetic loser.",
    "Shut up, your stupid opinion is embarrassing.",
    "You are an idiot and should never post again.",
    "I hate you, get out of this community.",
    "Nobody likes you, stop showing your face here.",
    "Your work is garbage and so are you.",
    "You are a disgusting failure.",
    "Keep quiet, nobody cares what you think.",
    "You make every conversation worse, go away.",
    "What a pathetic and useless person.",
    "You are so dumb it hurts to read this.",
    "I will make sure everyone mocks you.",
    "Get lost, you do not belong here.",
    "This loser should be ashamed of themselves.",
    "You are a hateful bully and a complete fool.",
    "Stop being such a ridiculous clown.",
    "Your face and your ideas are both awful.",
    "Nobody wants your annoying comments.",
    "You should feel bad for being so stupid.",
    "This is the most useless thing I have ever seen.",
    "You are a joke and everyone knows it.",
    "Go away, you pathetic waste of space.",
    "Your comments are cruel and completely disgusting.",
    "I hope you fail because you deserve it.",
    "You are an awful person and a terrible friend.",
    "Why are you so annoying and clueless?",
    "Your stupid posts ruin everything.",
    "Stop talking, you have nothing useful to say.",
    "You are a mean and miserable bully.",
    "That was a nasty thing to say, you fool.",
    "You are unwanted here, leave now.",
]


def demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comment": SAFE_EXAMPLES + TOXIC_EXAMPLES,
            "label": ["safe"] * len(SAFE_EXAMPLES) + ["toxic"] * len(TOXIC_EXAMPLES),
        }
    )


def load_csv_dataset(csv_bytes: bytes) -> pd.DataFrame:
    """Validate uploaded CSV columns and normalize common binary labels."""
    try:
        frame = pd.read_csv(StringIO(csv_bytes.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if not {"comment", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain columns named 'comment' and 'label'.")
    frame = frame[["comment", "label"]].dropna().copy()
    frame["comment"] = frame["comment"].astype(str).str.strip()
    labels = frame["label"].astype(str).str.strip().str.lower()
    mapping = {
        "1": "toxic",
        "true": "toxic",
        "yes": "toxic",
        "toxic": "toxic",
        "bullying": "toxic",
        "0": "safe",
        "false": "safe",
        "no": "safe",
        "safe": "safe",
        "non-toxic": "safe",
        "nontoxic": "safe",
    }
    frame["label"] = labels.map(mapping)
    frame = frame.dropna()
    frame = frame[frame["comment"].str.len() > 0].reset_index(drop=True)
    counts = frame["label"].value_counts()
    if not {"safe", "toxic"}.issubset(counts.index) or counts.min() < 4:
        raise ValueError("Provide at least 4 valid examples for each label (safe and toxic).")
    return frame
