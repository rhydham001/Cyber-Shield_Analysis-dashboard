import pytest

from src.cyberguard.data import demo_dataset, load_csv_dataset
from src.cyberguard.preprocessing import clean_text, nlp_features, preprocess_tokens


def test_clean_text_removes_markup_and_masks_links_and_mentions():
    cleaned = clean_text("<b>Hello</b> @Taylor https://example.com")
    assert "<b>" not in cleaned
    assert "https://" not in cleaned
    assert "hello" in cleaned
    assert "user" in cleaned


def test_preprocessor_returns_tokens_and_stems():
    assert preprocess_tokens("Helpful friends are sharing ideas!") == [
        "helpful",
        "friend",
        "are",
        "sharing",
        "idea",
    ]
    assert "friend" in preprocess_tokens("friends", use_stemming=True)


def test_sentiment_features_are_numeric():
    result = nlp_features("I love this kind and helpful community!")
    assert isinstance(result["polarity"], float)
    assert isinstance(result["subjectivity"], float)
    assert result["tokens"]


def test_demo_dataset_is_balanced():
    frame = demo_dataset()
    assert set(frame["label"]) == {"safe", "toxic"}
    assert frame["label"].value_counts().nunique() == 1


def test_csv_loader_normalizes_binary_labels():
    csv_bytes = (
        b"comment,label\n"
        b"kind one,safe\nkind two,safe\nkind three,safe\nkind four,safe\n"
        b"bad one,1\nbad two,toxic\nbad three,yes\nbad four,bullying\n"
    )
    frame = load_csv_dataset(csv_bytes)
    assert frame["label"].value_counts().to_dict() == {"safe": 4, "toxic": 4}


def test_csv_loader_requires_both_classes():
    with pytest.raises(ValueError, match="at least 4"):
        load_csv_dataset(b"comment,label\na,toxic\nb,toxic\nc,toxic\nd,toxic\n")
