"""CyberShield AI: interactive toxic-comment analysis dashboard."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from src.cyberguard.data import demo_dataset, load_csv_dataset
from src.cyberguard.modeling import dataset_metrics, predict_all, train_models
from src.cyberguard.preprocessing import nlp_features

st.set_page_config(
    page_title="CyberShield AI | Comment Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
      color-scheme:light;
      --ink:#292719; --muted:#625b48; --panel:#fffdf7; --line:#eadfc5;
      --saffron:#e18a16; --orange:#f4a629; --gold:#ffd568; --green:#28704d; --red:#a83d30;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      background:
        radial-gradient(ellipse at 92% 1%,#ffeab766 0%,transparent 34%),
        radial-gradient(ellipse at 4% 18%,#fff0cc88 0%,transparent 30%),
        #fffaf0;
      color:var(--ink);
    }
    .stApp, .stApp p, .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"] {
      color:var(--ink);
    }
    .block-container { max-width:1480px; padding-top:2rem; padding-bottom:2rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:#302914 !important; }
    h2 { font-size:1.55rem !important; }
    h3 { font-size:1.16rem !important; }
    [data-testid="stSidebar"] {
      background:linear-gradient(180deg,#fff4d9 0%,#fffaf0 100%);
      border-right:1px solid #ead9b3;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:#393321 !important; }
    [data-testid="stMetric"] {
      min-height:112px; display:flex; flex-direction:column; justify-content:center;
      background:linear-gradient(145deg,#fffefa,#fff6e3);
      border:1px solid #eddfc0; padding:17px 20px; border-radius:16px;
      box-shadow:0 7px 22px #8e651012; transition:transform .22s ease,box-shadow .22s ease;
    }
    [data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 12px 28px #8e651022; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#62583e !important; }
    [data-testid="stMetricValue"] { color:#332b17 !important; }
    [data-testid="stMetricDelta"] { color:#416847 !important; }
    .hero {
      position:relative; isolation:isolate; overflow:hidden; padding:34px 38px; border-radius:24px;
      border:1px solid #ebc778; background:linear-gradient(112deg,#fff0bd 0%,#ffdf91 48%,#ffc66c 100%);
      box-shadow:0 18px 42px #9a69121c; margin-bottom:24px; animation:enter .65s ease-out both;
    }
    .hero:before {
      content:''; position:absolute; z-index:-1; width:320px;height:320px;right:7%;top:-165px;border-radius:50%;
      background:radial-gradient(circle,#fffdf4c9 0%,#fffdf466 38%,transparent 70%);
      animation:orbit-glow 9s ease-in-out infinite alternate;
    }
    .hero:after {
      content:'';position:absolute;z-index:-1;width:150px;height:150px;right:25%;bottom:-112px;border-radius:50%;
      background:#f0a02930;filter:blur(2px);animation:float 7s ease-in-out infinite;
    }
    .eyebrow { color:#79500e; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(32px,4vw,50px); line-height:1.08; margin:10px 0 10px; color:#34270d !important; }
    .hero p { color:#57472b !important; font-size:15px; max-width:690px; margin:0; line-height:1.65; }
    .pill {
      display:inline-block; border:1px solid #dfbd70; background:#fff8e8; border-radius:999px;
      color:#624916 !important; padding:6px 11px; font-size:11px; font-weight:600; margin:12px 5px 0 0;
    }
    .panel {
      height:100%; min-height:195px; border:1px solid #ebdfc5; border-radius:16px; padding:20px 22px;
      background:linear-gradient(145deg,#fffefa,#fff8e9);
      box-shadow:0 6px 18px #7853120c; transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
      animation:enter .5s ease-out both;
    }
    .panel:hover { transform:translateY(-4px); border-color:#e7bd65; box-shadow:0 13px 25px #78531218; }
    .panel h3 { margin-top:12px; }
    .muted { color:#625b48 !important; font-size:13px; line-height:1.65; }
    .risk {
      border-radius:14px; padding:19px 21px; border:1px solid #dfd3b8; background:#fffdf7;
      animation:enter .45s ease-out both;
    }
    .risk-toxic { border-color:#e9b7a8; background:linear-gradient(120deg,#fff0e9,#fffaf0); }
    .risk-safe { border-color:#b9d7bf; background:linear-gradient(120deg,#edf8ec,#fffaf0); }
    .risk-label { font:700 25px 'Space Grotesk',sans-serif; color:#302914; margin:5px 0; }
    .stButton>button, .stFormSubmitButton>button {
      min-height:42px; border:1px solid #d88b18; border-radius:10px; font-weight:700;
      color:#38250a; background:linear-gradient(105deg,#ffd66f,#f5ad38);
      box-shadow:0 4px 12px #bd77121c; transition:transform .2s ease,box-shadow .2s ease,filter .2s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
      transform:translateY(-2px); filter:saturate(1.08); color:#30220b;
      box-shadow:0 8px 18px #bd771233;
    }
    .stTextArea textarea {
      border-radius:12px; border:1px solid #ddcfaf; background:#fffefa; color:#302914;
      line-height:1.6; transition:border-color .2s ease,box-shadow .2s ease;
    }
    .stTextArea textarea:focus { border-color:#dc941f; box-shadow:0 0 0 3px #e8aa3730; }
    div[data-baseweb="select"]>div { background:#fffefa; border-color:#ddcfaf; border-radius:10px; color:#302914; }
    [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stTable"] {
      border:1px solid #eadfc5; border-radius:12px; background:#fffdf8;
    }
    [data-testid="stDataFrame"] * { color:#302914; }
    [data-testid="stTabs"] [role="tab"] { color:#625b48; font-weight:600; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#9a5c08; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#df941c; }
    [data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#f3bd43,#e77d23); }
    [data-testid="stAlert"] { border-radius:12px; }
    .stCaption, [data-testid="stCaptionContainer"] { color:#675f4c !important; }
    [data-testid="stHorizontalBlock"] { align-items:stretch; gap:1rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:0; }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { height:100%; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(13px)} }
    @keyframes orbit-glow { from{transform:translate(-8px,0) scale(.96)} to{transform:translate(8px,14px) scale(1.05)} }
    @keyframes enter { from{opacity:0;transform:translateY(9px)} to{opacity:1;transform:translateY(0)} }
    .footer { color:#6b614b; border-top:1px solid #eadfc5; margin-top:38px; padding:18px 0; font-size:12px; }
    @media (max-width: 760px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .hero { padding:26px 22px; }
      .hero:before { right:-35%; }
      [data-testid="stHorizontalBlock"] { gap:.65rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_models(dataset_json: str):
    frame = pd.read_json(StringIO(dataset_json), orient="split")
    return train_models(frame)


def set_sample(sample: str) -> None:
    st.session_state["comment_input"] = sample


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Trust &amp; Safety · NLP Intelligence</div>
      <h1>Make every conversation<br>safer by design.</h1>
      <p>Explore a layered NLP pipeline that turns raw comments into transparent,
      attention-guided signals — built for thoughtful human review.</p>
      <span class="pill">LSTM + additive attention</span>
      <span class="pill">BoW &amp; TF-IDF benchmarks</span>
      <span class="pill">Explainable preprocessing</span>
      <span class="pill">Local-first demo</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.caption("COMMENT INTELLIGENCE STUDIO")
    st.divider()
    st.markdown("### Training data")
    st.caption("Start with the synthetic demo set, or upload your own labeled CSV.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: comment, label (safe/toxic or 0/1).")
    if uploaded is not None:
        try:
            dataset = load_csv_dataset(uploaded.getvalue())
            st.success(f"Loaded {len(dataset)} validated examples.")
            dataset_source = uploaded.name
        except ValueError as error:
            st.error(str(error))
            dataset = demo_dataset()
            dataset_source = "Synthetic demo corpus (upload not applied)"
    else:
        dataset = demo_dataset()
        dataset_source = "Synthetic demo corpus"
    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        '<span class="pill">Regex cleanup</span><span class="pill">spaCy tokenize</span>'
        '<span class="pill">NLTK lemma + stem</span><span class="pill">TextBlob sentiment</span>',
        unsafe_allow_html=True,
    )
    st.caption("All inference and demo training run locally on your machine.")
    st.divider()
    st.markdown("### Try a sample")
    st.button(
        "✦  Constructive feedback",
        use_container_width=True,
        on_click=set_sample,
        args=("I appreciate your perspective, even though I see it differently.",),
    )
    st.button(
        "⚡  Harmful comment",
        use_container_width=True,
        on_click=set_sample,
        args=("You are worthless and nobody wants you here.",),
    )

metrics = dataset_metrics(dataset)
metric_cols = st.columns(4)
metric_cols[0].metric("Training examples", f"{metrics['samples']:,}")
metric_cols[1].metric("Safe examples", f"{metrics['safe']:,}")
metric_cols[2].metric("Toxic examples", f"{metrics['toxic']:,}")
metric_cols[3].metric("Class balance", f"{metrics['balance']:.0%}")
st.caption(f"Active dataset: **{dataset_source}** · Demo examples are synthetic and not a quality benchmark.")

with st.spinner("Warming up local NLP models..."):
    bundle = cached_models(dataset.to_json(orient="split"))

analyze_tab, pipeline_tab, data_tab = st.tabs(
    ["✦  Analyze a comment", "⌘  NLP pipeline", "▦  Dataset & models"]
)

with analyze_tab:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Comment analysis")
        st.markdown('<p class="muted">Enter a comment to compare three model families.</p>', unsafe_allow_html=True)
        with st.form("analysis_form"):
            comment = st.text_area(
                "Comment",
                key="comment_input",
                height=145,
                placeholder="Paste a comment here to explore its language signals...",
                label_visibility="collapsed",
                max_chars=2000,
            )
            chosen_model = st.selectbox(
                "Primary model",
                ["LSTM · Attention", "TF-IDF · Logistic", "BoW · Naive Bayes"],
                help="The selected model drives the main risk indicator. All three are compared below.",
            )
            submitted = st.form_submit_button("Analyze comment  →", use_container_width=True)
        if submitted and not comment.strip():
            st.warning("Enter a comment before running the analysis.")
        if submitted and comment.strip():
            with st.spinner("Analyzing language signals..."):
                st.session_state["analysis_result"] = {
                    "comment": comment.strip(),
                    "predictions": predict_all(bundle, comment.strip()),
                    "features": nlp_features(comment.strip()),
                    "model": chosen_model,
                }
        result = st.session_state.get("analysis_result")
        if result:
            prediction = result["predictions"][result["model"]]
            toxic = prediction["label"] == "toxic"
            risk_class = "risk-toxic" if toxic else "risk-safe"
            risk_label = "Elevated review signal" if toxic else "Lower review signal"
            st.markdown(
                f'<div class="risk {risk_class}"><div class="muted">MODEL SIGNAL · {result["model"]}</div>'
                f'<div class="risk-label">{"⚠️ " if toxic else "✓ "}{risk_label}</div>'
                f'<div class="muted">Estimated toxic probability: {prediction["probability"]:.1%} · '
                "A model signal is not a verdict.</div></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(prediction["probability"], 0.0), 1.0), text="Estimated toxicity score")
    with right:
        st.subheader("Model comparison")
        if "analysis_result" not in st.session_state:
            st.markdown(
                '<div class="panel"><h3>Three lenses, one conversation</h3>'
                '<p class="muted">Run an analysis to compare a word-frequency baseline, '
                'a TF-IDF classifier and the attention-based sequence model.</p>'
                '<span class="pill">Naive Bayes</span><span class="pill">Logistic regression</span>'
                '<span class="pill">LSTM encoder-decoder</span></div>',
                unsafe_allow_html=True,
            )
        else:
            result = st.session_state["analysis_result"]
            comparison = pd.DataFrame(
                [
                    {
                        "Model": name,
                        "Toxic probability": values["probability"],
                        "Prediction": values["label"].title(),
                    }
                    for name, values in result["predictions"].items()
                ]
            ).set_index("Model")
            st.dataframe(
                comparison.style.format({"Toxic probability": "{:.1%}"}),
                width="stretch",
            )
            st.markdown("**Attention highlights**")
            attention = result["predictions"]["LSTM · Attention"].get("attention", [])
            if attention:
                attention_frame = pd.DataFrame(attention, columns=["Token", "Attention weight"]).set_index("Token")
                st.bar_chart(attention_frame, color="#9d85ff", height=190)
            else:
                st.caption("No tokens available to visualize.")
    if "analysis_result" in st.session_state:
        features = st.session_state["analysis_result"]["features"]
        with st.expander("🔬  Inspect preprocessing & sentiment"):
            feature_cols = st.columns(2)
            feature_cols[0].markdown("**Normalized text**")
            feature_cols[0].code(features["cleaned"] or "(empty)")
            feature_cols[0].markdown("**spaCy tokens**")
            feature_cols[0].write(features["spacy_tokens"])
            feature_cols[1].markdown(f"**Lemmatized tokens** · {features['lemma_source']}")
            feature_cols[1].write(features["tokens"])
            feature_cols[1].markdown("**Porter stems**")
            feature_cols[1].write(features["stems"])
            sent_cols = st.columns(2)
            sent_cols[0].metric("TextBlob polarity", f"{features['polarity']:+.3f}", help="Range from -1 (negative) to +1 (positive).")
            sent_cols[1].metric("Subjectivity", f"{features['subjectivity']:.3f}", help="Range from 0 (objective) to 1 (subjective).")
            st.caption(f"Tokenizer: {features['spacy_model']}. Sentiment is contextual metadata, not a toxicity verdict.")

with pipeline_tab:
    st.subheader("From raw language to review signal")
    st.markdown('<p class="muted">A compact, inspectable pipeline — every stage is visible, and no external API is called.</p>', unsafe_allow_html=True)
    stages = st.columns(5)
    stage_data = [
        ("01", "Normalize", "Regex removes markup, normalizes URLs and masks handles."),
        ("02", "Linguistics", "spaCy tokenization; NLTK WordNet lemmatization and Porter stemming."),
        ("03", "Represent", "Count-based n-grams and TF-IDF features for transparent baselines."),
        ("04", "Attend", "LSTM encoder outputs feed additive attention, then a decoder."),
        ("05", "Review", "Compare scores and highlight attended tokens for a human reviewer."),
    ]
    for column, (number, title, description) in zip(stages, stage_data):
        with column:
            st.markdown(
                f'<div class="panel"><div class="eyebrow">{number} / PIPELINE</div>'
                f'<h3>{title}</h3><p class="muted">{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.info(
        "The encoder-decoder is a compact binary sequence classifier: the encoder reads the comment, "
        "additive attention weights its token states, and a one-step decoder produces the class logit. "
        "Attention highlights help inspection but are not a causal explanation."
    )

with data_tab:
    st.subheader("Dataset & model transparency")
    st.markdown('<p class="muted">Current training corpus preview and design choices.</p>', unsafe_allow_html=True)
    preview_cols = st.columns([1.2, 0.8])
    with preview_cols[0]:
        st.dataframe(dataset, width="stretch", height=360, hide_index=True)
    with preview_cols[1]:
        st.markdown("#### What is being trained")
        st.markdown(
            "- **BoW baseline:** unigram/bigram counts + Multinomial Naive Bayes.\n"
            "- **TF-IDF baseline:** weighted n-grams + balanced Logistic Regression.\n"
            "- **Neural model:** token embeddings + LSTM encoder + additive attention + LSTM decoder.\n"
            "- **NLP features:** regex cleanup, spaCy tokens, NLTK lemma/stem, TextBlob sentiment."
        )
        st.warning(
            "The built-in set is synthetic and deliberately small. Its scores are only a functional "
            "walkthrough, not evidence of real-world accuracy, fairness, or safety."
        )
        csv_bytes = dataset.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download active dataset",
            data=csv_bytes,
            file_name="cybershield_training_data.csv",
            mime="text/csv",
        )

st.markdown(
    '<div class="footer">CYBERSHIELD AI · LOCAL NLP DEMO · Human judgment remains essential. '
    "Use representative, consented data and evaluate responsibly.</div>",
    unsafe_allow_html=True,
)
