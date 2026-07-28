"""NeuroPulse AI — Streamlit inference app for the Skip-Gram + GRU artifact."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="NeuroPulse AI | Mental Health Signal Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def ensure_nltk_data() -> None:
    """Install the small tokenizer resources once when a new host starts."""
    for package, resource in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


@st.cache_resource(show_spinner="Initializing the intelligence layer…")
def load_artifacts():
    """Load precisely the saved model and preprocessing objects used in training."""
    with (ARTIFACTS / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    with (ARTIFACTS / "tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    with (ARTIFACTS / "label_encoder.pkl").open("rb") as file:
        encoder = pickle.load(file)

    # Convert the notebook's model name ("Skip-Gram + GRU") to its saved filename.
    model_file = config["best_model"].lower().replace(" ", "_").replace("+", "") + ".keras"
    model = load_model(ARTIFACTS / model_file, compile=False)
    return model, tokenizer, encoder, config


NEGATIONS = {
    "not", "no", "nor", "never", "don't", "didn't", "doesn't", "can't",
    "won't", "isn't", "aren't", "wasn't", "weren't", "couldn't", "shouldn't",
    "wouldn't", "haven't", "hasn't", "hadn't",
}
URL_FRAGMENT_TOKENS = {
    "http", "https", "www", "com", "org", "net", "gov", "edu", "html", "htm",
    "php", "aspx", "asp", "co", "uk", "jpg", "png", "gif", "pdf", "id",
}
URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)")
HTML_TAG_REGEX = re.compile(r"<.*?>")
NON_WORD_REGEX = re.compile(r"[^\w\s']")
MULTI_SPACE_REGEX = re.compile(r"\s+")


@st.cache_resource
def preprocessing_tools():
    ensure_nltk_data()
    return WordNetLemmatizer(), set(stopwords.words("english")) - NEGATIONS


def clean_text(text: str) -> list[str]:
    """Canonical preprocessing copied from the training notebook."""
    lemmatizer, active_stopwords = preprocessing_tools()
    text = str(text).lower()
    text = URL_REGEX.sub(" ", text)
    text = HTML_TAG_REGEX.sub(" ", text)
    text = NON_WORD_REGEX.sub(" ", text)
    text = MULTI_SPACE_REGEX.sub(" ", text).strip()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in URL_FRAGMENT_TOKENS]
    tokens = [token for token in tokens if token not in active_stopwords]
    return [lemmatizer.lemmatize(token, pos="v") for token in tokens]


def predict(text: str):
    model, tokenizer, encoder, config = load_artifacts()
    tokens = clean_text(text)
    sequence = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(sequence, maxlen=int(config["MAXLEN"]))
    probabilities = model.predict(padded, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return encoder.inverse_transform([index])[0], probabilities, encoder.classes_, tokens


SENTIMENT_MAP = {
    -2: {"name": "Very Negative", "emoji": "🌩️", "color": "#ff5c7a", "glow": "255,92,122",
         "blurb": "Signals of significant distress"},
    -1: {"name": "Negative", "emoji": "🌧️", "color": "#ff9a5c", "glow": "255,154,92",
         "blurb": "Signals of mild distress or worry"},
    0: {"name": "Neutral", "emoji": "🌤️", "color": "#8fa9ff", "glow": "143,169,255",
        "blurb": "Balanced, everyday tone"},
    1: {"name": "Positive", "emoji": "🌈", "color": "#65e6b8", "glow": "101,230,184",
        "blurb": "Signals of hope or wellbeing"},
}

DEFAULT_SENTIMENT = {"name": "Unclassified", "emoji": "❔", "color": "#9dafc7", "glow": "157,175,199",
                      "blurb": "Outside the known intensity scale"}


def sentiment_info(label) -> dict:
    """Map a raw model intensity label to display metadata. Display-only — never used in prediction."""
    try:
        key = int(label)
    except (TypeError, ValueError):
        key = label
    return SENTIMENT_MAP.get(key, DEFAULT_SENTIMENT)


def level_name(label) -> str:
    info = sentiment_info(label)
    return f"{info['emoji']} {info['name']}"


# ---------------------------------------------------------------------------
# Static reference data — pulled directly from the training notebook's own
# printed outputs (dataset shape, value_counts, describe(), results_df).
# Display-only: none of this feeds back into clean_text()/predict().
# ---------------------------------------------------------------------------

DATASET_STATS = {
    "raw_rows": 10392,
    "clean_rows": 10391,
    "dropped_na": 1,
    "duplicates": 0,
    "train_rows": 8312,
    "test_rows": 2079,
    "external_test_rows": 40,
    "external_accuracy": 0.900,
}

CLASS_COUNTS = [
    {"label": 0, "count": 4375},
    {"label": -1, "count": 4112},
    {"label": -2, "count": 1155},
    {"label": 1, "count": 750},
]

TEXT_LENGTH_STATS = {
    "char_length": {"mean": 1178.17, "std": 1583.71, "min": 2, "p25": 452, "p50": 787, "p75": 1344, "max": 30504},
    "word_count": {"mean": 234.14, "std": 289.06, "min": 1, "p25": 93, "p50": 162, "p75": 276, "max": 5413},
}

EMBEDDINGS = [
    {"name": "Skip-Gram (Word2Vec)", "icon": "🎯", "desc": "Predicts context words from a target word — strong on rarer terms."},
    {"name": "Word2Vec (CBOW)", "icon": "🧩", "desc": "Predicts a target word from its context — fast and smooth on frequent terms."},
    {"name": "FastText", "icon": "🧵", "desc": "Adds subword n-grams, so it copes well with typos and unseen word forms."},
    {"name": "GloVe (pretrained)", "icon": "🌐", "desc": "Global co-occurrence vectors pretrained on a huge external corpus."},
]

RNN_HEADS = [
    {"name": "LSTM", "icon": "🔁", "desc": "Long Short-Term Memory — gated memory cell, strong long-range recall."},
    {"name": "GRU", "icon": "⚡", "desc": "Gated Recurrent Unit — lighter than LSTM, trains faster, fewer params."},
    {"name": "BiLSTM", "icon": "↔️", "desc": "Bidirectional LSTM — reads the text forward and backward at once."},
]

# results_df sorted by F1 (macro) desc, exactly as produced in the notebook
MODEL_RESULTS = [
    {"model": "Skip-Gram + GRU",     "accuracy": 0.696489, "f1_macro": 0.639568, "f1_weighted": 0.704475, "roc_auc": 0.893541, "epochs": 11},
    {"model": "Word2Vec + BiLSTM",   "accuracy": 0.671477, "f1_macro": 0.629026, "f1_weighted": 0.680257, "roc_auc": 0.867156, "epochs": 6},
    {"model": "FastText + GRU",      "accuracy": 0.673401, "f1_macro": 0.628163, "f1_weighted": 0.682210, "roc_auc": 0.877625, "epochs": 8},
    {"model": "Word2Vec + GRU",      "accuracy": 0.673401, "f1_macro": 0.627257, "f1_weighted": 0.679407, "roc_auc": 0.882665, "epochs": 7},
    {"model": "GloVe + LSTM",        "accuracy": 0.681097, "f1_macro": 0.625635, "f1_weighted": 0.684128, "roc_auc": 0.876097, "epochs": 8},
    {"model": "Skip-Gram + BiLSTM",  "accuracy": 0.663781, "f1_macro": 0.623509, "f1_weighted": 0.666722, "roc_auc": 0.882277, "epochs": 7},
    {"model": "GloVe + GRU",         "accuracy": 0.667148, "f1_macro": 0.621958, "f1_weighted": 0.673306, "roc_auc": 0.878385, "epochs": 7},
    {"model": "Word2Vec + LSTM",     "accuracy": 0.664262, "f1_macro": 0.621725, "f1_weighted": 0.674789, "roc_auc": 0.866791, "epochs": 8},
    {"model": "Skip-Gram + LSTM",    "accuracy": 0.676287, "f1_macro": 0.620376, "f1_weighted": 0.682453, "roc_auc": 0.877620, "epochs": 7},
    {"model": "GloVe + BiLSTM",      "accuracy": 0.661376, "f1_macro": 0.614965, "f1_weighted": 0.668060, "roc_auc": 0.867833, "epochs": 7},
    {"model": "FastText + LSTM",     "accuracy": 0.663300, "f1_macro": 0.609889, "f1_weighted": 0.669779, "roc_auc": 0.865657, "epochs": 9},
    {"model": "FastText + BiLSTM",   "accuracy": 0.639250, "f1_macro": 0.597889, "f1_weighted": 0.653892, "roc_auc": 0.865170, "epochs": 6},
]

BEST_MODEL = MODEL_RESULTS[0]

EXTERNAL_REPORT = [
    {"label": -2, "precision": 0.83, "recall": 1.00, "f1": 0.91, "support": 10},
    {"label": -1, "precision": 0.88, "recall": 0.70, "f1": 0.78, "support": 10},
    {"label": 0,  "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 10},
    {"label": 1,  "precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 10},
]


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      @keyframes auroraDrift {
        0%,100% { background-position: 10% 0%, 90% 10%, 50% 100%, 0 0; }
        50% { background-position: 16% 6%, 84% 4%, 44% 94%, 0 0; }
      }
      .stApp {
        background:
          radial-gradient(circle at 10% 0%, rgba(99,102,241,.28) 0, transparent 32%),
          radial-gradient(circle at 90% 10%, rgba(236,72,153,.22) 0, transparent 30%),
          radial-gradient(circle at 50% 100%, rgba(45,212,191,.18) 0, transparent 40%),
          linear-gradient(160deg, #050914 0%, #0a0f24 45%, #0c0a1f 100%);
        background-size: 140% 140%, 140% 140%, 140% 140%, 100% 100%;
        animation: auroraDrift 16s ease-in-out infinite;
        color: #eef5ff;
        background-attachment: fixed;
      }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { max-width: 1220px; padding-top: 2.6rem; padding-bottom: 4rem; }

      @keyframes floaty { 0%,100% { transform: translateY(0px);} 50% { transform: translateY(-6px);} }
      @keyframes pulseGlow { 0%,100% { opacity:.55; } 50% { opacity:1; } }
      @keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
      @keyframes scanline { 0% { transform: translateY(-100%); } 100% { transform: translateY(100%); } }

      .brand-row { display:flex; align-items:center; gap:.6rem; margin-bottom:.3rem; }
      .brand-badge {
        font-size: 2.1rem; animation: floaty 3.2s ease-in-out infinite;
        filter: drop-shadow(0 0 18px rgba(139,92,246,.65));
      }
      .eyebrow {
        background: linear-gradient(90deg, #6ee7e1, #a78bfa, #f472b6, #6ee7e1);
        background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: shimmer 6s linear infinite;
        font-size: .8rem; font-weight: 700; letter-spacing: .22em; text-transform: uppercase;
      }
      h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: clamp(2.3rem, 5.6vw, 4.3rem) !important;
        line-height: 1.02 !important; margin: .5rem 0 .8rem !important; letter-spacing: -.04em;
        background: linear-gradient(120deg, #eef5ff 30%, #a78bfa 60%, #6ee7e1 90%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .lead { color: #aab8d4; font-size: 1.06rem; max-width: 760px; line-height: 1.7; }
      .lead .hi { color: #6ee7e1; font-weight: 600; }

      .chip-row { display:flex; gap:.5rem; flex-wrap:wrap; margin: 1rem 0 1.6rem; }
      .chip {
        display:inline-flex; align-items:center; gap:.4rem;
        background: rgba(139,92,246,.12); border: 1px solid rgba(167,139,250,.35);
        color:#d8d3ff; font-size:.8rem; font-weight:600; padding:.35rem .8rem; border-radius:999px;
      }

      .glass {
        position:relative; overflow:hidden;
        background: linear-gradient(145deg, rgba(30,25,60,.55), rgba(10,14,32,.75));
        border: 1px solid rgba(167,139,250,.25);
        border-radius: 26px; padding: 1.5rem;
        box-shadow: 0 20px 70px rgba(76,29,149,.18), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(14px);
        margin-bottom: 1.2rem;
        transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
      }
      .glass:hover {
        transform: translateY(-3px);
        border-color: rgba(167,139,250,.5);
        box-shadow: 0 26px 80px rgba(76,29,149,.28), 0 0 0 1px rgba(110,231,225,.15), inset 0 1px 0 rgba(255,255,255,.06);
      }
      .glass::before {
        content:''; position:absolute; top:0; left:-60%; width:40%; height:100%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,.05), transparent);
        transform: skewX(-20deg); pointer-events:none;
        transition: left .7s ease;
      }
      .glass:hover::before { left: 130%; }
      .glass-header { display:flex; align-items:center; gap:.5rem; margin-bottom:.9rem; }
      .glass-header .icon { font-size:1.2rem; }
      .glass-title { font-family:'Space Grotesk', sans-serif; font-size:1.15rem; font-weight:700; color:#eef5ff; }
      .glass-sub { color:#9dafc7; font-size:.86rem; margin-top:-.5rem; margin-bottom:1rem; }

      .signal-card { text-align:center; padding: 1.2rem 1rem 1.5rem; }
      .signal-emoji {
        font-size: 3.4rem; animation: floaty 3s ease-in-out infinite;
        filter: drop-shadow(0 0 22px var(--glow-color, rgba(110,231,225,.7)));
      }
      .signal { font-family:'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; letter-spacing: -.03em; margin: .3rem 0 .15rem; }
      .signal-blurb { color:#9dafc7; font-size:.86rem; margin-bottom:.6rem; }
      .caption { color: #9dafc7; font-size: .85rem; }

      .dist-row { display:flex; align-items:center; gap:.6rem; margin: .55rem 0; }
      .dist-emoji { font-size:1.05rem; width:1.4rem; text-align:center; }
      .dist-label { flex: 0 0 128px; font-size:.85rem; color:#cfd9ee; font-weight:600; }
      .dist-pct { flex: 0 0 58px; text-align:right; font-size:.82rem; color:#9dafc7; font-variant-numeric: tabular-nums; transition: color .2s ease, transform .2s ease; }
      .dist-row:hover .dist-pct { color:#eef5ff; transform: scale(1.08); }

      /* --- Interactive bars: hover glow + tooltip + grow-in animation --- */
      @keyframes growBar { from { transform: scaleX(0); } to { transform: scaleX(1); } }
      @keyframes barSheen { 0% { background-position: -120% 0; } 100% { background-position: 220% 0; } }

      .dist-track, .metric-track {
        position: relative; cursor: pointer;
        transition: box-shadow .25s ease, background .25s ease;
      }
      .dist-track { flex:1; height:11px; background: rgba(255,255,255,.06); border-radius:999px; overflow:visible; }
      .dist-fill {
        display:block; height:100%; border-radius:999px; transform-origin:left;
        box-shadow: 0 0 12px 0 var(--glow, rgba(110,231,225,.6));
        animation: growBar .9s cubic-bezier(.16,1,.3,1) both;
        transition: filter .2s ease, box-shadow .2s ease;
        overflow: hidden; position: relative;
      }
      .dist-fill::after, .metric-fill::after {
        content:''; position:absolute; inset:0;
        background: linear-gradient(100deg, transparent 20%, rgba(255,255,255,.55) 50%, transparent 80%);
        background-size: 200% 100%; opacity:0; transition: opacity .2s ease;
      }
      .dist-row:hover .dist-fill, .lb-row:hover .metric-fill {
        filter: brightness(1.35) saturate(1.35);
      }
      .dist-row:hover .dist-fill::after, .lb-row:hover .metric-fill::after {
        opacity:1; animation: barSheen 1s linear infinite;
      }
      .dist-track:hover, .metric-track:hover {
        box-shadow: 0 0 0 2px rgba(167,139,250,.45), 0 0 18px rgba(167,139,250,.3);
      }
      .dist-track::after, .metric-track::after {
        content: attr(data-tooltip);
        position: absolute; bottom: 145%; left: 50%; transform: translateX(-50%) translateY(4px);
        background: rgba(8,10,26,.96); border: 1px solid rgba(167,139,250,.5); color:#eef5ff;
        font-size:.72rem; font-weight:700; padding:.4rem .65rem; border-radius:9px; white-space:nowrap;
        opacity:0; pointer-events:none; transition: opacity .2s ease, transform .2s ease; z-index:30;
        box-shadow: 0 10px 26px rgba(0,0,0,.45);
        font-family:'JetBrains Mono', monospace;
      }
      .dist-track:hover::after, .metric-track:hover::after {
        opacity:1; transform: translateX(-50%) translateY(-4px);
      }

      .stTextArea textarea {
        background: rgba(8,10,26,.75) !important; color: #eff8ff !important;
        border: 1px solid rgba(139,92,246,.4) !important; border-radius: 18px !important;
        font-size: 1.02rem !important; padding: 1rem !important;
      }
      .stTextArea textarea:focus { border-color: rgba(110,231,225,.8) !important; box-shadow: 0 0 0 3px rgba(110,231,225,.15) !important; }

      .stButton > button {
        width: 100%; border: 0; border-radius: 16px; padding: .8rem;
        font-weight: 750; font-size: 1rem; color: #05111b;
        background: linear-gradient(90deg, #6ee7e1, #a78bfa, #f472b6);
        background-size: 200% auto;
        transition: all .25s ease;
        box-shadow: 0 8px 30px rgba(139,92,246,.35);
      }
      .stButton > button:hover { background-position: right center; transform: translateY(-1px) scale(1.01); box-shadow: 0 10px 40px rgba(139,92,246,.6), 0 0 24px rgba(110,231,225,.35); }
      .stButton > button:active { transform: translateY(0) scale(.99); }

      [data-testid="stExpander"] { background: rgba(15,18,40,.55); border: 1px solid rgba(139,92,246,.22); border-radius: 18px; }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.03); padding: 6px; border-radius: 16px; border: 1px solid rgba(139,92,246,.18); }
      .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 12px; color:#aab8d4; font-weight:600; font-size:.9rem;
        background: transparent; padding: 0 1rem;
      }
      .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(110,231,225,.18), rgba(167,139,250,.22)) !important;
        color: #eef5ff !important; box-shadow: inset 0 0 0 1px rgba(167,139,250,.4);
      }

      /* Stat grid */
      .stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: .8rem; }
      .stat-card {
        background: rgba(255,255,255,.03); border: 1px solid rgba(139,92,246,.2); border-radius:18px;
        padding: 1rem; text-align:center;
      }
      .stat-num { font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700;
        background: linear-gradient(90deg, #6ee7e1, #a78bfa); -webkit-background-clip:text; background-clip:text; color:transparent; }
      .stat-lbl { color:#9dafc7; font-size:.76rem; margin-top:.25rem; text-transform:uppercase; letter-spacing:.06em; }

      /* Model grid */
      .model-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: .8rem; }
      .model-card {
        background: rgba(255,255,255,.03); border: 1px solid rgba(139,92,246,.2); border-radius:18px;
        padding: 1rem 1.1rem;
      }
      .model-card .m-icon { font-size:1.5rem; }
      .model-card .m-name { font-family:'Space Grotesk', sans-serif; font-weight:700; margin:.35rem 0 .2rem; color:#eef5ff; }
      .model-card .m-desc { color:#9dafc7; font-size:.83rem; line-height:1.5; }

      /* Leaderboard */
      .lb-table { width:100%; border-collapse: separate; border-spacing:0 8px; }
      .lb-table th { text-align:left; color:#9dafc7; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; padding: 0 .8rem; }
      .lb-row td { background: rgba(255,255,255,.03); padding: .7rem .8rem; font-size:.87rem; color:#eef5ff; }
      .lb-row td:first-child { border-radius: 12px 0 0 12px; }
      .lb-row td:last-child { border-radius: 0 12px 12px 0; }
      .lb-row.best { box-shadow: 0 0 0 1px rgba(110,231,225,.5); }
      .lb-row.best td { background: linear-gradient(90deg, rgba(110,231,225,.13), rgba(167,139,250,.13)); }
      .rank-badge {
        display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:8px;
        font-weight:700; font-size:.78rem; background: rgba(139,92,246,.18); color:#d8d3ff;
      }
      .rank-badge.gold { background: linear-gradient(135deg,#facc15,#f59e0b); color:#3a2a00; }
      .metric-track { display:inline-block; width:90px; height:8px; background:rgba(255,255,255,.08); border-radius:99px; vertical-align:middle; margin-left:.6rem; overflow:visible; }
      .metric-fill { display:block; height:100%; border-radius:99px; background: linear-gradient(90deg,#6ee7e1,#a78bfa); position:relative; overflow:hidden; }

      /* Pipeline */
      .pipeline { display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; }
      .pipe-step {
        background: rgba(139,92,246,.1); border:1px solid rgba(167,139,250,.3); border-radius:14px;
        padding:.55rem .9rem; font-size:.82rem; color:#d8d3ff; font-weight:600; white-space:nowrap;
      }
      .pipe-arrow { color:#6ee7e1; font-size:1.1rem; }

      .footer-note {
        margin-top: 2.4rem; padding: 1rem 1.2rem; border-radius: 16px;
        background: rgba(236,72,153,.07); border: 1px solid rgba(236,72,153,.22);
        color:#c9b8d8; font-size:.82rem; display:flex; gap:.6rem; align-items:flex-start;
      }
      .footer-note .icon { animation: pulseGlow 2.4s ease-in-out infinite; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-row"><span class="brand-badge">🧠</span>'
    '<span class="eyebrow">Skip-Gram × GRU Neural Network · Live Signal Explorer</span></div>',
    unsafe_allow_html=True,
)
st.title("NeuroPulse AI\nRead the emotion beneath the words.")
st.markdown(
    '<p class="lead">A futuristic, neural-powered lens into <span class="hi">mental-health text sentiment</span>. '
    'Explore the dataset, compare 12 embedding × architecture combinations, and run live inference — all in one '
    'command deck.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="chip-row">'
    '<span class="chip">🌈 Positive</span>'
    '<span class="chip">🌤️ Neutral</span>'
    '<span class="chip">🌧️ Negative</span>'
    '<span class="chip">🌩️ Very Negative</span>'
    '<span class="chip">✨ Real-time inference</span>'
    '<span class="chip">🧬 12 models benchmarked</span>'
    '</div>',
    unsafe_allow_html=True,
)

tab_analyze, tab_eda, tab_models, tab_leaderboard, tab_about = st.tabs(
    ["🔮 Live Analyzer", "📡 Dataset & EDA", "🧬 Model Zoo", "🏆 Leaderboard", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Live Analyzer (original inference flow, untouched logic)
# ---------------------------------------------------------------------------
with tab_analyze:
    with st.container(border=False):
        st.markdown('<div class="glass"><div class="glass-header"><span class="icon">💬</span>'
                    '<span class="eyebrow" style="font-size:.72rem;">Your text</span></div>', unsafe_allow_html=True)
        text = st.text_area(
            "Text to analyse",
            placeholder="Write or paste a message here… e.g. \"I've been feeling a lot lighter lately, things are looking up.\"",
            height=190,
            label_visibility="collapsed",
        )
        submitted = st.button("🔮  Analyse emotional signal", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if not text.strip():
            st.warning("Add some text first, then analyse it. ✍️")
        else:
            with st.spinner("🧠 Reading the emotional signal…"):
                label, probabilities, classes, tokens = predict(text)
            confidence = float(np.max(probabilities))
            info = sentiment_info(label)

            st.markdown("<br>", unsafe_allow_html=True)
            first, second = st.columns([1, 1.55], gap="large")

            with first:
                st.markdown(
                    f'''<div class="glass signal-card" style="--glow-color: rgba({info["glow"]},.7);">
                        <div class="eyebrow" style="font-size:.72rem;">Predicted sentiment</div>
                        <div class="signal-emoji">{info["emoji"]}</div>
                        <div class="signal" style="color:{info["color"]};">{info["name"]}</div>
                        <div class="signal-blurb">{info["blurb"]}</div>
                        <div class="caption">Raw intensity label · <b>{label}</b></div>
                        <div class="caption">Model confidence · <b>{confidence:.1%}</b></div>
                    </div>''',
                    unsafe_allow_html=True,
                )
                st.progress(confidence)

            with second:
                rows_html = '<div class="glass"><div class="glass-header"><span class="icon">📊</span>' \
                            '<span class="eyebrow" style="font-size:.72rem;">Class distribution</span></div>'
                for class_label, probability in sorted(
                    zip(classes, probabilities), key=lambda p: float(p[1]), reverse=True
                ):
                    c_info = sentiment_info(class_label)
                    pct = float(probability) * 100
                    rows_html += f'''
                    <div class="dist-row">
                        <span class="dist-emoji">{c_info["emoji"]}</span>
                        <span class="dist-label">{c_info["name"]}</span>
                        <span class="dist-track" data-tooltip="{c_info["name"]}: {pct:.1f}% of prediction confidence">
                            <span class="dist-fill" style="width:{pct:.1f}%; background:{c_info["color"]}; --glow: rgba({c_info["glow"]},.6);"></span>
                        </span>
                        <span class="dist-pct">{pct:.1f}%</span>
                    </div>'''
                rows_html += '</div>'
                st.markdown(rows_html, unsafe_allow_html=True)

            with st.expander("🔬 See the model-ready text"):
                st.caption("The tokens below follow the preprocessing pipeline used in the notebook.")
                st.code(" ".join(tokens) if tokens else "No recognised tokens", language=None)

# ---------------------------------------------------------------------------
# TAB 2 — Dataset & EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.markdown(
        f'''<div class="glass">
            <div class="glass-header"><span class="icon">📡</span><span class="glass-title">Dataset snapshot</span></div>
            <div class="glass-sub">Mental-health forum posts, labelled by intensity from −2 (very negative) to 1 (positive).</div>
            <div class="stat-grid">
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['raw_rows']:,}</div><div class="stat-lbl">Raw rows</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['clean_rows']:,}</div><div class="stat-lbl">After cleaning</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['dropped_na']}</div><div class="stat-lbl">Dropped (missing)</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['duplicates']}</div><div class="stat-lbl">Duplicate rows</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['train_rows']:,}</div><div class="stat-lbl">Train split</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['test_rows']:,}</div><div class="stat-lbl">Test split (80/20)</div></div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.1, 1], gap="large")

    with col_a:
        max_count = max(c["count"] for c in CLASS_COUNTS)
        rows_html = '<div class="glass"><div class="glass-header"><span class="icon">⚖️</span>' \
                    '<span class="glass-title">Class balance (imbalanced, real-world)</span></div>'
        total_rows = sum(c["count"] for c in CLASS_COUNTS)
        for c in CLASS_COUNTS:
            info = sentiment_info(c["label"])
            pct_of_max = c["count"] / max_count * 100
            pct_of_total = c["count"] / total_rows * 100
            rows_html += f'''
            <div class="dist-row">
                <span class="dist-emoji">{info["emoji"]}</span>
                <span class="dist-label">{info["name"]}</span>
                <span class="dist-track" data-tooltip="{c['count']:,} of {total_rows:,} posts · {pct_of_total:.1f}% of dataset">
                    <span class="dist-fill" style="width:{pct_of_max:.1f}%; background:{info["color"]}; --glow: rgba({info["glow"]},.6);"></span>
                </span>
                <span class="dist-pct">{c["count"]:,}</span>
            </div>'''
        rows_html += '<div class="caption" style="margin-top:.6rem;">Class 1 (positive) is the rarest — handled during ' \
                     'training with train-only <b>RandomOverSampler</b> resampling, never touching test data.</div></div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    with col_b:
        cl = TEXT_LENGTH_STATS["char_length"]
        wc = TEXT_LENGTH_STATS["word_count"]
        st.markdown(
            f'''<div class="glass">
                <div class="glass-header"><span class="icon">📏</span><span class="glass-title">Post length distribution</span></div>
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-num">{wc['mean']:.0f}</div><div class="stat-lbl">Avg words</div></div>
                    <div class="stat-card"><div class="stat-num">{wc['p50']:.0f}</div><div class="stat-lbl">Median words</div></div>
                    <div class="stat-card"><div class="stat-num">{wc['max']:,.0f}</div><div class="stat-lbl">Max words</div></div>
                    <div class="stat-card"><div class="stat-num">{cl['mean']:.0f}</div><div class="stat-lbl">Avg characters</div></div>
                    <div class="stat-card"><div class="stat-num">{cl['p50']:.0f}</div><div class="stat-lbl">Median characters</div></div>
                    <div class="stat-card"><div class="stat-num">{cl['max']:,.0f}</div><div class="stat-lbl">Max characters</div></div>
                </div>
                <div class="caption" style="margin-top:.8rem;">Posts range from short one-liners to long-form entries — 
                sequences are padded/truncated to a fixed <b>MAXLEN</b> before entering the network.</div>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'''<div class="glass">
            <div class="glass-header"><span class="icon">🧼</span><span class="glass-title">Preprocessing pipeline</span></div>
            <div class="glass-sub">The exact canonical steps applied to every post — in training and at inference.</div>
            <div class="pipeline">
                <span class="pipe-step">📝 Lowercase</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🔗 Strip URLs</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🏷️ Strip HTML tags</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🔤 Remove punctuation</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">✂️ Tokenize</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🚫 Drop URL-fragment tokens</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🛑 Remove stopwords (keep negations)</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🌱 Lemmatize (verb form)</span>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 3 — Model Zoo
# ---------------------------------------------------------------------------
with tab_models:
    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">🧬</span><span class="glass-title">4 embeddings × 3 recurrent heads = 12 models</span></div>
            <div class="glass-sub">Every combination is trained identically — same padding, same class-balancing, same callbacks — so the comparison is fair.</div>
        </div>''',
        unsafe_allow_html=True,
    )

    col_e, col_r = st.columns(2, gap="large")
    with col_e:
        cards = "".join(
            f'''<div class="model-card"><div class="m-icon">{e["icon"]}</div>
                <div class="m-name">{e["name"]}</div><div class="m-desc">{e["desc"]}</div></div>'''
            for e in EMBEDDINGS
        )
        st.markdown(
            f'<div class="glass"><div class="glass-header"><span class="icon">🌱</span>'
            f'<span class="glass-title">Word embeddings</span></div><div class="model-grid">{cards}</div></div>',
            unsafe_allow_html=True,
        )
    with col_r:
        cards = "".join(
            f'''<div class="model-card"><div class="m-icon">{r["icon"]}</div>
                <div class="m-name">{r["name"]}</div><div class="m-desc">{r["desc"]}</div></div>'''
            for r in RNN_HEADS
        )
        st.markdown(
            f'<div class="glass"><div class="glass-header"><span class="icon">🧠</span>'
            f'<span class="glass-title">Recurrent heads</span></div><div class="model-grid">{cards}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">🏗️</span><span class="glass-title">Shared architecture</span></div>
            <div class="pipeline">
                <span class="pipe-step">🔢 Embedding layer</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🔁 RNN (128 units, dropout .2)</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🧮 Dense 64 · ReLU</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">💧 Dropout .5</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🎯 Dense · Softmax (4 classes)</span>
            </div>
            <div class="caption" style="margin-top:.9rem;">Trained with <b>Adam</b>, sparse categorical cross-entropy, 
            <b>EarlyStopping</b> on val_loss (best-epoch weights restored) and <b>ReduceLROnPlateau</b> — so every model 
            trains exactly as long as it needs to.</div>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''<div class="glass" style="--glow-color: rgba(101,230,184,.7);">
            <div class="glass-header"><span class="icon">🏆</span><span class="glass-title">Champion model — {BEST_MODEL['model']}</span></div>
            <div class="glass-sub">Selected as best on the held-out, imbalanced, real-world test split (highest F1 macro) — this is the artifact powering the Live Analyzer.</div>
            <div class="stat-grid">
                <div class="stat-card"><div class="stat-num">{BEST_MODEL['accuracy']:.1%}</div><div class="stat-lbl">Accuracy</div></div>
                <div class="stat-card"><div class="stat-num">{BEST_MODEL['f1_macro']:.3f}</div><div class="stat-lbl">F1 macro</div></div>
                <div class="stat-card"><div class="stat-num">{BEST_MODEL['f1_weighted']:.3f}</div><div class="stat-lbl">F1 weighted</div></div>
                <div class="stat-card"><div class="stat-num">{BEST_MODEL['roc_auc']:.3f}</div><div class="stat-lbl">ROC-AUC (macro OvR)</div></div>
                <div class="stat-card"><div class="stat-num">{BEST_MODEL['epochs']}</div><div class="stat-lbl">Epochs trained</div></div>
                <div class="stat-card"><div class="stat-num">{DATASET_STATS['external_accuracy']:.0%}</div><div class="stat-lbl">External test accuracy</div></div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 4 — Leaderboard
# ---------------------------------------------------------------------------
with tab_leaderboard:
    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">🏆</span><span class="glass-title">Model comparison — all 12 combinations</span></div>
            <div class="glass-sub">Ranked by F1 macro on the held-out test split (imbalanced, never resampled).</div>
        </div>''',
        unsafe_allow_html=True,
    )

    max_f1 = max(r["f1_macro"] for r in MODEL_RESULTS)
    rows = ""
    medals = ["gold", "", ""]
    for i, r in enumerate(MODEL_RESULTS):
        is_best = i == 0
        badge_cls = "rank-badge gold" if i == 0 else "rank-badge"
        badge = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i+1}"))
        bar_pct = r["f1_macro"] / max_f1 * 100
        rows += f'''
        <tr class="lb-row {'best' if is_best else ''}">
            <td><span class="{badge_cls}">{badge}</span></td>
            <td><b>{r['model']}</b>{' <span class="chip" style="padding:.1rem .5rem;font-size:.68rem;">CHAMPION</span>' if is_best else ''}</td>
            <td>{r['accuracy']:.1%}</td>
            <td>{r['f1_macro']:.3f}<span class="metric-track" data-tooltip="{r['f1_macro']:.3f} · {bar_pct:.0f}% of top score ({max_f1:.3f})"><span class="metric-fill" style="width:{bar_pct:.0f}%;"></span></span></td>
            <td>{r['f1_weighted']:.3f}</td>
            <td>{r['roc_auc']:.3f}</td>
            <td>{r['epochs']}</td>
        </tr>'''

    table_html = f'''
    <div class="glass" style="overflow-x:auto;">
    <table class="lb-table">
        <thead><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>F1 (macro)</th><th>F1 (weighted)</th><th>ROC-AUC</th><th>Epochs</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>'''
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">🧪</span><span class="glass-title">Champion model — external validation report</span></div>
            <div class="glass-sub">Skip-Gram + GRU evaluated on a fully separate, never-seen <code>test_data.xlsx</code> file (10 posts per class) — 90% accuracy.</div>
        </div>''',
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="medium")
    for col, row in zip(cols, EXTERNAL_REPORT):
        info = sentiment_info(row["label"])
        with col:
            st.markdown(
                f'''<div class="glass signal-card" style="padding:1rem; --glow-color: rgba({info["glow"]},.6);">
                    <div style="font-size:1.8rem;">{info["emoji"]}</div>
                    <div style="font-weight:700;color:{info["color"]};margin:.2rem 0;">{info["name"]}</div>
                    <div class="caption">Precision · {row['precision']:.0%}</div>
                    <div class="caption">Recall · {row['recall']:.0%}</div>
                    <div class="caption">F1 · {row['f1']:.0%}</div>
                    <div class="caption">n = {row['support']}</div>
                </div>''',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB 5 — About
# ---------------------------------------------------------------------------
with tab_about:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '''<div class="glass">
                <div class="glass-header"><span class="icon">📚</span><span class="glass-title">About the dataset</span></div>
                <div class="caption" style="line-height:1.8;">
                Posts sourced from mental-health-related forum discussions, each hand-labelled with an
                <b>intensity</b> score reflecting the emotional tone of the writer, on a 4-point scale from
                <b>−2 (very negative)</b> through <b>1 (positive)</b>. Text length varies widely — from short
                comments to long-form personal entries — reflecting real, unfiltered writing rather than
                curated samples. The class distribution is naturally imbalanced (neutral and negative posts
                dominate), which the training pipeline addresses via train-only oversampling while keeping
                evaluation on the true, imbalanced distribution.
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '''<div class="glass">
                <div class="glass-header"><span class="icon">🤖</span><span class="glass-title">About the models</span></div>
                <div class="caption" style="line-height:1.8;">
                Twelve neural classifiers were trained — every pairing of <b>4 word-embedding strategies</b>
                (Skip-Gram Word2Vec, CBOW Word2Vec, FastText, pretrained GloVe) with <b>3 recurrent architectures</b>
                (LSTM, GRU, BiLSTM). Each network embeds tokens, passes them through its recurrent head, then a
                dense + dropout block before a softmax over the 4 intensity classes. All 12 were trained under
                identical conditions (same padding, same class-balancing, same early-stopping) so the leaderboard
                is an apples-to-apples comparison. The <b>Skip-Gram + GRU</b> model won on F1 macro and is the one
                deployed in the Live Analyzer.
                </div>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">⚙️</span><span class="glass-title">Inference pipeline (what happens when you click Analyse)</span></div>
            <div class="pipeline">
                <span class="pipe-step">✍️ Raw text</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🧼 clean_text()</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🔢 Tokenizer → sequence</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">📐 Pad to MAXLEN</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🧠 Skip-Gram + GRU</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🎯 Softmax probabilities</span><span class="pipe-arrow">→</span>
                <span class="pipe-step">🏷️ LabelEncoder → intensity</span>
            </div>
            <div class="caption" style="margin-top:.9rem;">The <b>exact</b> tokenizer, label encoder and MAXLEN saved
            during training are reloaded at inference time — never re-fit — so predictions match training behaviour precisely.</div>
        </div>''',
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer-note'><span class='icon'>💙</span>"
    "<span>For education and demonstration only — this tool is not a diagnosis, crisis service, "
    "or substitute for a qualified mental-health professional. If you or someone you know is struggling, "
    "please reach out to a licensed professional or a local crisis line.</span></div>",
    unsafe_allow_html=True,
)