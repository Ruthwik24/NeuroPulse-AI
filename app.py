"""NeuroPulse AI — Streamlit inference app for the Skip-Gram + GRU artifact."""

from __future__ import annotations

import io
import json
import pickle
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
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
    -2: {"name": "Very Negative", "emoji": "🌩️", "color": "#ff4d6d", "glow": "255,77,109",
         "blurb": "Signals of significant distress", "mood": "😞"},
    -1: {"name": "Negative", "emoji": "🌧️", "color": "#ff9f43", "glow": "255,159,67",
         "blurb": "Signals of mild distress or worry", "mood": "😕"},
    0: {"name": "Neutral", "emoji": "🌤️", "color": "#ffd93d", "glow": "255,217,61",
        "blurb": "Balanced, everyday tone", "mood": "😐"},
    1: {"name": "Positive", "emoji": "🌈", "color": "#34d399", "glow": "52,211,153",
        "blurb": "Signals of hope or wellbeing", "mood": "🙂"},
}

DEFAULT_SENTIMENT = {"name": "Unclassified", "emoji": "❔", "color": "#a7b8cf", "glow": "167,184,207",
                      "blurb": "Outside the known intensity scale", "mood": "❔"}

# Sample statements the person can tap to try the analyzer instantly — one
# gentle, illustrative example per intensity class.
EXAMPLE_STATEMENTS = {
    1: [
        "I finally slept through the night and woke up actually feeling like myself again.",
        "Talking to my therapist today helped me see how far I've really come.",
        "Started running again this week and honestly, I forgot how good it feels to be proud of myself.",
    ],
    0: [
        "Went to work, made dinner, watched a show. Pretty ordinary day overall.",
        "Still figuring out my routine, nothing exciting to report either way.",
        "Appointment got rescheduled to next Tuesday, so I guess I'll just wait.",
    ],
    -1: [
        "I've been feeling kind of low and unmotivated the past few days.",
        "Work has been overwhelming and I can't seem to shake this worry.",
        "I keep replaying that conversation in my head and I don't know why it's bothering me so much.",
    ],
    -2: [
        "I feel completely numb and don't see the point in anything anymore.",
        "Everything feels hopeless right now and I don't know how to keep going.",
        "I can't remember the last time I felt anything other than exhausted and empty.",
    ],
}

# ---------------------------------------------------------------------------
# Live theme state — the whole interface's accent colour tracks the most
# recently predicted sentiment, like a nervous system reacting to signal.
# Reverts to a calm resting-state teal whenever the input is cleared.
# ---------------------------------------------------------------------------
RESTING_COLOR = "#8b5cf6"
RESTING_GLOW = "139,92,246"

if "theme_color" not in st.session_state:
    st.session_state.theme_color = RESTING_COLOR
    st.session_state.theme_glow = RESTING_GLOW
    st.session_state.last_result = None  # (text, label, probabilities, classes, tokens)
if "analyzer_text_area" not in st.session_state:
    st.session_state.analyzer_text_area = ""


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
    f"""
    <style>
      :root {{
        --accent: {st.session_state.theme_color};
        --accent-glow: {st.session_state.theme_glow};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      @keyframes auroraDrift {
        0%,100% { background-position: 10% 0%, 90% 10%, 50% 100%, 0 0; }
        50% { background-position: 16% 6%, 84% 4%, 44% 94%, 0 0; }
      }
      .stApp {
        background:
          radial-gradient(circle at 8% 4%, rgba(255,110,180,.30) 0, transparent 32%),
          radial-gradient(circle at 92% 8%, rgba(96,180,255,.32) 0, transparent 34%),
          radial-gradient(circle at 76% 88%, rgba(255,196,64,.24) 0, transparent 36%),
          radial-gradient(circle at 15% 92%, rgba(120,255,180,.24) 0, transparent 38%),
          radial-gradient(circle at 50% 45%, rgba(168,120,255,.20) 0, transparent 46%),
          linear-gradient(160deg, #12081f 0%, #0d1430 40%, #071626 75%, #0a0f22 100%);
        background-size: 150% 150%, 150% 150%, 150% 150%, 150% 150%, 160% 160%, 100% 100%;
        animation: auroraDrift 18s ease-in-out infinite;
        color: #f2f0ff;
        background-attachment: fixed;
        transition: background 1.2s ease;
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
        background: linear-gradient(90deg, #ff8fd6, var(--accent), #7ee0ff, #ffe08a, #ff8fd6);
        background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: shimmer 6s linear infinite;
        font-size: .8rem; font-weight: 700; letter-spacing: .22em; text-transform: uppercase;
      }
      h1 {
        font-family: 'Orbitron', 'Space Grotesk', sans-serif !important;
        font-size: clamp(2.1rem, 5.2vw, 3.9rem) !important;
        line-height: 1.08 !important; margin: .5rem 0 .8rem !important; letter-spacing: -.01em;
        background: linear-gradient(120deg, #ffb3ec 8%, #a789ff 38%, #7ee0ff 66%, var(--accent) 92%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        text-shadow: 0 0 34px rgba(var(--accent-glow),.3);
      }
      .lead { color: #aab8d4; font-size: 1.06rem; max-width: 760px; line-height: 1.7; }
      .lead .hi { color: var(--accent); font-weight: 600; }

      .chip-row { display:flex; gap:.5rem; flex-wrap:wrap; margin: 1rem 0 1.6rem; }
      .chip {
        display:inline-flex; align-items:center; gap:.4rem;
        background: linear-gradient(90deg, rgba(255,143,214,.16), rgba(126,224,255,.16));
        border: 1px solid rgba(190,150,255,.35);
        color:#f2ecff; font-size:.8rem; font-weight:600; padding:.35rem .8rem; border-radius:999px;
      }

      .glass {
        position:relative; overflow:hidden;
        background: linear-gradient(145deg, rgba(48,40,80,.5), rgba(18,20,42,.68));
        border: 1px solid rgba(190,150,255,.22);
        border-radius: 26px; padding: 1.5rem;
        box-shadow: 0 16px 50px rgba(60,20,90,.25), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(14px);
        margin-bottom: 1.2rem;
        transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
      }
      .glass:hover {
        transform: translateY(-3px);
        border-color: rgba(var(--accent-glow),.5);
        box-shadow: 0 26px 80px rgba(13,60,110,.32), 0 0 0 1px rgba(var(--accent-glow),.18), inset 0 1px 0 rgba(255,255,255,.06);
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
      .glass-title { font-family:'Orbitron', 'Space Grotesk', sans-serif; font-size:1.05rem; font-weight:700; color:#eef5ff; letter-spacing:.01em; }
      .glass-sub { color:#9dafc7; font-size:.86rem; margin-top:-.5rem; margin-bottom:1rem; }

      .signal-card { text-align:center; padding: 1.2rem 1rem 1.5rem; }
      .signal-emoji {
        font-size: 3.4rem; animation: floaty 3s ease-in-out infinite;
        filter: drop-shadow(0 0 22px var(--glow-color, rgba(110,231,225,.7)));
      }
      .signal { font-family:'Orbitron', 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; letter-spacing: 0; margin: .3rem 0 .15rem; }
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
        box-shadow: 0 0 0 2px rgba(var(--accent-glow),.45), 0 0 18px rgba(var(--accent-glow),.3);
      }
      .dist-track::after, .metric-track::after {
        content: attr(data-tooltip);
        position: absolute; bottom: 145%; left: 50%; transform: translateX(-50%) translateY(4px);
        background: rgba(6,10,24,.96); border: 1px solid rgba(var(--accent-glow),.5); color:#eef5ff;
        font-size:.72rem; font-weight:700; padding:.4rem .65rem; border-radius:9px; white-space:nowrap;
        opacity:0; pointer-events:none; transition: opacity .2s ease, transform .2s ease; z-index:30;
        box-shadow: 0 10px 26px rgba(0,0,0,.45);
        font-family:'JetBrains Mono', monospace;
      }
      .dist-track:hover::after, .metric-track:hover::after {
        opacity:1; transform: translateX(-50%) translateY(-4px);
      }

      .stTextArea textarea {
        background: rgba(6,10,24,.75) !important; color: #eff8ff !important;
        border: 1px solid rgba(88,101,242,.45) !important; border-radius: 18px !important;
        font-size: 1.02rem !important; padding: 1rem !important;
      }
      .stTextArea textarea:focus { border-color: rgba(var(--accent-glow),.85) !important; box-shadow: 0 0 0 3px rgba(var(--accent-glow),.15) !important; }

      @keyframes btnGradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      @keyframes btnPulseRing {
        0% { box-shadow: 0 0 0 0 rgba(var(--accent-glow),.55), 0 8px 30px rgba(88,101,242,.4); }
        70% { box-shadow: 0 0 0 12px rgba(var(--accent-glow),0), 0 8px 30px rgba(88,101,242,.4); }
        100% { box-shadow: 0 0 0 0 rgba(var(--accent-glow),0), 0 8px 30px rgba(88,101,242,.4); }
      }
      @keyframes btnScan {
        0% { transform: translateX(-120%) skewX(-20deg); }
        100% { transform: translateX(220%) skewX(-20deg); }
      }
      .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        position: relative; isolation: isolate; overflow: hidden;
        width: 100%; border: 1px solid rgba(124,255,178,.55); border-radius: 14px;
        padding: .85rem 1rem;
        font-family: 'Orbitron', 'Space Grotesk', sans-serif !important;
        font-weight: 700; font-size: .95rem; letter-spacing: .04em; text-transform: uppercase;
        color: #1a0f2e;
        background: linear-gradient(100deg, #ff8fd6, var(--accent), #7ee0ff, #ffe08a, #ff8fd6);
        background-size: 320% auto;
        animation: btnGradientFlow 5s ease-in-out infinite, btnPulseRing 3.2s ease-in-out infinite;
        transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .25s ease, filter .25s ease;
      }
      .stButton > button::before, .stDownloadButton > button::before, .stFormSubmitButton > button::before {
        content: ''; position: absolute; inset: -1px; z-index: -1; border-radius: inherit;
        background: linear-gradient(90deg, var(--accent), #5865f2, #7cffb2, var(--accent));
        background-size: 300% auto; filter: blur(10px); opacity: .0;
        transition: opacity .25s ease;
      }
      .stButton > button::after, .stDownloadButton > button::after, .stFormSubmitButton > button::after {
        content: ''; position: absolute; top: 0; left: 0; width: 35%; height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,.75), transparent);
        transform: translateX(-120%) skewX(-20deg); pointer-events: none;
      }
      .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        filter: saturate(1.25) brightness(1.06);
        box-shadow: 0 14px 46px rgba(88,101,242,.65), 0 0 30px rgba(var(--accent-glow),.55), 0 0 0 1px rgba(124,255,178,.7);
        animation-play-state: running;
      }
      .stButton > button:hover::before, .stDownloadButton > button:hover::before, .stFormSubmitButton > button:hover::before { opacity: .85; }
      .stButton > button:hover::after, .stDownloadButton > button:hover::after, .stFormSubmitButton > button:hover::after {
        animation: btnScan .9s ease forwards;
      }
      .stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {
        transform: translateY(0) scale(.97);
        filter: brightness(.95);
      }
      .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: none; box-shadow: 0 0 0 3px rgba(var(--accent-glow),.5), 0 14px 46px rgba(88,101,242,.6);
      }
      .stButton > button p, .stDownloadButton > button p { font-family: inherit !important; letter-spacing: inherit !important; }

      [data-testid="stExpander"] { background: rgba(10,14,34,.55); border: 1px solid rgba(88,101,242,.25); border-radius: 18px; }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.03); padding: 6px; border-radius: 16px; border: 1px solid rgba(88,101,242,.2); }
      .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 12px; color:#aab8d4; font-weight:600; font-size:.9rem;
        background: transparent; padding: 0 1rem;
      }
      .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(var(--accent-glow),.18), rgba(88,101,242,.24)) !important;
        color: #eef5ff !important; box-shadow: inset 0 0 0 1px rgba(var(--accent-glow),.4), 0 0 18px rgba(var(--accent-glow),.25);
        animation: pulseGlow 3s ease-in-out infinite;
      }
      .stTabs [data-baseweb="tab"]:hover { color:#eef5ff !important; background: rgba(var(--accent-glow),.08) !important; }

      [data-testid="stExpander"]:hover { border-color: rgba(var(--accent-glow),.5); box-shadow: 0 0 22px rgba(var(--accent-glow),.15); }

      /* Stat grid */
      .stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: .8rem; }
      .stat-card {
        background: rgba(255,255,255,.03); border: 1px solid rgba(88,101,242,.22); border-radius:18px;
        padding: 1rem; text-align:center;
      }
      .stat-num { font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700;
        background: linear-gradient(90deg, var(--accent), #5865f2); -webkit-background-clip:text; background-clip:text; color:transparent; }
      .stat-lbl { color:#9dafc7; font-size:.76rem; margin-top:.25rem; text-transform:uppercase; letter-spacing:.06em; }

      /* Model grid */
      .model-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: .8rem; }
      .model-card {
        background: rgba(255,255,255,.03); border: 1px solid rgba(88,101,242,.22); border-radius:18px;
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
        font-weight:700; font-size:.78rem; background: rgba(88,101,242,.2); color:#c9e8ff;
      }
      .rank-badge.gold { background: linear-gradient(135deg,#facc15,#f59e0b); color:#3a2a00; }
      .metric-track { display:inline-block; width:90px; height:8px; background:rgba(255,255,255,.08); border-radius:99px; vertical-align:middle; margin-left:.6rem; overflow:visible; }
      .metric-fill { display:block; height:100%; border-radius:99px; background: linear-gradient(90deg,var(--accent),#5865f2); position:relative; overflow:hidden; }

      /* Pipeline */
      .pipeline { display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; }
      .pipe-step {
        background: rgba(88,101,242,.1); border:1px solid rgba(88,101,242,.32); border-radius:14px;
        padding:.55rem .9rem; font-size:.82rem; color:#c9e8ff; font-weight:600; white-space:nowrap;
      }
      .pipe-arrow { color:var(--accent); font-size:1.1rem; }

      .nerve-status {
        display:inline-flex; align-items:center; gap:.4rem; margin-left:.7rem;
        padding:.25rem .7rem; border-radius:999px; font-size:.72rem; font-weight:700;
        color:#eef5ff; letter-spacing:.03em;
        background: rgba(255,255,255,.04);
        border: 1px solid var(--glow-color, rgba(34,227,255,.6));
        box-shadow: 0 0 16px 0 var(--glow-color, rgba(34,227,255,.4));
        transition: box-shadow .4s ease, border-color .4s ease;
      }
      .nerve-dot {
        width:7px; height:7px; border-radius:50%;
        background: var(--accent); box-shadow: 0 0 10px 2px var(--accent);
        animation: pulseGlow 1.6s ease-in-out infinite;
      }

      .footer-note {
        margin-top: 2.4rem; padding: 1rem 1.2rem; border-radius: 16px;
        background: rgba(var(--accent-glow),.06); border: 1px solid rgba(var(--accent-glow),.2);
        color:#bcd6e8; font-size:.82rem; display:flex; gap:.6rem; align-items:flex-start;
      }
      .footer-note .icon { animation: pulseGlow 2.4s ease-in-out infinite; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Anatomical nervous-system background: a brain, a curving spinal cord, and
# bilateral peripheral-nerve branches (cervical/brachial, intercostal,
# lumbosacral) fanning outward — pulses of "signal" travel down each path.
# ---------------------------------------------------------------------------
BRAIN_PATH = (
    "M50,4 C40,3 32,8 30,15 C24,16 20,22 22,28 C19,32 20,38 25,40 "
    "C26,45 32,48 38,47 C42,50 48,51 52,49 C58,51 65,49 68,44 "
    "C74,44 78,39 77,33 C81,29 80,22 75,19 C75,12 68,6 60,6 C57,4 53,3 50,4 Z"
)
BRAIN_FOLDS = [
    "M32,14 C36,18 36,24 31,27",
    "M42,9 C44,16 40,22 44,28",
    "M54,8 C56,15 60,20 57,28",
    "M64,13 C68,17 67,24 71,29",
    "M38,33 C43,36 48,35 52,38",
]

# Each branch trunk starts at a point on the spine and fans outward+downward;
# every trunk ends in 2 fine "dendrite" twigs. Right-side coordinates only —
# mirrored across x=50 at render time to give the left side automatically.
SPINE_D = "M50,30 C47,50 53,72 49,94 C46,114 52,134 50,158"

NERVE_TRUNKS_RIGHT = [
    {"pts": [(50, 34), (61, 30), (73, 24), (84, 18)], "twigs": [[(84, 18), (91, 12)], [(84, 18), (90, 24)]]},
    {"pts": [(50, 46), (63, 49), (76, 51), (89, 52)], "twigs": [[(89, 52), (96, 46)], [(89, 52), (95, 59)]]},
    {"pts": [(49, 58), (64, 60), (79, 60)], "twigs": [[(79, 60), (86, 56)], [(79, 60), (86, 64)]]},
    {"pts": [(48, 70), (63, 73), (77, 74)], "twigs": [[(77, 74), (84, 70)], [(77, 74), (84, 79)]]},
    {"pts": [(48, 84), (61, 88), (72, 90)], "twigs": [[(72, 90), (79, 86)], [(72, 90), (79, 94)]]},
    {"pts": [(49, 98), (60, 105), (69, 110)], "twigs": [[(69, 110), (76, 106)], [(69, 110), (76, 114)]]},
    {"pts": [(50, 118), (57, 130), (63, 144), (67, 160)], "twigs": [[(67, 160), (71, 156)], [(67, 160), (64, 166)]]},
]


NETWORK_NODES = [
    (35, 12), (45, 10), (55, 11), (65, 14), (30, 20), (42, 22), (58, 21),
    (70, 20), (36, 30), (50, 32), (64, 30), (40, 40), (60, 40), (50, 44),
]
NETWORK_EDGES = (
    [(i, (i + 1) % len(NETWORK_NODES)) for i in range(len(NETWORK_NODES))]
    + [(i, (i + 4) % len(NETWORK_NODES)) for i in range(len(NETWORK_NODES))]
)

# Four fixed markers — one per emotion class — pulsing in their own true
# sentiment colour (independent of the live accent) so the background always
# reads as "the four human emotions" this model was trained to recognise.
EMOTION_MARKERS = [
    (14, 52, 1), (86, 52, 0), (14, 122, -1), (86, 122, -2),
]


def _mirror(points):
    return [(100 - x, y) for x, y in points]


def _polyline_d(points):
    x0, y0 = points[0]
    d = f"M{x0:.1f},{y0:.1f} "
    d += " ".join(f"L{x:.1f},{y:.1f}" for x, y in points[1:])
    return d


def _build_neural_bg() -> str:
    """Fixed-position SVG blending a brain, an internal neural-network mesh,
    a spinal cord with branching peripheral nerves, and four pulsing
    emotion markers — brain + neural net + nervous system + human emotion,
    all in one living diagram."""
    fold_paths = "".join(f'<path class="fold" d="{d}"/>' for d in BRAIN_FOLDS)

    nerve_paths = []
    ganglia = []
    twig_paths = []
    for i, trunk in enumerate(NERVE_TRUNKS_RIGHT):
        for side_points in (trunk["pts"], _mirror(trunk["pts"])):
            delay = (i * 0.55) % 4.2
            nerve_paths.append(
                f'<path class="nerve" d="{_polyline_d(side_points)}" style="animation-delay:{delay:.2f}s"/>'
            )
            gx, gy = side_points[-1]
            ganglia.append(f'<circle class="ganglion" cx="{gx:.1f}" cy="{gy:.1f}" r="0.85" '
                            f'style="animation-delay:{delay:.2f}s"/>')
        for side_twigs in (trunk["twigs"], [_mirror(t) for t in trunk["twigs"]]):
            for twig in side_twigs:
                twig_paths.append(
                    f'<path class="twig" d="{_polyline_d(twig)}" style="animation-delay:{(delay + 0.3):.2f}s"/>'
                )

    net_edges = "".join(
        f'<line class="net-edge" x1="{NETWORK_NODES[a][0]}" y1="{NETWORK_NODES[a][1]}" '
        f'x2="{NETWORK_NODES[b][0]}" y2="{NETWORK_NODES[b][1]}" '
        f'style="animation-delay:{((a + b) * 0.21) % 4:.2f}s"/>'
        for a, b in NETWORK_EDGES
    )
    net_nodes = "".join(
        f'<circle class="net-node" cx="{x}" cy="{y}" r="0.6" style="animation-delay:{(i * 0.33) % 3.6:.2f}s"/>'
        for i, (x, y) in enumerate(NETWORK_NODES)
    )

    emo_markers = "".join(
        f'<g class="emo-marker" style="--ex:{x}; --ey:{y}; animation-delay:{abs(label) * 0.4:.2f}s">'
        f'<circle class="emo-halo" cx="{x}" cy="{y}" r="3.4" fill="{sentiment_info(label)["color"]}"/>'
        f'<text class="emo-glyph" x="{x}" y="{y + 1.3}" text-anchor="middle">{sentiment_info(label)["mood"]}</text>'
        f'</g>'
        for x, y, label in EMOTION_MARKERS
    )

    return f'''
    <div class="neural-bg">
        <svg viewBox="0 0 100 170" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
            <g class="emo-markers">{emo_markers}</g>
            <g class="nerves">{''.join(nerve_paths)}</g>
            <g class="twigs">{''.join(twig_paths)}</g>
            <path class="spine" d="{SPINE_D}"/>
            <path class="brain-outline" d="{BRAIN_PATH}"/>
            <g class="brain-folds">{fold_paths}</g>
            <g class="net-edges">{net_edges}</g>
            <g class="net-nodes">{net_nodes}</g>
            <g class="ganglia">{''.join(ganglia)}</g>
        </svg>
    </div>
    '''


st.markdown(
    """
    <style>
      .neural-bg {
        position: fixed; inset: 0; width: 100vw; height: 100vh;
        z-index: 0; pointer-events: none; overflow: hidden;
        opacity: .5;
      }
      .neural-bg svg { width: 100%; height: 100%; }

      .brain-outline {
        fill: url(#brainFill); fill-opacity: .16;
        stroke: url(#nerveGrad); stroke-width: .35; stroke-linejoin: round;
        filter: drop-shadow(0 0 2.5px rgba(var(--accent-glow),.55));
        animation: brainPulse 4.8s ease-in-out infinite;
      }
      .brain-folds .fold {
        fill: none; stroke: rgba(var(--accent-glow),.55); stroke-width: .28;
        stroke-linecap: round; opacity: .5;
        animation: foldGlow 3.6s ease-in-out infinite;
      }
      .spine {
        fill: none; stroke: url(#nerveGrad); stroke-width: .55; stroke-linecap: round;
        opacity: .5; filter: drop-shadow(0 0 2px rgba(var(--accent-glow),.5));
      }
      .nerve {
        fill: none; stroke: url(#nerveGrad); stroke-width: .32; stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 2.4 4.5; opacity: .38;
        animation: pulseTravel 3.4s linear infinite;
      }
      .twig {
        fill: none; stroke: rgba(var(--accent-glow),.5); stroke-width: .22; stroke-linecap: round;
        opacity: .3; stroke-dasharray: 1.2 2.4;
        animation: pulseTravel 3.4s linear infinite;
      }
      .ganglion {
        fill: var(--accent); opacity: .5;
        filter: drop-shadow(0 0 1.6px rgba(var(--accent-glow),.9));
        animation: neuronFire 3.4s ease-in-out infinite;
      }
      .net-edge {
        stroke: url(#nerveGrad); stroke-width: .14; opacity: .3;
        stroke-dasharray: 1.4 1.6;
        animation: netFire 3s linear infinite;
      }
      .net-node {
        fill: #ffe08a; opacity: .45;
        filter: drop-shadow(0 0 1.2px rgba(var(--accent-glow),.8));
        animation: neuronFire 2.8s ease-in-out infinite;
      }
      .emo-marker { transform-box: fill-box; transform-origin: center; animation: emoDrift 6s ease-in-out infinite; }
      .emo-halo { opacity: .16; animation: emoPulse 3.2s ease-in-out infinite; }
      .emo-glyph { font-size: 3.6px; opacity: .55; animation: emoPulse 3.2s ease-in-out infinite; }
      @keyframes netFire {
        0%   { stroke-dashoffset: 12; opacity: .12; }
        50%  { stroke-dashoffset: 0; opacity: .42; }
        100% { stroke-dashoffset: -12; opacity: .12; }
      }
      @keyframes emoPulse {
        0%, 100% { opacity: .18; }
        50% { opacity: .65; }
      }
      @keyframes emoDrift {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-2px); }
      }
      @keyframes pulseTravel {
        0%   { stroke-dashoffset: 34; opacity: .12; }
        45%  { opacity: .55; }
        50%  { stroke-dashoffset: 0; opacity: .62; }
        55%  { opacity: .5; }
        100% { stroke-dashoffset: -34; opacity: .12; }
      }
      @keyframes neuronFire {
        0%, 100% { opacity: .3; r: .7; }
        50% { opacity: 1; r: 1.15; filter: drop-shadow(0 0 3px rgba(var(--accent-glow),1)); }
      }
      @keyframes brainPulse {
        0%, 100% { fill-opacity: .12; stroke-opacity: .6; }
        50% { fill-opacity: .24; stroke-opacity: 1; }
      }
      @keyframes foldGlow {
        0%, 100% { opacity: .3; }
        50% { opacity: .75; }
      }
      /* Keep real app content above the nervous-system layer */
      .block-container, [data-testid="stHeader"] { position: relative; z-index: 1; }
    </style>
    <svg width="0" height="0">
      <defs>
        <linearGradient id="nerveGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:var(--accent)"/>
          <stop offset="35%" stop-color="#ff8fd6"/>
          <stop offset="65%" stop-color="#7ee0ff"/>
          <stop offset="100%" stop-color="#ffe08a"/>
        </linearGradient>
        <radialGradient id="brainFill" cx="50%" cy="35%" r="65%">
          <stop offset="0%" stop-color="var(--accent)"/>
          <stop offset="100%" stop-color="#7ee0ff" stop-opacity="0"/>
        </radialGradient>
      </defs>
    </svg>
    """,
    unsafe_allow_html=True,
)
st.markdown(_build_neural_bg(), unsafe_allow_html=True)

_live = st.session_state.last_result
_live_info = sentiment_info(_live["label"]) if _live else {
    "name": "Resting potential", "emoji": "🧠", "color": "#22e3ff", "glow": "34,227,255",
}

st.markdown(
    f'<div class="brand-row"><span class="brand-badge">🧠</span>'
    f'<span class="eyebrow">Skip-Gram × GRU Neural Network · Live Signal Explorer</span>'
    f'<span class="nerve-status" style="--glow-color: rgba({_live_info["glow"]},.8);">'
    f'<span class="nerve-dot"></span>{_live_info["emoji"]} {_live_info["name"]}</span></div>',
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
    st.markdown(
        '<div class="glass"><div class="glass-header"><span class="icon">💡</span>'
        '<span class="glass-title">Try an example</span></div>'
        '<div class="glass-sub">Tap any sample post below to load it into the analyzer.</div>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(4, gap="small")
    for col, key in zip(ex_cols, [1, 0, -1, -2]):
        ex_info = sentiment_info(key)
        with col:
            st.markdown(
                f'<div style="text-align:center;font-weight:700;color:{ex_info["color"]};'
                f'font-size:.85rem;margin-bottom:.4rem;">{ex_info["emoji"]} {ex_info["name"]}</div>',
                unsafe_allow_html=True,
            )
            for i, sample in enumerate(EXAMPLE_STATEMENTS[key]):
                short = sample if len(sample) <= 58 else sample[:55] + "…"
                if st.button(short, key=f"example_{key}_{i}", use_container_width=True):
                    st.session_state.analyzer_text_area = sample
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container(border=False):
        st.markdown('<div class="glass"><div class="glass-header"><span class="icon">💬</span>'
                    '<span class="eyebrow" style="font-size:.72rem;">Your text</span></div>', unsafe_allow_html=True)
        text = st.text_area(
            "Text to analyse",
            placeholder="Write or paste a message here… e.g. \"I've been feeling a lot lighter lately, things are looking up.\"",
            height=190,
            label_visibility="collapsed",
            key="analyzer_text_area",
        )
        btn_col1, btn_col2 = st.columns([3, 1], gap="small")
        with btn_col1:
            submitted = st.button("🔮  Analyse emotional signal", type="primary", use_container_width=True)
        with btn_col2:
            cleared = st.button("🌙  Clear", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if cleared:
        st.session_state.analyzer_text_area = ""
        st.session_state.last_result = None
        st.session_state.theme_color = RESTING_COLOR
        st.session_state.theme_glow = RESTING_GLOW
        st.rerun()

    if submitted:
        if not text.strip():
            st.warning("Add some text first, then analyse it. ✍️")
        else:
            with st.spinner("🧠⚡ Signal travelling down the axon…"):
                label, probabilities, classes, tokens = predict(text)
            info = sentiment_info(label)
            # Re-theme the whole nervous system to match the fired signal, then
            # rerun so every panel (buttons, neurons, synapses, headings) relights.
            st.session_state.theme_color = info["color"]
            st.session_state.theme_glow = info["glow"]
            st.session_state.last_result = {
                "text": text,
                "label": label,
                "probabilities": probabilities.tolist(),
                "classes": list(classes),
                "tokens": tokens,
            }
            st.rerun()

    result = st.session_state.last_result
    if result:
            label = result["label"]
            probabilities = np.array(result["probabilities"])
            classes = result["classes"]
            tokens = result["tokens"]
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass">
            <div class="glass-header"><span class="icon">📦</span><span class="glass-title">Batch analysis — upload a CSV</span></div>
            <div class="glass-sub">Upload a CSV with a text column and NeuroPulse will score every row's emotional intensity in one pass.</div>
        </div>''',
        unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader(
        "Upload CSV for batch prediction", type=["csv"], label_visibility="collapsed", key="batch_csv_uploader"
    )

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that CSV: {exc}")
            batch_df = None

        if batch_df is not None and not batch_df.empty:
            text_columns = [c for c in batch_df.columns if batch_df[c].dtype == object] or list(batch_df.columns)
            default_idx = 0
            for guess in ("text", "post", "message", "content", "input"):
                if guess in [c.lower() for c in text_columns]:
                    default_idx = [c.lower() for c in text_columns].index(guess)
                    break
            text_col = st.selectbox(
                "Which column holds the text to analyse?", options=text_columns, index=default_idx,
                key="batch_text_col",
            )
            run_batch = st.button("⚡ Run batch analysis", type="primary", key="run_batch_btn")

            if run_batch:
                rows = batch_df[text_col].astype(str).tolist()
                labels, moods, emojis, confidences = [], [], [], []
                progress = st.progress(0.0, text="🧠 Signals travelling through the network…")
                for i, row_text in enumerate(rows):
                    if row_text.strip():
                        label, probabilities, _, _ = predict(row_text)
                        info = sentiment_info(label)
                        labels.append(label)
                        moods.append(info["name"])
                        emojis.append(info["emoji"])
                        confidences.append(float(np.max(probabilities)))
                    else:
                        labels.append(None)
                        moods.append("—")
                        emojis.append("❔")
                        confidences.append(None)
                    progress.progress((i + 1) / max(len(rows), 1))
                progress.empty()

                results_df = batch_df.copy()
                results_df["predicted_intensity"] = labels
                results_df["predicted_sentiment"] = moods
                results_df["emoji"] = emojis
                results_df["confidence"] = confidences

                st.session_state.batch_results = results_df.to_csv(index=False)
                st.session_state.batch_results_display = results_df

            if "batch_results_display" in st.session_state:
                results_df = st.session_state.batch_results_display
                st.markdown("<br>", unsafe_allow_html=True)

                summary_counts = results_df["predicted_sentiment"].value_counts()
                summary_cols = st.columns(4, gap="small")
                for col, key in zip(summary_cols, [1, 0, -1, -2]):
                    info = sentiment_info(key)
                    n = int(summary_counts.get(info["name"], 0))
                    with col:
                        st.markdown(
                            f'''<div class="glass signal-card" style="padding:.9rem; --glow-color: rgba({info["glow"]},.6);">
                                <div style="font-size:1.6rem;">{info["emoji"]}</div>
                                <div style="font-weight:700;color:{info["color"]};margin:.15rem 0;font-size:.9rem;">{info["name"]}</div>
                                <div class="caption">{n} post{'s' if n != 1 else ''}</div>
                            </div>''',
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True, height=360)
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=st.session_state.batch_results,
                    file_name="neuropulse_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# TAB 2 — Dataset & EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.markdown(
        f'''<div class="glass">
            <div class="glass-header"><span class="icon">📡</span><span class="glass-title">Dataset snapshot</span></div>
            <div class="glass-sub">Mental-health forum posts, labelled by intensity from −2 (very negative) to 1 (positive).</div>
            <div class="chip-row" style="margin:.2rem 0 1rem;">
                <span class="chip">🌩️😞 Very Negative</span>
                <span class="chip">🌧️😕 Negative</span>
                <span class="chip">🌤️😐 Neutral</span>
                <span class="chip">🌈🙂 Positive</span>
            </div>
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