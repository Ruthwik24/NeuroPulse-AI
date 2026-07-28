# """MindSignal — Streamlit inference app for the Skip-Gram + GRU artifact."""

# from __future__ import annotations

# import json
# import pickle
# import re
# from pathlib import Path

# import nltk
# import numpy as np
# import streamlit as st
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# st.set_page_config(
#     page_title="MindSignal | Text Insight",
#     page_icon="✦",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# ROOT = Path(__file__).resolve().parent
# ARTIFACTS = ROOT / "artifacts"


# def ensure_nltk_data() -> None:
#     """Install the small tokenizer resources once when a new host starts."""
#     for package, resource in (
#         ("punkt", "tokenizers/punkt"),
#         ("punkt_tab", "tokenizers/punkt_tab"),
#         ("stopwords", "corpora/stopwords"),
#         ("wordnet", "corpora/wordnet"),
#     ):
#         try:
#             nltk.data.find(resource)
#         except LookupError:
#             nltk.download(package, quiet=True)


# @st.cache_resource(show_spinner="Initializing the intelligence layer…")
# def load_artifacts():
#     """Load precisely the saved model and preprocessing objects used in training."""
#     with (ARTIFACTS / "config.json").open(encoding="utf-8") as file:
#         config = json.load(file)
#     with (ARTIFACTS / "tokenizer.pkl").open("rb") as file:
#         tokenizer = pickle.load(file)
#     with (ARTIFACTS / "label_encoder.pkl").open("rb") as file:
#         encoder = pickle.load(file)

#     # Convert the notebook's model name ("Skip-Gram + GRU") to its saved filename.
#     model_file = config["best_model"].lower().replace(" ", "_").replace("+", "") + ".keras"
#     model = load_model(ARTIFACTS / model_file, compile=False)
#     return model, tokenizer, encoder, config


# NEGATIONS = {
#     "not", "no", "nor", "never", "don't", "didn't", "doesn't", "can't",
#     "won't", "isn't", "aren't", "wasn't", "weren't", "couldn't", "shouldn't",
#     "wouldn't", "haven't", "hasn't", "hadn't",
# }
# URL_FRAGMENT_TOKENS = {
#     "http", "https", "www", "com", "org", "net", "gov", "edu", "html", "htm",
#     "php", "aspx", "asp", "co", "uk", "jpg", "png", "gif", "pdf", "id",
# }
# URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)")
# HTML_TAG_REGEX = re.compile(r"<.*?>")
# NON_WORD_REGEX = re.compile(r"[^\w\s']")
# MULTI_SPACE_REGEX = re.compile(r"\s+")


# @st.cache_resource
# def preprocessing_tools():
#     ensure_nltk_data()
#     return WordNetLemmatizer(), set(stopwords.words("english")) - NEGATIONS


# def clean_text(text: str) -> list[str]:
#     """Canonical preprocessing copied from the training notebook."""
#     lemmatizer, active_stopwords = preprocessing_tools()
#     text = str(text).lower()
#     text = URL_REGEX.sub(" ", text)
#     text = HTML_TAG_REGEX.sub(" ", text)
#     text = NON_WORD_REGEX.sub(" ", text)
#     text = MULTI_SPACE_REGEX.sub(" ", text).strip()
#     tokens = word_tokenize(text)
#     tokens = [token for token in tokens if token not in URL_FRAGMENT_TOKENS]
#     tokens = [token for token in tokens if token not in active_stopwords]
#     return [lemmatizer.lemmatize(token, pos="v") for token in tokens]


# def predict(text: str):
#     model, tokenizer, encoder, config = load_artifacts()
#     tokens = clean_text(text)
#     sequence = tokenizer.texts_to_sequences([tokens])
#     padded = pad_sequences(sequence, maxlen=int(config["MAXLEN"]))
#     probabilities = model.predict(padded, verbose=0)[0]
#     index = int(np.argmax(probabilities))
#     return encoder.inverse_transform([index])[0], probabilities, encoder.classes_, tokens


# def level_name(label) -> str:
#     return f"Intensity level {label}"


# st.markdown(
#     """
#     <style>
#       .stApp { background: radial-gradient(circle at 12% 4%, #183764 0, transparent 28%),
#                radial-gradient(circle at 88% 8%, #3c1d6f 0, transparent 25%), #07111f; color: #eef5ff; }
#       [data-testid="stHeader"] { background: transparent; }
#       .block-container { max-width: 1160px; padding-top: 3.6rem; padding-bottom: 4rem; }
#       .eyebrow { color: #6ee7e1; font-size: .76rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
#       h1 { font-size: clamp(2.4rem, 6vw, 4.8rem) !important; line-height: .98 !important; margin: .42rem 0 1rem !important; letter-spacing: -.05em; }
#       .lead { color: #a9bad0; font-size: 1.08rem; max-width: 670px; line-height: 1.65; }
#       .glass { background: linear-gradient(135deg, rgba(20,39,65,.86), rgba(19,20,47,.78)); border: 1px solid rgba(123,224,240,.22); border-radius: 24px; padding: 1.35rem; box-shadow: 0 18px 60px rgba(0,0,0,.2); }
#       .signal { color: #aef8ef; font-size: 2rem; font-weight: 750; letter-spacing: -.045em; margin: .1rem 0; }
#       .caption { color: #9dafc7; font-size: .84rem; }
#       .stTextArea textarea { background: rgba(5,15,29,.8) !important; color: #eff8ff !important; border: 1px solid rgba(105,218,231,.35) !important; border-radius: 16px !important; font-size: 1.02rem !important; }
#       .stButton > button { width: 100%; border: 0; border-radius: 14px; padding: .75rem; font-weight: 750; color: #05111b; background: linear-gradient(90deg, #65e6dc, #8fa9ff); }
#       [data-testid="stExpander"] { background: rgba(15,28,48,.55); border: 1px solid rgba(123,224,240,.16); border-radius: 16px; }
#       .stProgress > div > div > div > div { background: linear-gradient(90deg, #65e6dc, #8fa9ff); }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown('<div class="eyebrow">Skip-Gram × GRU · Signal Explorer</div>', unsafe_allow_html=True)
# st.title("Find the signal\nin your words.")
# st.markdown(
#     '<p class="lead">An interactive demonstration of a neural text classifier trained on mental-health post intensity. Paste a short piece of text to view the model’s predicted class distribution.</p>',
#     unsafe_allow_html=True,
# )

# with st.container(border=False):
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     text = st.text_area(
#         "Text to analyse",
#         placeholder="Write or paste a message here…",
#         height=190,
#         label_visibility="collapsed",
#     )
#     submitted = st.button("Analyse signal  ✦", type="primary")
#     st.markdown('</div>', unsafe_allow_html=True)

# if submitted:
#     if not text.strip():
#         st.warning("Add some text first, then analyse it.")
#     else:
#         label, probabilities, classes, tokens = predict(text)
#         confidence = float(np.max(probabilities))

#         st.markdown("<br>", unsafe_allow_html=True)
#         first, second = st.columns([1, 1.55], gap="large")
#         with first:
#             st.markdown('<div class="glass">', unsafe_allow_html=True)
#             st.markdown('<div class="eyebrow">Predicted class</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="signal">{level_name(label)}</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="caption">Model confidence · {confidence:.1%}</div>', unsafe_allow_html=True)
#             st.progress(confidence)
#             st.markdown('</div>', unsafe_allow_html=True)
#         with second:
#             st.markdown('<div class="glass">', unsafe_allow_html=True)
#             st.markdown('<div class="eyebrow">Class distribution</div>', unsafe_allow_html=True)
#             for class_label, probability in zip(classes, probabilities):
#                 st.markdown(f"<div class='caption'>{level_name(class_label)} · {float(probability):.1%}</div>", unsafe_allow_html=True)
#                 st.progress(float(probability))
#             st.markdown('</div>', unsafe_allow_html=True)

#         with st.expander("See the model-ready text"):
#             st.caption("The tokens below follow the preprocessing pipeline used in the notebook.")
#             st.code(" ".join(tokens) if tokens else "No recognised tokens", language=None)

# st.markdown("<br><div class='caption'>For education and demonstration only — this tool is not a diagnosis, crisis service, or substitute for a qualified mental-health professional.</div>", unsafe_allow_html=True)


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


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      .stApp {
        background:
          radial-gradient(circle at 10% 0%, rgba(99,102,241,.28) 0, transparent 32%),
          radial-gradient(circle at 90% 10%, rgba(236,72,153,.22) 0, transparent 30%),
          radial-gradient(circle at 50% 100%, rgba(45,212,191,.18) 0, transparent 40%),
          linear-gradient(160deg, #050914 0%, #0a0f24 45%, #0c0a1f 100%);
        color: #eef5ff;
        background-attachment: fixed;
      }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }

      @keyframes floaty { 0%,100% { transform: translateY(0px);} 50% { transform: translateY(-6px);} }
      @keyframes pulseGlow { 0%,100% { opacity:.55; } 50% { opacity:1; } }
      @keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }

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
      .lead { color: #aab8d4; font-size: 1.06rem; max-width: 700px; line-height: 1.7; }
      .lead .hi { color: #6ee7e1; font-weight: 600; }

      .chip-row { display:flex; gap:.5rem; flex-wrap:wrap; margin: 1rem 0 1.6rem; }
      .chip {
        display:inline-flex; align-items:center; gap:.4rem;
        background: rgba(139,92,246,.12); border: 1px solid rgba(167,139,250,.35);
        color:#d8d3ff; font-size:.8rem; font-weight:600; padding:.35rem .8rem; border-radius:999px;
      }

      .glass {
        background: linear-gradient(145deg, rgba(30,25,60,.55), rgba(10,14,32,.75));
        border: 1px solid rgba(167,139,250,.25);
        border-radius: 26px; padding: 1.5rem;
        box-shadow: 0 20px 70px rgba(76,29,149,.18), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(14px);
      }
      .glass-header { display:flex; align-items:center; gap:.5rem; margin-bottom:.9rem; }
      .glass-header .icon { font-size:1.2rem; }

      .signal-card {
        text-align:center; padding: 1.2rem 1rem 1.5rem;
      }
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
      .dist-pct { flex: 0 0 48px; text-align:right; font-size:.82rem; color:#9dafc7; font-variant-numeric: tabular-nums; }
      .dist-track { flex:1; height:9px; background: rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
      .dist-fill { height:100%; border-radius:999px; box-shadow: 0 0 12px 0 var(--glow, rgba(110,231,225,.6)); }

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
      .stButton > button:hover { background-position: right center; transform: translateY(-1px); box-shadow: 0 10px 36px rgba(139,92,246,.5); }

      [data-testid="stExpander"] { background: rgba(15,18,40,.55); border: 1px solid rgba(139,92,246,.22); border-radius: 18px; }

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
    'Paste a short message and watch the model surface its emotional intensity — from '
    '<span class="hi">🌈 positive</span> to <span class="hi">🌩️ very negative</span> — in real time.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="chip-row">'
    '<span class="chip">🌈 Positive</span>'
    '<span class="chip">🌤️ Neutral</span>'
    '<span class="chip">🌧️ Negative</span>'
    '<span class="chip">🌩️ Very Negative</span>'
    '<span class="chip">✨ Real-time inference</span>'
    '</div>',
    unsafe_allow_html=True,
)

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
                    <span class="dist-track">
                        <span class="dist-fill" style="width:{pct:.1f}%; background:{c_info["color"]}; --glow: rgba({c_info["glow"]},.6);"></span>
                    </span>
                    <span class="dist-pct">{pct:.1f}%</span>
                </div>'''
            rows_html += '</div>'
            st.markdown(rows_html, unsafe_allow_html=True)

        with st.expander("🔬 See the model-ready text"):
            st.caption("The tokens below follow the preprocessing pipeline used in the notebook.")
            st.code(" ".join(tokens) if tokens else "No recognised tokens", language=None)

st.markdown(
    "<div class='footer-note'><span class='icon'>💙</span>"
    "<span>For education and demonstration only — this tool is not a diagnosis, crisis service, "
    "or substitute for a qualified mental-health professional. If you or someone you know is struggling, "
    "please reach out to a licensed professional or a local crisis line.</span></div>",
    unsafe_allow_html=True,
)