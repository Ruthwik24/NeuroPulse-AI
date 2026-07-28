# 🧠 NeuroPulse AI

**Read the emotion beneath the words.**

A futuristic, neural-powered Streamlit app that classifies the emotional **intensity** of mental-health-related text — from `🌈 Positive` to `🌩️ Very Negative` — using a Skip-Gram Word2Vec + GRU neural network, and lets you explore the full dataset, EDA, and a 12-model benchmark behind it.

🔗 **Live demo:** https://neuropulse-ai-12.streamlit.app/
📦 **Repository:** https://github.com/Ruthwik24/NeuroPulse-AI

> ⚠️ For education and demonstration only. This is not a diagnostic tool, crisis service, or substitute for a qualified mental-health professional.

---

## ✨ What it does

NeuroPulse AI is a Streamlit interface built on top of a saved Keras artifact (`artifacts/`) trained in the accompanying notebook. It's organized into five tabs:

| Tab | What's inside |
|---|---|
| 🔮 **Live Analyzer** | Paste any text and get a real-time predicted sentiment, confidence score, full class-probability breakdown, and the exact model-ready tokens. |
| 📡 **Dataset & EDA** | Dataset snapshot, class balance chart, post-length statistics, and the canonical preprocessing pipeline. |
| 🧬 **Model Zoo** | The 4 embeddings × 3 recurrent architectures that were trained, the shared network design, and the champion model's stats. |
| 🏆 **Leaderboard** | All 12 trained models ranked by F1 macro, plus the champion model's external validation report. |
| ℹ️ **About** | Dataset description, modeling methodology, and the end-to-end inference pipeline. |

All scores, bars, and cards are interactive (hover glow, animated fill-in, tooltips) inside a dark, aurora-gradient, glassmorphic UI.

---

## 🧾 About the dataset

- **Source:** Mental-health-related forum posts, each labelled with an **intensity** score reflecting the emotional tone of the writer.
- **Labels:** 4-point scale — `-2` (Very Negative) → `-1` (Negative) → `0` (Neutral) → `1` (Positive).
- **Size:** 10,392 raw rows → **10,391** after dropping 1 row with missing text/label (0 exact duplicates found).
- **Split:** 80/20 stratified train/test → **8,312** train / **2,079** test rows.
- **Class balance (imbalanced, real-world):**

  | Intensity | Sentiment | Count |
  |---|---|---|
  | `0` | 🌤️ Neutral | 4,375 |
  | `-1` | 🌧️ Negative | 4,112 |
  | `-2` | 🌩️ Very Negative | 1,155 |
  | `1` | 🌈 Positive | 750 |

- **Post length:** Highly variable — average **234 words** / **1,178 characters** per post (median 162 words / 787 characters), ranging up to 5,413 words in the longest post.
- **Imbalance handling:** `RandomOverSampler` applied **only to the training split** (safe for text — duplicates real rows rather than interpolating between token IDs like SMOTE would); the test split stays untouched and imbalanced for a realistic evaluation.

### Preprocessing pipeline (canonical, used identically in training and inference)

```
Raw text → Lowercase → Strip URLs → Strip HTML tags → Remove punctuation
→ Tokenize → Drop URL-fragment tokens → Remove stopwords (keep negations)
→ Lemmatize (verb form)
```

Negation words (`not`, `never`, `don't`, etc.) are deliberately **kept** in the stopword removal step, since they flip sentiment meaning.

---

## 🤖 About the models

Twelve neural classifiers were trained — every pairing of **4 word-embedding strategies** with **3 recurrent architectures**:

**Embeddings**
- 🎯 Skip-Gram (Word2Vec) — predicts context from a target word; strong on rarer terms
- 🧩 Word2Vec (CBOW) — predicts a target word from context; fast, smooth on frequent terms
- 🧵 FastText — adds subword n-grams, robust to typos/unseen word forms
- 🌐 GloVe (pretrained) — global co-occurrence vectors from a large external corpus

**Recurrent heads**
- 🔁 LSTM — gated memory cell, strong long-range recall
- ⚡ GRU — lighter than LSTM, faster to train, fewer parameters
- ↔️ BiLSTM — reads text forward and backward simultaneously

**Shared architecture:**

```
Embedding layer → RNN (128 units, dropout 0.2) → Dense 64 (ReLU)
→ Dropout 0.5 → Dense · Softmax (4 classes)
```

All 12 models were trained under **identical conditions** — same padding (`MAXLEN`), same train-only oversampling, same `Adam` optimizer with sparse categorical cross-entropy, `EarlyStopping` (on `val_loss`, restoring best-epoch weights), and `ReduceLROnPlateau` — making the comparison a fair, apples-to-apples benchmark.

### 🏆 Model comparison (ranked by F1 macro, held-out test split)

| Rank | Model | Accuracy | F1 (macro) | F1 (weighted) | ROC-AUC (macro OvR) | Epochs |
|---|---|---|---|---|---|---|
| 🥇 | **Skip-Gram + GRU** | 69.65% | **0.6396** | 0.7045 | 0.8935 | 11 |
| 🥈 | Word2Vec + BiLSTM | 67.15% | 0.6290 | 0.6803 | 0.8672 | 6 |
| 🥉 | FastText + GRU | 67.34% | 0.6282 | 0.6822 | 0.8776 | 8 |
| 4 | Word2Vec + GRU | 67.34% | 0.6273 | 0.6794 | 0.8827 | 7 |
| 5 | GloVe + LSTM | 68.11% | 0.6256 | 0.6841 | 0.8761 | 8 |
| 6 | Skip-Gram + BiLSTM | 66.38% | 0.6235 | 0.6667 | 0.8823 | 7 |
| 7 | GloVe + GRU | 66.71% | 0.6220 | 0.6733 | 0.8784 | 7 |
| 8 | Word2Vec + LSTM | 66.43% | 0.6217 | 0.6748 | 0.8668 | 8 |
| 9 | Skip-Gram + LSTM | 67.63% | 0.6204 | 0.6825 | 0.8776 | 7 |
| 10 | GloVe + BiLSTM | 66.14% | 0.6150 | 0.6681 | 0.8678 | 7 |
| 11 | FastText + LSTM | 66.33% | 0.6099 | 0.6698 | 0.8657 | 9 |
| 12 | FastText + BiLSTM | 63.93% | 0.5979 | 0.6539 | 0.8652 | 6 |

**Champion: Skip-Gram + GRU** — selected as best on F1 macro over the held-out, imbalanced, real-world test split. This is the exact artifact powering the Live Analyzer tab.

### 🧪 External validation

The champion model was additionally evaluated on a **fully separate, never-seen** `test_data.xlsx` file (10 posts per class, 40 total) and achieved **90% accuracy**:

| Intensity | Sentiment | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| `-2` | 🌩️ Very Negative | 0.83 | 1.00 | 0.91 | 10 |
| `-1` | 🌧️ Negative | 0.88 | 0.70 | 0.78 | 10 |
| `0` | 🌤️ Neutral | 1.00 | 1.00 | 1.00 | 10 |
| `1` | 🌈 Positive | 0.90 | 0.90 | 0.90 | 10 |

---

## ⚙️ Inference pipeline

What happens when you hit "Analyse":

```
Raw text → clean_text() → Tokenizer → sequence → Pad to MAXLEN
→ Skip-Gram + GRU model → Softmax probabilities → LabelEncoder → Intensity label
```

The **exact** tokenizer, label encoder, and `MAXLEN` saved during training are reloaded at inference time (never re-fit), so predictions match training behaviour precisely — this was the core train/inference-mismatch bug the notebook fixes.

---

## 🗂️ Project structure

```
.
├── app.py                 # Streamlit app (NeuroPulse AI interface)
├── NLP_MODIFIED.ipynb     # Training notebook: EDA → preprocessing → embeddings
│                           # → 12-model training/comparison → artifact export
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version pin for deployment (3.11)
├── README.md               # This file
└── artifacts/               # Saved model + preprocessing objects (generated
    ├── config.json          # by the notebook — not committed by default;
    ├── tokenizer.pkl         # add your own after running the notebook)
    ├── label_encoder.pkl
    └── skipgram_gru.keras    # (or whichever model is named `best_model`)
```

> **Note:** The `artifacts/` folder is produced by running the notebook end-to-end (see `## 12. Save Everything Needed For Inference` in `NLP_MODIFIED.ipynb`). It must be present alongside `app.py` for the app to run.

---

## 🚀 Run locally

TensorFlow does not currently provide Windows wheels for Python 3.14. Install Python 3.11 first, then create the environment with that interpreter:

```powershell
winget install -e --id Python.Python.3.11
# Close and reopen PowerShell after the installation, then return to this folder.
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

---

## ☁️ Deploy

1. Commit this repository with the `artifacts/` directory included.
2. Create a [Streamlit Community Cloud](https://streamlit.io/cloud) app whose entry point is `app.py`.
3. The included `runtime.txt` requests Python 3.11 automatically.
4. Once deployed, update the **Live demo** link at the top of this README.

---

## 🧰 Tech stack

- **Streamlit** — UI & app framework
- **TensorFlow / Keras** — GRU neural network
- **NLTK** — tokenization, stopwords, lemmatization
- **scikit-learn** — label encoding, train/test split, evaluation metrics
- **gensim** — Word2Vec (Skip-Gram/CBOW) & FastText embeddings *(training notebook only)*
- **GloVe** — pretrained embeddings *(training notebook only)*
- **imbalanced-learn** — `RandomOverSampler` for train-only class balancing *(training notebook only)*

---

## ⚠️ Disclaimer

This tool is for **education and demonstration only**. Its predictions are not a medical assessment, diagnosis, or crisis response. If you or someone you know is struggling, please reach out to a licensed mental-health professional or a local crisis line.
