# MindSignal

A Streamlit interface for the saved **Skip-Gram + GRU** mental-health text classifier in `artifacts/`.

## Run locally

TensorFlow does not currently provide Windows wheels for Python 3.14. Install
Python 3.11 first, then create the environment with that interpreter:

```powershell
winget install -e --id Python.Python.3.11
# Close and reopen PowerShell after the installation, then return to this folder.
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

## Deploy

Commit this repository with the `artifacts/` directory included, then create a Streamlit Community Cloud app whose entry point is `app.py`. In Community Cloud's **Advanced settings**, select Python 3.10 to match the TensorFlow environment used locally.

> This demo is for education only. Its output is not a medical assessment or diagnosis.
