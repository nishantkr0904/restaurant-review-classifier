# Restaurant Review Classifier

A FastAPI web app that predicts restaurant review sentiment (Liked / Not Liked) and highlights aspect-level sentiment (food, service, ambience, price, etc.) using a TF‑IDF + Bernoulli Naive Bayes model.

## Features

- Overall sentiment prediction with confidence
- Aspect extraction + per-sentence sentiment scoring
- Web UI (Jinja2 templates) + JSON API endpoint

## Tech Stack

- FastAPI + Uvicorn
- scikit-learn (TF‑IDF, BernoulliNB)
- NLTK (stopwords, stemming)
- Jinja2 templates + CSS UI

## Project Structure

- app/main.py — FastAPI app + routes
- app/text_utils.py — text preprocessing
- app/templates/index.html — UI template
- app/static/style.css — styling
- train_model.py — training pipeline (generates artifacts)
- model.pkl / vectorizer.pkl — saved model + vectorizer
- Restaurant_Reviews.tsv — dataset

## Setup (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Web App

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

## API Usage (Optional)

Run this **after the server is running** to test the JSON API:

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
	-H "Content-Type: application/json" \
	-d '{"review":"The food was great but the service was slow."}'
```

## Train / Rebuild the Model Artifacts (Optional)

Run this **only if you want to retrain** or if `model.pkl` / `vectorizer.pkl` are missing:

```bash
python train_model.py
```
