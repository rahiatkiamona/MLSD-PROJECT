# Deployment Guide

This project is built as a custom FastAPI machine learning system (no Streamlit).

## 1. Push the code to GitHub
The repository is already connected to:

`https://github.com/rahiatkiamona/MLSD-PROJECT`

Make sure these files are in the repo:
- `app.py`
- `ml_system/service.py`
- `templates/index.html`
- `static/style.css`
- `requirements.txt`
- `README.md`
- `artifacts/v*/`
- `Dockerfile`
- `render.yaml`

## 2. Deploy on Render (recommended)
1. Go to https://render.com/ and connect GitHub.
2. Create a new **Web Service** from `rahiatkiamona/MLSD-PROJECT`.
3. Render will detect `render.yaml` / `Dockerfile`.
4. Deploy and open the generated public URL.

Alternative platforms:
- Railway
- Fly.io
- Any Docker-compatible cloud

## 3. What the online app will do
- Show the project dashboard
- Let users explore the data
- Provide a live purchase-intent prediction studio
- Show drift monitoring and saved artifacts
- Explain the full ML pipeline visually

## 4. Local run
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5. API endpoints
- `GET /health`
- `GET /api/versions`
- `POST /api/version/{version}`
- `POST /api/predict`
- `POST /api/predict-batch`
