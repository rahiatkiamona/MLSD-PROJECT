# Deployment Guide

This project is built as a Streamlit website. The easiest deployment path is Streamlit Community Cloud.

## 1. Push the code to GitHub
The repository is already connected to:

`https://github.com/rahiatkiamona/MLSD-PROJECT`

Make sure these files are in the repo:
- `app.py`
- `dashboard_site.py`
- `requirements.txt`
- `README.md`
- `artifacts/v1/`

## 2. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub.
3. Choose your repository: `rahiatkiamona/MLSD-PROJECT`.
4. Set the main file path to `app.py`.
5. Click Deploy.

## 3. What the online app will do
- Show the project dashboard
- Let users explore the data
- Provide a live purchase-intent prediction studio
- Show drift monitoring and saved artifacts
- Explain the full ML pipeline visually

## 4. Local run
```bash
streamlit run app.py
```

## 5. If you want a custom deployment later
You can also deploy the same app on Render, Railway, or Docker-based hosting. The repo already has the Streamlit entry point, so the app can be moved with minimal changes.
