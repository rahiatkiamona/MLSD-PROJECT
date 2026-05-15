# Online Shoppers Purchase Intent System

A custom ML web system (FastAPI + HTML/CSS/JS) for the Online Shoppers Purchasing Intention dataset.

## What it includes
- Business goal and project stakeholders
- Raw data to versioned data pipeline
- Feature engineering and encoding choices
- Drift monitoring with PSI
- Live serving API and prediction form
- Batch scoring API endpoint
- Deployment-ready backend architecture

## Files
- `app.py` - FastAPI application entrypoint
- `ml_system/service.py` - ML data/model service layer
- `templates/index.html` - custom web UI
- `static/style.css` - custom styling
- `Data_preprocess.ipynb` - preprocessing notebook
- `artifacts/v*/` - versioned saved outputs

## Run it
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the system:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Open: `http://localhost:8000`

## Notes
- The app uses the latest artifact version automatically and supports switching versions.
- API endpoints are available for serving and integration:
	- `POST /api/predict`
	- `POST /api/predict-batch`
	- `POST /api/version/{version}`
	- `GET /health`
