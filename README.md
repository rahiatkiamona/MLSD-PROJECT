# Online Shoppers Purchase Intent Dashboard

A calm Streamlit dashboard that visualizes the full MLSD project for the Online Shoppers Purchasing Intention dataset.

## What it shows
- Business goal and project stakeholders
- Raw data to versioned data pipeline
- Feature engineering and encoding choices
- Drift monitoring with PSI
- Saved artifacts and version metadata
- ML and serving pipeline roadmap

## Files
- `app.py` - main dashboard app
- `Data_preprocess.ipynb` - preprocessing notebook
- `artifacts/v1/` - saved pipeline outputs

## Run it
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. Open the local URL shown in the terminal.

## Notes
- The dashboard loads the real saved artifacts from `artifacts/v1/`.
- If you re-run the notebook and generate a new version folder, the app can be pointed to that version later.
