# Exoplanet Transit Detection Pipeline

Detects exoplanetary transits from TESS photometric data using BLS algorithm + XGBoost classifier.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Stack
- lightkurve (TESS data)
- astropy (BLS, constants)
- XGBoost (transit classifier)
- Streamlit (frontend)