# Smart Internship Performance Analytics System

A clean, interactive Streamlit dashboard for internship placement analytics, student performance insights, and placement readiness prediction.

## Project overview

This repository delivers a practical internship analytics prototype with:

- A Streamlit dashboard in `dashboard/app.py`
- A cleaned placement dataset in `data/combined_dataset_FINAL_CLEAN.xlsx`
- The model feature configuration in `models/feature_columns.pkl`
- A Power BI report in `powerbi/`
- A retraining script at `train_model.py`

## Repository structure

```
VSC project beau roi/
├─ dashboard/
│  └─ app.py
├─ data/
│  └─ combined_dataset_FINAL_CLEAN.xlsx
├─ models/
│  ├─ feature_columns.pkl
├─ powerbi/
│  └─ smart_internship_analytics_Dashboard.pbix(1).pbix
├─ train_model.py
└─ README.md
```

## Run the dashboard

1. Open PowerShell in the project root.
2. Activate your virtual environment, for example:

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

3. Install dependencies if needed:

```powershell
python -m pip install streamlit pandas plotly scikit-learn openpyxl
```

4. Start the app:

```powershell
python -m streamlit run dashboard/app.py
```

5. Open the dashboard in your browser:

```text
http://localhost:8501
```

## Train or regenerate the model

If `models/placement_model.pkl` or `models/feature_columns.pkl` are missing, the Streamlit app now attempts to train the model automatically from `data/combined_dataset_FINAL_CLEAN.xlsx`.

If automatic generation fails, run `train_model.py` locally to retrain the placement prediction model and regenerate the feature list:

```powershell
python train_model.py
```

This script saves:

- `models/placement_model.pkl`
- `models/feature_columns.pkl`

## Important note

`models/placement_model.pkl` is intentionally kept out of GitHub because it is a large binary file. The repository includes the model training script and feature metadata so the model can be rebuilt locally.

## Why this repo is useful

- Interactive internship analytics for placement trends
- Student-level placement prediction
- Skill-gap evaluation and recommendation support
- Easy local setup and retraining workflow

---

Created for internship analytics and placement performance reporting with Streamlit and machine learning.
