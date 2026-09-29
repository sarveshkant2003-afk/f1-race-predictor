# F1 Race Result Predictor

Predicts the full finishing order of a Formula 1 Grand Prix from information
available *before* the race (qualifying, grid, driver/team form, circuit
history, qualifying-session weather), using historical data pulled from the
[FastF1](https://docs.fastf1.dev/) API, four models compared on a strict
chronological split, and a Streamlit app to explore predictions.

**Live app:** [f1-race-predictor-sarvesh.streamlit.app](https://f1-race-predictor-sarvesh.streamlit.app/)
(no install needed; if it has been idle, it takes about 30 seconds to wake up)

## Setup (macOS / Linux)

```bash
# 1. Python 3.10+ (the project was built and tested on 3.12)
python3.12 -m venv venv          # or: python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. macOS only: XGBoost needs the OpenMP runtime
brew install libomp
```

## Run the pipeline

Every command is run from the repository root with the venv active.

```bash
# 1. Pull 2021-2025 qualifying / race / weather data from FastF1  (~10-15 min first time,
#    FastF1 caps itself at 500 API calls per hour; the script pauses and resumes automatically)
python src/data_collection.py
#    other seasons:  python src/data_collection.py --seasons 2024 2025 2026

# 2. Build the leakage-free feature table  ->  data/processed/features.parquet
python src/feature_engineering.py

# 3. Train baseline + Ridge + Random Forest + XGBoost, evaluate on the last season,
#    save models/*.pkl, models/metrics.json, models/feature_importance.csv
python src/train_model.py

# 4. Predict one race from the command line
python src/predict.py --season 2025 --round 5
python src/predict.py --season 2025 --round 5 --model ridge

# 5. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. Pick a model, a season and a race in
the sidebar. Past races show the predicted lineup next to the actual result
with per-race metrics; a race that has been qualified but not yet collected is
fetched live from FastF1.

## Results (held-out 2025 season, 24 races)

| Model | MAE | Spearman | Podium exact-set | Podium overlap | Exact position | Winner |
|---|---|---|---|---|---|---|
| Baseline (finish = grid) | 3.340 | 0.648 | 33 % | 75 % | 18 % | 67 % |
| **Ridge (best)** | **3.185** | **0.667** | **42 %** | **78 %** | 17 % | 54 % |
| Random Forest | 3.210 | 0.665 | 33 % | 75 % | **19 %** | 58 % |
| XGBoost | 3.252 | 0.665 | 21 % | 71 % | 16 % | 50 % |

Trained on 2021-2024 (1,790 driver-race rows); hyper-parameters chosen on 2024
only. The metric definitions live in `src/utils.py`.

## Layout

```
data/cache/         FastF1 HTTP cache (gitignored)
data/raw/           per-season CSVs written by data_collection.py
data/processed/     features.parquet / features.csv
src/utils.py        paths, team lineage, circuit metadata, metrics
src/data_collection.py
src/feature_engineering.py
src/train_model.py
src/predict.py
models/             trained pipelines + metrics.json + feature_importance.csv
app.py              Streamlit interface
```
