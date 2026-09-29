"""
Load a trained model and produce a predicted finishing lineup for one race.

Usage
-----
    python src/predict.py --season 2025 --round 5
    python src/predict.py --season 2025 --round 5 --model random_forest

Two paths are supported:
  * the race is already in ``data/processed/features.parquet``  ->  instant
  * it is not (e.g. a race from the current season that has not been collected
    yet)  ->  the qualifying session (and the race session, if it has already
    happened) is fetched from FastF1 and features are built on the fly on top
    of the stored history.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import FEATURE_COLUMNS, TARGET, build_feature_table, load_raw  # noqa: E402
from utils import (  # noqa: E402
    FEATURES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    add_predicted_rank,
    compute_metrics,
    enable_fastf1_cache,
    is_street_circuit,
)

LINEUP_COLUMNS = [
    "predicted_position", "driver", "driver_name", "team", "grid_position", "quali_position",
    "predicted_score", "finish_position", "status", "dnf",
]


def load_metrics() -> dict:
    """The JSON report written by ``train_model.py``."""
    return json.loads(METRICS_PATH.read_text())


def load_model(name: str | None = None):
    """
    Load a fitted pipeline from ``models/``.  With no name, the best model
    recorded in ``metrics.json`` is used.
    """
    if name is None:
        name = load_metrics()["best_model"]
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run train_model.py first")
    return joblib.load(path)


def load_features() -> pd.DataFrame:
    """The processed feature table produced by ``feature_engineering.py``."""
    return pd.read_parquet(FEATURES_PATH)


def predict_lineup(race_rows: pd.DataFrame, model) -> pd.DataFrame:
    """
    Score every driver of *one* race and sort them into a lineup.

    ``race_rows`` must contain ``FEATURE_COLUMNS`` (and season/round/driver).
    The raw regression output is kept as ``predicted_score`` – it is a
    continuous "expected finishing position" – and ranked into the integer
    ``predicted_position`` that forms the lineup.
    """
    out = race_rows.copy()
    out["predicted_score"] = model.predict(out[FEATURE_COLUMNS])
    out = add_predicted_rank(out, "predicted_score", "predicted_position")
    cols = [c for c in LINEUP_COLUMNS if c in out.columns]
    return out.sort_values("predicted_position")[cols].reset_index(drop=True)


def race_metrics(lineup: pd.DataFrame, season: int, rnd: int) -> dict | None:
    """Metrics for a single already-run race, or None when no result exists."""
    if TARGET not in lineup or lineup[TARGET].isna().all():
        return None
    d = lineup.assign(season=season, round=rnd)
    return compute_metrics(d, "predicted_score", TARGET)


# --------------------------------------------------------------------------- #
# Races that are not in the processed table yet
# --------------------------------------------------------------------------- #
def fetch_new_race_raw(season: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pull the qualifying session (and the race session when it exists) for a
    round that ``data_collection.py`` has not stored, and return the four raw
    frames (events, qualifying, races, weather) in exactly the format the
    collector writes - so they can be appended to the stored history.

    If the race has not been run yet, the "race" rows contain the grid
    (= qualifying order, since penalties are unknown) and NaN results.
    """
    import fastf1
    from data_collection import load_session, qualifying_rows, race_rows, summarise_weather

    enable_fastf1_cache()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    ev = schedule[schedule["RoundNumber"] == rnd]
    if ev.empty:
        raise ValueError(f"Round {rnd} does not exist in the {season} schedule")
    ev = ev.iloc[0]
    events = pd.DataFrame(
        [{
            "season": season, "round": rnd, "event_name": ev["EventName"], "location": ev["Location"],
            "country": ev["Country"], "event_date": str(pd.to_datetime(ev["EventDate"]).date()),
            "event_format": ev["EventFormat"],
            "is_sprint_weekend": int("sprint" in str(ev["EventFormat"]).lower()),
            "is_street_circuit": is_street_circuit(ev["Location"]),
        }]
    )

    q = load_session(season, rnd, "Q")
    if q.results is None or len(q.results) == 0 or q.results["Position"].isna().all():
        raise ValueError(f"Qualifying for {season} round {rnd} has not taken place yet")
    quali = pd.DataFrame(qualifying_rows(q, season, rnd))
    weather = [summarise_weather(q, season, rnd, "Q")]

    try:
        r = load_session(season, rnd, "R")
        has_race = r.results is not None and len(r.results) > 0 and r.results["Position"].notna().any()
    except Exception:
        has_race = False
    if has_race:
        races = pd.DataFrame(race_rows(r, season, rnd))
        weather.append(summarise_weather(r, season, rnd, "R"))
    else:
        races = pd.DataFrame(
            {
                "season": season, "round": rnd, "driver_number": quali["driver_number"],
                "driver": quali["driver"], "driver_name": quali["driver_name"], "team": quali["team"],
                "grid_position": quali["quali_position"], "finish_position": float("nan"),
                "classified_position": None, "status": None, "points": float("nan"), "laps_completed": float("nan"),
            }
        )
    return events, quali, races, pd.DataFrame(weather)


def build_features_for_new_race(season: int, rnd: int) -> pd.DataFrame:
    """
    Append the freshly fetched round to the stored raw history, rebuild the
    whole feature table (so rolling features see the real past) and return only
    the rows of the requested race.
    """
    events, quali, races, weather = load_raw()
    n_events, n_quali, n_races, n_weather = fetch_new_race_raw(season, rnd)
    # If the round was already stored, drop the stored copy first.
    mask = lambda d: ~((d["season"] == season) & (d["round"] == rnd))  # noqa: E731
    events, quali, races, weather = events[mask(events)], quali[mask(quali)], races[mask(races)], weather[mask(weather)]
    full = build_feature_table(
        pd.concat([events, n_events], ignore_index=True),
        pd.concat([quali, n_quali], ignore_index=True),
        pd.concat([races, n_races], ignore_index=True),
        pd.concat([weather, n_weather], ignore_index=True),
    )
    return full[(full["season"] == season) & (full["round"] == rnd)].reset_index(drop=True)


def get_race_rows(season: int, rnd: int, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows for one race - from the processed table if present, else built live."""
    if features is None:
        features = load_features()
    rows = features[(features["season"] == season) & (features["round"] == rnd)]
    if len(rows):
        return rows.reset_index(drop=True)
    return build_features_for_new_race(season, rnd)


def predict_race(season: int, rnd: int, model_name: str | None = None, features: pd.DataFrame | None = None):
    """Convenience wrapper: rows -> model -> lineup (+ metrics if the race has been run)."""
    rows = get_race_rows(season, rnd, features)
    lineup = predict_lineup(rows, load_model(model_name))
    return lineup, race_metrics(lineup, season, rnd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model", default=None, help="ridge | random_forest | xgboost (default: best)")
    args = parser.parse_args()

    lineup, metrics = predict_race(args.season, args.round, args.model)
    pd.set_option("display.width", 200)
    print(lineup.to_string(index=False))
    if metrics:
        print("\nRace metrics:", json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
