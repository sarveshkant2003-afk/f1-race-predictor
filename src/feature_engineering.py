"""
Turn the raw per-session CSVs in ``data/raw/`` into one model-ready table with
one row per (driver, race) and only *pre-race* information as columns.

Usage
-----
    python src/feature_engineering.py

Output
------
    data/processed/features.parquet   (and features.csv for easy inspection)

Leakage policy
--------------
A row describing driver D at race N may only contain information that existed
before race N started:
  * qualifying + grid data for race N (known on Saturday / before lights-out)
  * results, points, DNFs from races < N  (always shifted by one race)
  * qualifying-session weather for race N (the race-session weather is stored
    for reporting but deliberately *not* used as a feature, because rain during
    the race is not known before the race)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    FEATURES_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    canonical_team,
    classified_to_dnf,
    setup_logging,
)

log = setup_logging("feature_engineering")

TARGET = "finish_position"

# The exact columns fed to every model, in this order.
FEATURE_COLUMNS = [
    # --- this weekend's qualifying / grid ---------------------------------
    "quali_position",
    "quali_gap_to_pole_pct",
    "q2_reached",
    "q3_reached",
    "grid_position",
    "grid_penalty",
    # --- driver form (all from *previous* races) --------------------------
    "driver_avg_finish_last3",
    "driver_avg_finish_last5",
    "driver_avg_quali_last5",
    "driver_dnf_rate_last10",
    "driver_season_points",
    "driver_season_rank",
    "driver_experience",
    # --- team form (previous races) ---------------------------------------
    "team_avg_finish_last5",
    "team_season_points",
    "team_season_rank",
    # --- driver history at this circuit (previous visits) -----------------
    "driver_circuit_avg_finish",
    "driver_circuit_visits",
    # --- event context -----------------------------------------------------
    "is_street_circuit",
    "is_sprint_weekend",
    # --- conditions known before the race (qualifying weather) -------------
    "quali_is_wet",
    "quali_rain_fraction",
    "quali_air_temp",
    "quali_track_temp",
    "quali_humidity",
    "quali_wind_speed",
]

# Columns carried through for display / evaluation but never used as inputs.
META_COLUMNS = [
    "season", "round", "event_name", "location", "country", "event_date",
    "driver", "driver_name", "driver_number", "team", "team_key",
    "finish_position", "classified_position", "status", "points", "dnf",
    "race_is_wet", "race_rain_fraction",
]


# --------------------------------------------------------------------------- #
# 1. Load raw files
# --------------------------------------------------------------------------- #
def _read_all(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / f"{prefix}_*.csv")))
    if not files:
        raise FileNotFoundError(f"No {prefix}_<year>.csv files in {RAW_DIR}; run data_collection.py first")
    # classified_position mixes digits and letter codes ('R', 'D'); driver
    # numbers are labels, not quantities - keep both as strings.
    dtypes = {"classified_position": str, "driver_number": str}
    return pd.concat([pd.read_csv(f, dtype=dtypes) for f in files], ignore_index=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read every season's events / qualifying / races / weather CSV and concatenate them."""
    events = _read_all("events")
    quali = _read_all("qualifying")
    races = _read_all("races")
    weather = _read_all("weather")
    return events, quali, races, weather


# --------------------------------------------------------------------------- #
# 2. Merge sessions into one base table
# --------------------------------------------------------------------------- #
def merge_sessions(
    events: pd.DataFrame, quali: pd.DataFrame, races: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Join race results with the same weekend's qualifying results, the event
    metadata and the per-session weather summaries.

    The race table is the spine (left side) because the row we want to predict
    is "driver D in race N".  Qualifying columns are attached on
    (season, round, driver); a driver who never set a qualifying time simply
    gets NaNs there, which later steps turn into "back of the grid" values.
    """
    key = ["season", "round", "driver"]
    q = quali.drop(columns=["driver_number", "driver_name", "team"])
    df = races.merge(q, on=key, how="left")
    df = df.merge(events, on=["season", "round"], how="left")

    wx_q = weather[weather["session"] == "Q"].drop(columns="session").add_prefix("quali_")
    wx_r = weather[weather["session"] == "R"].drop(columns="session").add_prefix("race_")
    df = df.merge(wx_q, left_on=["season", "round"], right_on=["quali_season", "quali_round"], how="left")
    df = df.merge(wx_r, left_on=["season", "round"], right_on=["race_season", "race_round"], how="left")
    df = df.drop(columns=["quali_season", "quali_round", "race_season", "race_round"])

    df["team_key"] = df["team"].map(canonical_team)
    df["dnf"] = df["classified_position"].map(classified_to_dnf)
    # Rows without a race result (a race still to be predicted) are not DNFs.
    df.loc[df["finish_position"].isna(), "dnf"] = np.nan

    # Drivers who never started (withdrew, did not start) have no meaningful
    # finishing position to learn from, so they are removed from the table.
    never_started = df["status"].fillna("").str.contains("not start|withdr", case=False, regex=True)
    if never_started.any():
        log.info("Dropping %d rows where the driver never started the race", int(never_started.sum()))
        df = df[~never_started].copy()

    df = df.sort_values(["season", "round", "finish_position"], kind="mergesort").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Feature blocks
# --------------------------------------------------------------------------- #
def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features describing Saturday's qualifying and the final starting grid.

    * best qualifying lap = fastest of Q1/Q2/Q3 (a driver knocked out in Q1
      only has a Q1 time)
    * gap to pole is expressed as a *percentage* of the pole time so that a
      0.5 s gap in Monaco (70 s lap) and Spa (105 s lap) are comparable
    * ``grid_position`` 0 means "pit-lane start" in FastF1; both that and a
      missing qualifying position are mapped to the back of the field
    """
    df = df.copy()
    n_drivers = df.groupby(["season", "round"])["driver"].transform("count")

    df["best_quali_time"] = df[["q1_time", "q2_time", "q3_time"]].min(axis=1)
    pole_time = df.groupby(["season", "round"])["best_quali_time"].transform("min")
    df["quali_gap_to_pole_s"] = df["best_quali_time"] - pole_time
    df["quali_gap_to_pole_pct"] = 100.0 * df["quali_gap_to_pole_s"] / pole_time
    df["q2_reached"] = df["q2_time"].notna().astype(int)
    df["q3_reached"] = df["q3_time"].notna().astype(int)

    df["quali_position"] = df["quali_position"].fillna(n_drivers)
    grid = df["grid_position"].replace(0, np.nan)
    df["grid_position"] = grid.fillna(n_drivers)
    df["grid_penalty"] = df["grid_position"] - df["quali_position"]
    return df


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous ``window`` values, *excluding* the current row."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling driver statistics.  Every series is shifted by one race inside the
    driver's own chronological history, so the value stored on race N only
    summarises races 1 .. N-1.
    """
    df = df.sort_values(["season", "round", "driver"], kind="mergesort").copy()
    g = df.groupby("driver", sort=False)

    df["driver_avg_finish_last3"] = g["finish_position"].transform(lambda s: _shifted_rolling_mean(s, 3))
    df["driver_avg_finish_last5"] = g["finish_position"].transform(lambda s: _shifted_rolling_mean(s, 5))
    df["driver_avg_quali_last5"] = g["quali_position"].transform(lambda s: _shifted_rolling_mean(s, 5))
    df["driver_dnf_rate_last10"] = g["dnf"].transform(lambda s: _shifted_rolling_mean(s, 10))
    df["driver_experience"] = g.cumcount()  # races in the dataset before this one

    # Championship points accumulated *before* this race (within the season):
    # running total including this race, minus this race's own points.  A race
    # that has not happened yet has NaN points, which counts as 0 here.
    pts = df["points"].fillna(0)
    df["driver_season_points"] = pts.groupby([df["season"], df["driver"]]).cumsum() - pts
    df["driver_season_rank"] = (
        df.groupby(["season", "round"])["driver_season_points"].rank(method="min", ascending=False)
    )
    return df


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team-level form: average of the team's two cars per race, rolled over the
    previous five races, plus constructor points / rank before this race.
    """
    df = df.copy()
    team_race = (
        df.groupby(["season", "round", "team_key"], as_index=False)
        .agg(team_race_avg_finish=("finish_position", "mean"), team_race_points=("points", "sum"))
        .sort_values(["season", "round"], kind="mergesort")
    )
    tg = team_race.groupby("team_key", sort=False)
    team_race["team_avg_finish_last5"] = tg["team_race_avg_finish"].transform(lambda s: _shifted_rolling_mean(s, 5))
    team_race["team_season_points"] = (
        team_race.groupby(["season", "team_key"], sort=False)["team_race_points"].cumsum() - team_race["team_race_points"]
    )
    team_race["team_season_rank"] = (
        team_race.groupby(["season", "round"])["team_season_points"].rank(method="min", ascending=False)
    )
    cols = ["season", "round", "team_key", "team_avg_finish_last5", "team_season_points", "team_season_rank"]
    return df.merge(team_race[cols], on=["season", "round", "team_key"], how="left")


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How the driver has done at this venue in *earlier* visits (expanding mean,
    shifted by one visit) and how many earlier visits there were.
    """
    df = df.sort_values(["season", "round", "driver"], kind="mergesort").copy()
    g = df.groupby(["driver", "location"], sort=False)
    df["driver_circuit_avg_finish"] = g["finish_position"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_circuit_visits"] = g.cumcount()
    return df


# --------------------------------------------------------------------------- #
# 4. Orchestration
# --------------------------------------------------------------------------- #
def build_feature_table(
    events: pd.DataFrame, quali: pd.DataFrame, races: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Run every feature block in order and return the tidy table.

    The function is pure (raw frames in, feature frame out) so ``predict.py``
    can call it with an extra, not-yet-raced round appended to the raw frames
    and obtain leakage-safe features for that round too.
    """
    df = merge_sessions(events, quali, races, weather)
    df = add_qualifying_features(df)
    df = add_driver_form_features(df)
    df = add_team_form_features(df)
    df = add_circuit_features(df)
    df = df.sort_values(["season", "round", "finish_position", "grid_position"], kind="mergesort").reset_index(drop=True)
    keep = [c for c in META_COLUMNS if c in df.columns] + FEATURE_COLUMNS
    return df[keep]


def sanity_checks(df: pd.DataFrame) -> None:
    """Print null counts and assert the properties that guarantee no leakage."""
    log.info("Feature table: %d rows x %d cols, %d races, seasons %s",
             len(df), df.shape[1], df.groupby(["season", "round"]).ngroups, sorted(df["season"].unique()))
    nulls = df[FEATURE_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    log.info("Null counts (expected only for first races of a driver/team/circuit, and rare quali gaps):\n%s",
             nulls.to_string() if len(nulls) else "  none")

    # 1) every driver's first appearance has no rolling history
    first = df.sort_values(["season", "round"]).groupby("driver").head(1)
    assert first["driver_avg_finish_last5"].isna().all(), "rolling feature leaks the current race"
    assert (first["driver_experience"] == 0).all()
    # 2) season points before round 1 are always zero
    r1 = df[df["round"] == df.groupby("season")["round"].transform("min")]
    assert (r1["driver_season_points"] == 0).all() and (r1["team_season_points"] == 0).all()
    # 3) one row per driver per race
    assert not df.duplicated(["season", "round", "driver"]).any()
    # 4) the target is a real classification position for every completed race
    done = df[df[TARGET].notna()]
    assert (done[TARGET] >= 1).all()
    log.info("Sanity checks passed")


def main() -> None:
    events, quali, races, weather = load_raw()
    df = build_feature_table(events, quali, races, weather)
    sanity_checks(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    df.to_csv(FEATURES_PATH.with_suffix(".csv"), index=False)
    log.info("Saved %s", FEATURES_PATH)


if __name__ == "__main__":
    main()
