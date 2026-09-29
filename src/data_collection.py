"""
Pull qualifying, race and weather data for whole F1 seasons via FastF1.

Usage
-----
    python src/data_collection.py                      # 2021 .. 2025
    python src/data_collection.py --seasons 2024 2025  # specific seasons
    python src/data_collection.py --force              # ignore existing raw files

Output (one set of files per season in ``data/raw/``):
    events_<year>.csv      one row per Grand Prix (schedule metadata)
    qualifying_<year>.csv  one row per driver per qualifying session
    races_<year>.csv       one row per driver per race
    weather_<year>.csv     one row per session (Q and R) with aggregated weather

Every row is written as soon as a round finishes loading, so an interrupted run
can be resumed and only the missing rounds are fetched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    DEFAULT_SEASONS,
    RAW_DIR,
    enable_fastf1_cache,
    is_street_circuit,
    setup_logging,
    timedelta_to_seconds,
)

log = setup_logging("data_collection")

# FastF1 politely caps itself at 500 API calls per rolling hour and raises
# RateLimitExceededError beyond that.  A round costs roughly 10 calls, so a
# full 5-season pull needs a few pauses.  When we hit the cap we simply wait.
RATE_LIMIT_WAIT_S = 10 * 60


def get_schedule(year: int) -> pd.DataFrame:
    """
    Return the race calendar for ``year`` with only the columns we need.

    Testing events are excluded and events that have not happened yet
    (EventDate in the future) are dropped so the loop below never asks FastF1
    for a session that does not exist.
    """
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = pd.Timestamp.now().normalize()
    schedule = schedule[schedule["EventDate"] <= today + pd.Timedelta(days=1)]
    events = pd.DataFrame(
        {
            "season": year,
            "round": schedule["RoundNumber"].astype(int),
            "event_name": schedule["EventName"],
            "location": schedule["Location"],
            "country": schedule["Country"],
            "event_date": pd.to_datetime(schedule["EventDate"]).dt.date.astype(str),
            "event_format": schedule["EventFormat"],
            "is_sprint_weekend": schedule["EventFormat"].str.contains("sprint", case=False).astype(int),
            "is_street_circuit": schedule["Location"].map(is_street_circuit),
        }
    )
    return events.reset_index(drop=True)


def load_session(year: int, rnd: int, kind: str):
    """
    Fetch one session ('Q' or 'R') with results + weather but *without* laps or
    telemetry.  Lap/telemetry data is orders of magnitude larger and is not
    needed for session-level modelling.
    """
    import fastf1

    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=False, telemetry=False, weather=True, messages=False)
    return session


def load_round(year: int, rnd: int):
    """
    Load the qualifying and race sessions of one round, pausing and retrying
    whenever FastF1's self-imposed hourly rate limit is reached.
    """
    from fastf1.exceptions import RateLimitExceededError

    while True:
        try:
            return load_session(year, rnd, "Q"), load_session(year, rnd, "R")
        except RateLimitExceededError as exc:
            log.warning("%s - sleeping %d min before retrying %d R%02d", exc, RATE_LIMIT_WAIT_S // 60, year, rnd)
            time.sleep(RATE_LIMIT_WAIT_S)


def summarise_weather(session, year: int, rnd: int, kind: str) -> dict:
    """
    Collapse the minute-by-minute weather stream of a session into one row.

    ``rain_fraction`` is the share of samples where the rain sensor was on;
    ``is_wet`` flags sessions where it rained for at least 10 % of the samples.
    """
    wx = session.weather_data
    row = {"season": year, "round": rnd, "session": kind}
    if wx is None or len(wx) == 0:
        row.update(
            air_temp=float("nan"), track_temp=float("nan"), humidity=float("nan"),
            wind_speed=float("nan"), rain_fraction=float("nan"), is_wet=float("nan"),
        )
        return row
    rain_fraction = float(wx["Rainfall"].astype(bool).mean())
    row.update(
        air_temp=float(wx["AirTemp"].mean()),
        track_temp=float(wx["TrackTemp"].mean()),
        humidity=float(wx["Humidity"].mean()),
        wind_speed=float(wx["WindSpeed"].mean()),
        rain_fraction=rain_fraction,
        is_wet=int(rain_fraction >= 0.10),
    )
    return row


def qualifying_rows(session, year: int, rnd: int) -> list[dict]:
    """
    One dict per driver from a qualifying session.

    Q1/Q2/Q3 are pandas Timedeltas (NaT when the driver did not take part in
    that segment); they are converted to plain float seconds here so the CSV
    is trivially readable later.
    """
    rows = []
    for _, r in session.results.iterrows():
        q1 = timedelta_to_seconds(r.get("Q1"))
        q2 = timedelta_to_seconds(r.get("Q2"))
        q3 = timedelta_to_seconds(r.get("Q3"))
        rows.append(
            {
                "season": year,
                "round": rnd,
                "driver_number": str(r["DriverNumber"]),
                "driver": r["Abbreviation"],
                "driver_name": r["FullName"],
                "team": r["TeamName"],
                "quali_position": float(r["Position"]) if pd.notna(r["Position"]) else float("nan"),
                "q1_time": q1,
                "q2_time": q2,
                "q3_time": q3,
            }
        )
    return rows


def race_rows(session, year: int, rnd: int) -> list[dict]:
    """
    One dict per driver from a race session.

    ``finish_position`` is FastF1's ``Position`` – the full classification
    order including retired cars (ordered by laps completed), which is exactly
    the "race-day lineup" we want to predict.  ``classified_position`` keeps
    the raw code ('R', 'D', ...) so the DNF flag can be derived later.
    """
    rows = []
    for _, r in session.results.iterrows():
        rows.append(
            {
                "season": year,
                "round": rnd,
                "driver_number": str(r["DriverNumber"]),
                "driver": r["Abbreviation"],
                "driver_name": r["FullName"],
                "team": r["TeamName"],
                "grid_position": float(r["GridPosition"]) if pd.notna(r["GridPosition"]) else float("nan"),
                "finish_position": float(r["Position"]) if pd.notna(r["Position"]) else float("nan"),
                "classified_position": r["ClassifiedPosition"],
                "status": r["Status"],
                "points": float(r["Points"]) if pd.notna(r["Points"]) else 0.0,
                "laps_completed": float(r["Laps"]) if pd.notna(r.get("Laps")) else float("nan"),
            }
        )
    return rows


def _read_existing(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect_season(year: int, force: bool = False) -> None:
    """
    Download every completed round of ``year`` and append it to the season's
    raw CSV files.  Rounds already present on disk are skipped unless
    ``force`` is set.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    events = get_schedule(year)
    events.to_csv(RAW_DIR / f"events_{year}.csv", index=False)

    q_path = RAW_DIR / f"qualifying_{year}.csv"
    r_path = RAW_DIR / f"races_{year}.csv"
    w_path = RAW_DIR / f"weather_{year}.csv"
    if force:
        for p in (q_path, r_path, w_path):
            p.unlink(missing_ok=True)

    quali_df, race_df, wx_df = _read_existing(q_path), _read_existing(r_path), _read_existing(w_path)
    done_rounds = set(race_df["round"].astype(int)) if len(race_df) else set()

    for _, ev in events.iterrows():
        rnd = int(ev["round"])
        if rnd in done_rounds:
            log.info("%d R%02d %-28s already collected, skipping", year, rnd, ev["event_name"])
            continue
        try:
            q, r = load_round(year, rnd)
        except Exception as exc:  # network error, cancelled event, missing data ...
            log.warning("%d R%02d %-28s FAILED: %s", year, rnd, ev["event_name"], exc)
            continue
        if r.results is None or len(r.results) == 0:
            log.warning("%d R%02d %-28s has no race results, skipping", year, rnd, ev["event_name"])
            continue

        quali_df = pd.concat([quali_df, pd.DataFrame(qualifying_rows(q, year, rnd))], ignore_index=True)
        race_df = pd.concat([race_df, pd.DataFrame(race_rows(r, year, rnd))], ignore_index=True)
        wx_df = pd.concat(
            [wx_df, pd.DataFrame([summarise_weather(q, year, rnd, "Q"), summarise_weather(r, year, rnd, "R")])],
            ignore_index=True,
        )
        # persist after every round so an interrupted run loses nothing
        quali_df.to_csv(q_path, index=False)
        race_df.to_csv(r_path, index=False)
        wx_df.to_csv(w_path, index=False)
        log.info(
            "%d R%02d %-28s Q=%2d drivers  R=%2d drivers  wet=%s",
            year, rnd, ev["event_name"], len(q.results), len(r.results), bool(wx_df.iloc[-1]["is_wet"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true", help="re-download seasons even if raw files exist")
    args = parser.parse_args()

    enable_fastf1_cache()
    for year in args.seasons:
        log.info("=== Season %d ===", year)
        collect_season(year, force=args.force)
    log.info("Done. Raw files are in %s", RAW_DIR)


if __name__ == "__main__":
    main()
