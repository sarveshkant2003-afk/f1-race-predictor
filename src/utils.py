"""
Shared helpers and constants used by every other module in ``src/``.

Everything that more than one script needs lives here so that paths, the
FastF1 cache location, team-name normalisation and circuit metadata are
defined exactly once.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

# Seasons collected by default: 2021 up to the most recently *completed* season.
DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]

# --------------------------------------------------------------------------- #
# Team lineage
# --------------------------------------------------------------------------- #
# The same racing team changes its commercial name between (and even within)
# seasons.  For "team form" features we need one stable key per lineage so
# that e.g. AlphaTauri's 2023 results count as RB's history in 2024.
TEAM_ALIASES: dict[str, str] = {
    "Red Bull Racing": "Red Bull",
    "Red Bull": "Red Bull",
    "Ferrari": "Ferrari",
    "Mercedes": "Mercedes",
    "McLaren": "McLaren",
    "Alpine": "Alpine",
    "Alpine F1 Team": "Alpine",
    "Aston Martin": "Aston Martin",
    "Williams": "Williams",
    "Haas F1 Team": "Haas",
    "Haas": "Haas",
    "AlphaTauri": "RB",
    "Scuderia AlphaTauri": "RB",
    "RB": "RB",
    "Visa Cash App RB": "RB",
    "Racing Bulls": "RB",
    "Alfa Romeo": "Sauber",
    "Alfa Romeo Racing": "Sauber",
    "Kick Sauber": "Sauber",
    "Sauber": "Sauber",
    "Stake F1 Team Kick Sauber": "Sauber",
    "Audi": "Sauber",
    "Cadillac": "Cadillac",
}


def canonical_team(name: str) -> str:
    """Map a FastF1 ``TeamName`` to a stable lineage key (see ``TEAM_ALIASES``)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Unknown"
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


# --------------------------------------------------------------------------- #
# Circuit metadata
# --------------------------------------------------------------------------- #
# Keyed by the FastF1 event ``Location`` field.  Street / temporary circuits
# have walls close to the track, fewer overtaking spots and more safety cars,
# which changes how much the finishing order can differ from the grid.
STREET_CIRCUITS: set[str] = {
    "Monaco",
    "Baku",
    "Singapore",
    "Jeddah",
    "Melbourne",
    "Miami",
    "Las Vegas",
    "Montréal",
    "Montreal",
    "Sochi",
    "Madrid",
}


def is_street_circuit(location: str) -> int:
    """Return 1 if the event location is a street / temporary circuit, else 0."""
    return int(str(location) in STREET_CIRCUITS)


# --------------------------------------------------------------------------- #
# FastF1 setup
# --------------------------------------------------------------------------- #
def enable_fastf1_cache() -> None:
    """
    Turn on FastF1's on-disk cache and silence its chatty logger.

    Every script that touches the FastF1 API must call this first: session
    loads are slow network operations and the cache makes re-runs nearly free.
    """
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("ERROR")
    # set_log_level only touches FastF1's own logger; the child loggers it
    # creates per module still propagate INFO lines to the root logger.
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("fastf1"):
            logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)


def setup_logging(name: str) -> logging.Logger:
    """Create a module logger that prints timestamped lines to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Small conversions
# --------------------------------------------------------------------------- #
def timedelta_to_seconds(value) -> float:
    """Convert a pandas ``Timedelta`` (or NaT / None) to float seconds (or NaN)."""
    if value is None or pd.isna(value):
        return float("nan")
    return pd.to_timedelta(value).total_seconds()


def classified_to_dnf(classified_position) -> int:
    """
    FastF1's ``ClassifiedPosition`` is a digit string for classified finishers
    and a letter code otherwise ('R' retired, 'D' disqualified, 'W' withdrawn,
    'E' excluded, 'F' failed to qualify, 'N' not classified).  Return 1 when
    the driver did *not* finish classified.
    """
    if classified_position is None or pd.isna(classified_position):
        return 1
    return int(not str(classified_position).strip().isdigit())


# --------------------------------------------------------------------------- #
# Evaluation metrics (shared by train_model.py, predict.py and app.py)
# --------------------------------------------------------------------------- #
def add_predicted_rank(df: pd.DataFrame, pred_col: str = "predicted_score", out_col: str = "predicted_position") -> pd.DataFrame:
    """
    Convert raw regression outputs into a lineup: within each race, the driver
    with the lowest predicted score gets predicted position 1, the next 2, ...
    ``method="first"`` breaks ties by row order so positions are always the
    integers 1..n with no gaps.
    """
    df = df.copy()
    df[out_col] = df.groupby(["season", "round"])[pred_col].rank(method="first").astype(int)
    return df


def compute_metrics(df: pd.DataFrame, pred_col: str = "predicted_score", actual_col: str = "finish_position") -> dict:
    """
    Score a set of per-driver predictions against actual finishing positions.

    ``df`` must contain ``season`` and ``round`` (to group drivers into races),
    the raw model output ``pred_col`` and the truth ``actual_col``.

    Returned keys
    -------------
    mae                   mean |predicted position (ranked) - actual position|
    mae_raw               same but on the un-ranked regression output
    spearman              mean per-race Spearman correlation between predicted
                          and actual order
    podium_exact_rate     share of races where the predicted top-3 *set* equals
                          the real top-3 set (order-agnostic)
    podium_overlap        mean fraction of the predicted podium that really
                          finished on the podium (2 of 3 right -> 0.667)
    exact_position_acc    share of drivers whose predicted position is exactly
                          their actual position
    winner_acc            share of races where the predicted winner won
    n_races, n_rows       sizes, for context
    """
    from scipy.stats import spearmanr

    d = df.dropna(subset=[actual_col]).copy()
    d = add_predicted_rank(d, pred_col, "_rank")
    races = d.groupby(["season", "round"], sort=False)

    spearmans, podium_exact, podium_overlap, winners = [], [], [], []
    for _, race in races:
        if len(race) < 3:
            continue
        rho = spearmanr(race["_rank"], race[actual_col]).correlation
        spearmans.append(0.0 if pd.isna(rho) else float(rho))
        pred_top3 = set(race.nsmallest(3, "_rank")["driver"])
        real_top3 = set(race.nsmallest(3, actual_col)["driver"])
        podium_exact.append(pred_top3 == real_top3)
        podium_overlap.append(len(pred_top3 & real_top3) / 3.0)
        pred_winner = race.loc[race["_rank"].idxmin(), "driver"]
        real_winner = race.loc[race[actual_col].idxmin(), "driver"]
        winners.append(pred_winner == real_winner)

    return {
        "mae": float((d["_rank"] - d[actual_col]).abs().mean()),
        "mae_raw": float((d[pred_col] - d[actual_col]).abs().mean()),
        "spearman": float(sum(spearmans) / max(len(spearmans), 1)),
        "podium_exact_rate": float(sum(podium_exact) / max(len(podium_exact), 1)),
        "podium_overlap": float(sum(podium_overlap) / max(len(podium_overlap), 1)),
        "exact_position_acc": float((d["_rank"] == d[actual_col]).mean()),
        "winner_acc": float(sum(winners) / max(len(winners), 1)),
        "n_races": int(len(spearmans)),
        "n_rows": int(len(d)),
    }
