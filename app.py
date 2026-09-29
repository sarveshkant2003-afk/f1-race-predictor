"""
F1 Race Result Predictor - Streamlit interface.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from feature_engineering import FEATURE_COLUMNS, TARGET  # noqa: E402
from predict import (  # noqa: E402
    build_features_for_new_race,
    load_features,
    load_metrics,
    load_model,
    predict_lineup,
    race_metrics,
)
from utils import IMPORTANCE_PATH, MODELS_DIR, RAW_DIR, enable_fastf1_cache  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
COLORS = {"predicted": "#2a78d6", "actual": "#eb6834", "accent": "#1baf7a", "neutral": "#9a9890"}
MODEL_LABELS = {"baseline": "Baseline (grid = finish)", "ridge": "Ridge regression",
                "random_forest": "Random Forest", "xgboost": "XGBoost"}
METRIC_LABELS = {
    "mae": "MAE (positions)", "spearman": "Spearman ρ", "podium_exact_rate": "Podium exact-set rate",
    "podium_overlap": "Podium overlap", "exact_position_acc": "Exact-position accuracy",
    "winner_acc": "Winner accuracy", "n_races": "Races",
}
FEATURE_LABELS = {
    "quali_position": "Qualifying position", "quali_gap_to_pole_pct": "Gap to pole (%)",
    "q2_reached": "Reached Q2", "q3_reached": "Reached Q3", "grid_position": "Grid position",
    "grid_penalty": "Grid penalty (places)", "driver_avg_finish_last3": "Driver avg finish, last 3",
    "driver_avg_finish_last5": "Driver avg finish, last 5", "driver_avg_quali_last5": "Driver avg quali, last 5",
    "driver_dnf_rate_last10": "Driver DNF rate, last 10", "driver_season_points": "Driver season points",
    "driver_season_rank": "Driver championship rank", "driver_experience": "Driver races in dataset",
    "team_avg_finish_last5": "Team avg finish, last 5", "team_season_points": "Team season points",
    "team_season_rank": "Constructor rank", "driver_circuit_avg_finish": "Driver avg finish at circuit",
    "driver_circuit_visits": "Driver visits to circuit", "is_street_circuit": "Street circuit",
    "is_sprint_weekend": "Sprint weekend", "quali_is_wet": "Wet qualifying",
    "quali_rain_fraction": "Qualifying rain fraction", "quali_air_temp": "Qualifying air temp",
    "quali_track_temp": "Qualifying track temp", "quali_humidity": "Qualifying humidity",
    "quali_wind_speed": "Qualifying wind speed",
}

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders - disk reads happen once per session, models once per process
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    return load_features()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return load_model(name)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_schedule(season: int) -> pd.DataFrame:
    """
    Event list for the dropdown.  Collected seasons come from the events CSV
    written by ``data_collection.py``; any other season is read live from the
    FastF1 schedule (cached for six hours) so upcoming rounds are selectable.
    """
    path = RAW_DIR / f"events_{season}.csv"
    if path.exists():
        ev = pd.read_csv(path)
        return ev[["round", "event_name", "location", "country", "event_date"]]
    import fastf1

    enable_fastf1_cache()
    sched = fastf1.get_event_schedule(season, include_testing=False)
    return pd.DataFrame(
        {
            "round": sched["RoundNumber"].astype(int),
            "event_name": sched["EventName"],
            "location": sched["Location"],
            "country": sched["Country"],
            "event_date": pd.to_datetime(sched["EventDate"]).dt.date.astype(str),
        }
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def get_live_race_rows(season: int, rnd: int) -> pd.DataFrame:
    """Feature rows for a round that is not in the processed table (hits FastF1)."""
    return build_features_for_new_race(season, rnd)


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def lineup_chart(lineup: pd.DataFrame) -> go.Figure:
    """Dumbbell chart: one row per driver, predicted position vs actual position."""
    d = lineup.sort_values(TARGET, ascending=False)
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["predicted_position"], r[TARGET]], y=[r["driver"], r["driver"]], mode="lines",
            line=dict(color=COLORS["neutral"], width=2), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=d["predicted_position"], y=d["driver"], mode="markers", name="Predicted",
        marker=dict(color=COLORS["predicted"], size=11, line=dict(color="white", width=2)),
        hovertemplate="%{y}<br>Predicted P%{x}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d[TARGET], y=d["driver"], mode="markers", name="Actual",
        marker=dict(color=COLORS["actual"], size=11, line=dict(color="white", width=2)),
        hovertemplate="%{y}<br>Actual P%{x}<extra></extra>",
    ))
    fig.update_layout(
        height=max(420, 24 * len(d)), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Position", dtick=1, range=[0.5, len(d) + 0.5], gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=None, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def importance_chart(imp: pd.DataFrame) -> go.Figure:
    d = imp.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["importance"], y=d["feature"].map(lambda f: FEATURE_LABELS.get(f, f)), orientation="h",
        marker=dict(color=COLORS["predicted"]), text=d["importance"].map(lambda v: f"{v:.3f}"),
        textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=26 * len(d) + 80, margin=dict(l=10, r=40, t=20, b=10),
        xaxis=dict(title="Normalised importance (sums to 1)", gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=None), bargap=0.25,
    )
    return fig


def metrics_table(report_models: dict, split: str, order: list[str], best: str) -> pd.DataFrame:
    """One row per model, one column per metric, formatted as strings for st.table."""
    t = pd.DataFrame({k: v[split] for k, v in report_models.items()}).T.loc[order]
    t = t[list(METRIC_LABELS)].rename(columns=METRIC_LABELS)
    for c in t.columns:
        t[c] = t[c].map(lambda v: f"{v:.3f}") if c != "Races" else t[c].astype(int).astype(str)
    t.index = [MODEL_LABELS[m] + ("  ★" if m == best and split == "test" else "") for m in t.index]
    t.index.name = "Model"
    return t


def comparison_chart(table: pd.DataFrame, metric: str, title: str) -> go.Figure:
    colors = [COLORS["neutral"] if m == "baseline" else COLORS["predicted"] for m in table.index]
    fig = go.Figure(go.Bar(
        x=[MODEL_LABELS.get(m, m) for m in table.index], y=table[metric], marker=dict(color=colors),
        text=table[metric].map(lambda v: f"{v:.3f}"), textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=320, margin=dict(l=10, r=10, t=50, b=10), bargap=0.4,
        yaxis=dict(title=None, gridcolor="rgba(128,128,128,0.15)"), xaxis=dict(title=None),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: model + race selection
# --------------------------------------------------------------------------- #
features = get_features()
metrics_report = get_metrics()
importance_df = get_importance()

collected_seasons = sorted(int(s) for s in features["season"].unique())
current_year = date.today().year
seasons = sorted(set(collected_seasons) | {current_year}, reverse=True)

with st.sidebar:
    st.title("🏎️ F1 Race Predictor")
    model_names = [m for m in ("xgboost", "random_forest", "ridge") if (MODELS_DIR / f"{m}.pkl").exists()]
    best = metrics_report["best_model"]
    model_name = st.radio(
        "Model", model_names, index=model_names.index(best),
        format_func=lambda m: MODEL_LABELS[m] + ("  ★ best" if m == best else ""),
    )
    season = st.selectbox("Season", seasons, index=seasons.index(max(collected_seasons)))
    schedule = get_schedule(season)
    race_label = st.selectbox(
        "Race", schedule["round"].tolist(),
        format_func=lambda r: f"R{r:02d} · {schedule.set_index('round').loc[r, 'event_name']}",
    )
    rnd = int(race_label)
    st.caption(
        f"Trained on {metrics_report['train_seasons'][0]}–{metrics_report['train_seasons'][-1]}, "
        f"tested on {metrics_report['test_season']}. {metrics_report['n_train_rows']} training rows."
    )

event = schedule.set_index("round").loc[rnd]
model = get_model(model_name)

# --------------------------------------------------------------------------- #
# Feature rows for the chosen race
# --------------------------------------------------------------------------- #
race_rows = features[(features["season"] == season) & (features["round"] == rnd)]
source = "Processed"
fetch_error = "no rows returned"
if race_rows.empty:
    source = "Live fetch"
    with st.spinner(f"{event['event_name']} {season} is not in the processed data - fetching qualifying from FastF1…"):
        try:
            race_rows = get_live_race_rows(season, rnd)
        except Exception as exc:  # future race, network failure ...
            race_rows = pd.DataFrame()
            fetch_error = str(exc)

st.title(f"{event['event_name']} {season}")
st.caption(f"Round {rnd} · {event['location']}, {event['country']} · {event['event_date']}")

if race_rows.empty:
    st.warning(
        f"No qualifying data is available for this race yet, so no lineup can be predicted. "
        f"({fetch_error})"
    )
    st.stop()

lineup = predict_lineup(race_rows, model)
has_result = lineup[TARGET].notna().any()
race_m = race_metrics(lineup, season, rnd)
row0 = race_rows.iloc[0]

tab_lineup, tab_importance, tab_models, tab_data = st.tabs(
    ["Predicted lineup", "Feature importance", "Model comparison", "Data"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - lineup and (if available) predicted vs actual
# --------------------------------------------------------------------------- #
with tab_lineup:
    ctx = st.columns(5)
    ctx[0].metric("Circuit type", "Street" if row0["is_street_circuit"] else "Permanent")
    ctx[1].metric("Weekend format", "Sprint" if row0["is_sprint_weekend"] else "Conventional")
    ctx[2].metric("Qualifying", "Wet" if row0["quali_is_wet"] == 1 else "Dry")
    if has_result:
        ctx[3].metric("Race", "Wet" if row0.get("race_is_wet") == 1 else "Dry")
    ctx[4].metric("Data source", source)

    if has_result and race_m:
        st.subheader("How good was the prediction?")
        m = st.columns(4)
        m[0].metric("MAE", f"{race_m['mae']:.2f} positions")
        m[1].metric("Spearman ρ", f"{race_m['spearman']:.3f}")
        m[2].metric("Podium overlap", f"{race_m['podium_overlap'] * 100:.0f}%",
                    help="Share of the predicted top-3 who really finished on the podium")
        m[3].metric("Exact positions", f"{race_m['exact_position_acc'] * 100:.0f}%")

    table = lineup.copy()
    table["grid_position"] = table["grid_position"].astype(int)
    table["quali_position"] = table["quali_position"].astype(int)
    if has_result:
        table["finish_position"] = table["finish_position"].astype("Int64")
        table["delta"] = (table["predicted_position"] - table["finish_position"]).astype("Int64")
        table["status"] = table["status"].fillna("")
        shown = table[["predicted_position", "driver", "driver_name", "team", "grid_position", "quali_position",
                       "finish_position", "delta", "status", "predicted_score"]]
        col_cfg = {
            "predicted_position": st.column_config.NumberColumn("Predicted", format="P%d"),
            "finish_position": st.column_config.NumberColumn("Actual", format="P%d"),
            "delta": st.column_config.NumberColumn("Pred − actual", help="Negative = finished worse than predicted"),
            "status": "Status",
        }
    else:
        shown = table[["predicted_position", "driver", "driver_name", "team", "grid_position", "quali_position",
                       "predicted_score"]]
        col_cfg = {"predicted_position": st.column_config.NumberColumn("Predicted", format="P%d")}
        st.info("This race has not been run yet (or its result is not collected). Grid = qualifying order, "
                "penalties unknown.")
    col_cfg.update({
        "driver": "Code", "driver_name": "Driver", "team": "Team",
        "grid_position": st.column_config.NumberColumn("Grid", format="P%d"),
        "quali_position": st.column_config.NumberColumn("Quali", format="P%d"),
        "predicted_score": st.column_config.NumberColumn("Model score", format="%.2f",
                                                          help="Raw regression output: expected finishing position"),
    })
    left, right = st.columns([1.15, 1]) if has_result else (st.container(), None)
    with left:
        st.subheader("Predicted lineup" + (" vs actual result" if has_result else ""))
        st.dataframe(shown, hide_index=True, width="stretch", column_config=col_cfg,
                     height=38 * (len(shown) + 1))
    if has_result:
        with right:
            st.subheader("Predicted vs actual, per driver")
            st.plotly_chart(lineup_chart(lineup), width="stretch")

    with st.expander("Model inputs for this race (the feature table rows)"):
        st.dataframe(race_rows[["driver", "team"] + FEATURE_COLUMNS], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2 - feature importance
# --------------------------------------------------------------------------- #
with tab_importance:
    st.subheader(f"Feature importance · {MODEL_LABELS[model_name]}")
    st.caption(
        "Tree models: impurity-based importance (how much each feature reduced squared error across all splits). "
        "Ridge: absolute standardised coefficient. Each model's vector is normalised to sum to 1."
    )
    st.plotly_chart(importance_chart(importance_df[importance_df["model"] == model_name]), width="stretch")
    with st.expander("All models side by side"):
        wide = importance_df.pivot(index="feature", columns="model", values="importance")
        wide.index = wide.index.map(lambda f: FEATURE_LABELS.get(f, f))
        st.table(wide.sort_values(model_name, ascending=False).round(4))

# --------------------------------------------------------------------------- #
# Tab 3 - model comparison
# --------------------------------------------------------------------------- #
with tab_models:
    st.subheader(f"Held-out test season {metrics_report['test_season']}")
    order = [m for m in ("baseline", "ridge", "random_forest", "xgboost") if m in metrics_report["models"]]
    test_table = pd.DataFrame({k: v["test"] for k, v in metrics_report["models"].items()}).T.loc[order]
    # st.table (static HTML) is used for the small summary tables because the
    # interactive data grid mis-measures its width when first drawn in a tab
    # that is not visible yet.
    st.table(metrics_table(metrics_report["models"], "test", order, best))

    c1, c2 = st.columns(2)
    c1.plotly_chart(comparison_chart(test_table, "mae", "Mean absolute error (lower is better)"), width="stretch")
    c2.plotly_chart(comparison_chart(test_table, "spearman", "Spearman rank correlation (higher is better)"), width="stretch")

    with st.expander("Training-set metrics (to spot over-fitting)"):
        st.table(metrics_table(metrics_report["models"], "train", order, best))
    st.caption(f"Trained {metrics_report['trained_at']} on seasons {metrics_report['train_seasons']} "
               f"({metrics_report['n_train_rows']} driver-race rows); tested on {metrics_report['n_test_rows']} rows.")

# --------------------------------------------------------------------------- #
# Tab 4 - dataset overview
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Processed dataset")
    per_season = (
        features.groupby("season")
        .agg(races=("round", "nunique"), rows=("driver", "size"), drivers=("driver", "nunique"),
             wet_qualifyings=("quali_is_wet", lambda s: int(s.groupby(features.loc[s.index, "round"]).first().sum())),
             dnf_rate=("dnf", "mean"))
        .reset_index()
    )
    per_season["dnf_rate"] = (per_season["dnf_rate"] * 100).map(lambda v: f"{v:.1f} %")
    st.table(per_season.set_index("season"))
    st.caption(f"{len(features)} driver-race rows · {len(FEATURE_COLUMNS)} features · "
               f"{features.groupby(['season', 'round']).ngroups} races")
    st.dataframe(features.tail(40), hide_index=True, width="stretch", key="features_tail")
