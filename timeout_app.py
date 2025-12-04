# timeout_app.py

# timeout_app.py

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb  # <-- add this line


# --------------------------------------------------------------------------------
# 1. Load model + feature columns (cached)
# --------------------------------------------------------------------------------

@st.cache_resource
def load_model_and_columns():
    # Assumes these files live in the repo root
    with open("timeout_wp_xgb.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


xgb_model, FEATURE_COLUMNS = load_model_and_columns()

# These MUST match the training script for your WP model
feature_cols_numeric = [
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "clock_stop",
    "offense_trailing",
]

feature_cols_categorical = [
    "qtr",
    "play_type",
]

# --------------------------------------------------------------------------------
# 2. Shared helpers (design matrices + timeout logic)
# --------------------------------------------------------------------------------

def _build_design_matrix_from_situations(
    decisions_df: pd.DataFrame,
    call_timeout: bool,
    full_feature_columns,
):
    """
    Internal helper to convert a decisions_df into the X design matrix
    for a given action (call timeout vs not).

    Model is a pure WP model; it expects the same features used in training.
    We simulate the effect of a timeout only through `clock_stop`:

        - No timeout: use clock_stop as entered.
        - Timeout: same state but with clock_stop = 1.
    """

    df = decisions_df.copy()

    # offense_trailing flag
    df["offense_trailing"] = (df["score_differential"] < 0).astype(int)

    # normalize clock_stop to 0/1
    df["clock_stop"] = df["clock_stop"].astype(int).clip(lower=0, upper=1)

    if call_timeout:
        # Local model: stop the clock, keep timeouts the same
        df["clock_stop"] = 1

    # Split numeric/categorical
    X_num_new = df[feature_cols_numeric].copy()
    X_cat_new = df[feature_cols_categorical].copy()

    # One-hot encode categoricals
    X_cat_dummies_new = pd.get_dummies(X_cat_new, drop_first=True)

    # Combine numeric + categorical
    X_new = pd.concat(
        [X_num_new.reset_index(drop=True), X_cat_dummies_new.reset_index(drop=True)],
        axis=1,
    )

    # Align to training columns
    X_new = X_new.reindex(columns=full_feature_columns, fill_value=0)

    return X_new


def add_timeout_probs(
    decisions_df: pd.DataFrame,
    model,
    full_feature_columns=None,
) -> pd.DataFrame:
    """
    Compute nfl4th-style timeout decision probabilities.

    decisions_df must contain at least:
        qtr, game_seconds_remaining, half_seconds_remaining,
        score_differential, down, ydstogo, yardline_100,
        posteam_timeouts_remaining, defteam_timeouts_remaining,
        posteam, defteam, play_type, clock_stop.

    Returns:
        original cols + wp_timeout, wp_no_timeout, timeout_boost, recommendation
    """
    if full_feature_columns is None:
        full_feature_columns = FEATURE_COLUMNS

    required_for_decisions = [
        "qtr",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "score_differential",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam",
        "defteam",
        "play_type",
        "clock_stop",
    ]

    missing_decision_cols = [c for c in required_for_decisions if c not in decisions_df.columns]
    if missing_decision_cols:
        raise ValueError(f"Missing columns: {missing_decision_cols}")

    # Build X matrices
    X_timeout = _build_design_matrix_from_situations(
        decisions_df,
        call_timeout=True,
        full_feature_columns=full_feature_columns,
    )
    X_no_timeout = _build_design_matrix_from_situations(
        decisions_df,
        call_timeout=False,
        full_feature_columns=full_feature_columns,
    )

    # Predict WP under each choice
    wp_timeout = model.predict(X_timeout)
    wp_no_timeout = model.predict(X_no_timeout)

    wp_timeout = np.clip(wp_timeout, 0, 1)
    wp_no_timeout = np.clip(wp_no_timeout, 0, 1)

    probs = decisions_df.copy()
    probs["wp_timeout"] = wp_timeout
    probs["wp_no_timeout"] = wp_no_timeout
    probs["timeout_boost"] = probs["wp_timeout"] - probs["wp_no_timeout"]

    # Recommendation:
    # > 0.02  -> Call timeout
    # <= 0    -> Do not call timeout
    # (0,0.02]-> Toss up
    def _recommend(boost: float) -> str:
        if boost > 0.02:
            return "Recommendation: Call a timeout"
        elif boost <= 0:
            return "Recommendation: Do not call a timeout"
        else:
            return "Recommendation: Toss Up"

    probs["recommendation"] = probs["timeout_boost"].apply(_recommend)
    return probs


def make_timeout_table(probs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "posteam",
        "defteam",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "clock_stop",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "wp_no_timeout",
        "wp_timeout",
        "timeout_boost",
        "recommendation",
    ]
    existing = [c for c in cols if c in probs.columns]
    table = probs[existing].copy()

    if "timeout_boost" in table.columns:
        table = table.sort_values("timeout_boost", ascending=False)

    return table


# ---------- Delay of game vs timeout helpers ----------

def _build_design_matrix_delay_vs_timeout(
    decisions_df: pd.DataFrame,
    accept_penalty: bool,
    full_feature_columns,
):
    """
    Build X for:
        - accept_penalty = True  -> move ball back 5 yards, ydstogo + 5
        - accept_penalty = False -> call timeout: spend one timeout, stop clock
    """

    df = decisions_df.copy()

    df["offense_trailing"] = (df["score_differential"] < 0).astype(int)
    df["clock_stop"] = df["clock_stop"].astype(int).clip(lower=0, upper=1)

    if accept_penalty:
        # Option A: accept delay of game
        df["yardline_100"] = (df["yardline_100"] + 5).clip(upper=100)
        df["ydstogo"] = df["ydstogo"] + 5
        # timeouts and clock_stop unchanged (local field position comparison)
    else:
        # Option B: timeout to avoid penalty
        df["posteam_timeouts_remaining"] = (
            df["posteam_timeouts_remaining"] - 1
        ).clip(lower=0)
        df["clock_stop"] = 1

    X_num_new = df[feature_cols_numeric].copy()
    X_cat_new = df[feature_cols_categorical].copy()

    X_cat_dummies_new = pd.get_dummies(X_cat_new, drop_first=True)

    X_new = pd.concat(
        [X_num_new.reset_index(drop=True), X_cat_dummies_new.reset_index(drop=True)],
        axis=1,
    )

    X_new = X_new.reindex(columns=full_feature_columns, fill_value=0)
    return X_new


def add_delay_vs_timeout_probs(
    decisions_df: pd.DataFrame,
    model,
    full_feature_columns=None,
) -> pd.DataFrame:
    if full_feature_columns is None:
        full_feature_columns = FEATURE_COLUMNS

    required_for_decisions = [
        "qtr",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "score_differential",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam",
        "defteam",
        "play_type",
        "clock_stop",
    ]

    missing = [c for c in required_for_decisions if c not in decisions_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X_penalty = _build_design_matrix_delay_vs_timeout(
        decisions_df,
        accept_penalty=True,
        full_feature_columns=full_feature_columns,
    )
    X_timeout = _build_design_matrix_delay_vs_timeout(
        decisions_df,
        accept_penalty=False,
        full_feature_columns=full_feature_columns,
    )

    wp_penalty = model.predict(X_penalty)
    wp_timeout = model.predict(X_timeout)

    wp_penalty = np.clip(wp_penalty, 0, 1)
    wp_timeout = np.clip(wp_timeout, 0, 1)

    probs = decisions_df.copy()
    probs["wp_penalty"] = wp_penalty
    probs["wp_timeout"] = wp_timeout
    probs["dog_timeout_boost"] = probs["wp_timeout"] - probs["wp_penalty"]

    # Recommendation:
    # > 0.02  -> Call timeout (avoid penalty)
    # <= 0    -> Accept penalty
    # (0,0.02]-> Toss Up
    def _recommend(boost: float) -> str:
        if boost > 0.02:
            return "Recommendation: Call a timeout (avoid penalty)"
        elif boost <= 0:
            return "Recommendation: Accept delay of game penalty"
        else:
            return "Recommendation: Toss Up"

    probs["recommendation"] = probs["dog_timeout_boost"].apply(_recommend)
    return probs


def make_delay_vs_timeout_table(probs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "posteam",
        "defteam",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "wp_penalty",
        "wp_timeout",
        "dog_timeout_boost",
        "recommendation",
    ]
    existing = [c for c in cols if c in probs.columns]
    table = probs[existing].copy()

    if "dog_timeout_boost" in table.columns:
        table = table.sort_values("dog_timeout_boost", ascending=False)

    return table


# --------------------------------------------------------------------------------
# 3. Streamlit UI
# --------------------------------------------------------------------------------

st.title("NFL Timeout Decision Model (Streamlit Demo)")

tab1, tab2 = st.tabs(["Timeout vs No Timeout", "Delay of Game vs Timeout"])

teams = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
    "DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA",
    "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
]

with tab1:
    st.header("Timeout vs No Timeout")

    col1, col2 = st.columns(2)

    with col1:
        posteam = st.selectbox("Offense (posteam)", teams, index=teams.index("MIN"))
        defteam = st.selectbox("Defense (defteam)", teams, index=teams.index("GB"))
        qtr = st.selectbox("Quarter", [1, 2, 3, 4], index=3)
        game_seconds_remaining = st.number_input("Game seconds remaining", min_value=0, max_value=3600, value=120)
        half_seconds_remaining = st.number_input("Half seconds remaining", min_value=0, max_value=1800, value=120)
        down = st.selectbox("Down", [1, 2, 3, 4], index=1)
        ydstogo = st.number_input("Yards to go", min_value=1, max_value=99, value=7)

    with col2:
        yardline_100 = st.number_input("Yardline_100 (0=own GL, 100=opp GL)", min_value=0, max_value=100, value=40)
        score_diff = st.number_input("Score differential (offense - defense)", min_value=-99, max_value=99, value=-3)
        posteam_timeouts_remaining = st.number_input("Offense timeouts remaining", min_value=0, max_value=3, value=3)
        defteam_timeouts_remaining = st.number_input("Defense timeouts remaining", min_value=0, max_value=3, value=3)
        play_type = st.selectbox("Expected play type", ["run", "pass", "qb_scramble"], index=1)
        clock_stop = st.selectbox("Is game clock currently stopped?", [0, 1], index=0)

    if st.button("Compute timeout recommendation"):
        situation = pd.DataFrame(
            {
                "qtr": [qtr],
                "game_seconds_remaining": [game_seconds_remaining],
                "half_seconds_remaining": [half_seconds_remaining],
                "score_differential": [score_diff],
                "down": [down],
                "ydstogo": [ydstogo],
                "yardline_100": [yardline_100],
                "posteam_timeouts_remaining": [posteam_timeouts_remaining],
                "defteam_timeouts_remaining": [defteam_timeouts_remaining],
                "posteam": [posteam],
                "defteam": [defteam],
                "play_type": [play_type],
                "clock_stop": [clock_stop],
            }
        )

        probs = add_timeout_probs(situation, xgb_model, FEATURE_COLUMNS)
        table = make_timeout_table(probs)

        st.subheader("Results")
        st.dataframe(
            table.style.format(
                {
                    "wp_no_timeout": "{:.3f}",
                    "wp_timeout": "{:.3f}",
                    "timeout_boost": "{:+.3f}",
                }
            )
        )

        rec = table["recommendation"].iloc[0]
        st.success(rec)

with tab2:
    st.header("Delay of Game vs Timeout")

    col1, col2 = st.columns(2)

    with col1:
        posteam2 = st.selectbox("Offense (posteam)", teams, index=teams.index("MIN"), key="dog_posteam")
        defteam2 = st.selectbox("Defense (defteam)", teams, index=teams.index("GB"), key="dog_defteam")
        qtr2 = st.selectbox("Quarter", [1, 2, 3, 4], index=2, key="dog_qtr")
        game_seconds_remaining2 = st.number_input("Game seconds remaining", min_value=0, max_value=3600, value=900, key="dog_gsr")
        half_seconds_remaining2 = st.number_input("Half seconds remaining", min_value=0, max_value=1800, value=900, key="dog_hsr")
        down2 = st.selectbox("Down", [1, 2, 3, 4], index=0, key="dog_down")
        ydstogo2 = st.number_input("Yards to go", min_value=1, max_value=99, value=5, key="dog_ytg")

    with col2:
        yardline_100_2 = st.number_input("Yardline_100 (0=own GL, 100=opp GL)", min_value=0, max_value=100, value=75, key="dog_y100")
        score_diff2 = st.number_input("Score differential (offense - defense)", min_value=-99, max_value=99, value=0, key="dog_sd")
        posteam_to2 = st.number_input("Offense timeouts remaining", min_value=0, max_value=3, value=3, key="dog_pto")
        defteam_to2 = st.number_input("Defense timeouts remaining", min_value=0, max_value=3, value=3, key="dog_dto")
        play_type2 = st.selectbox("Expected play type", ["run", "pass", "qb_scramble"], index=1, key="dog_play")
        clock_stop2 = st.selectbox("Is game clock currently stopped?", [0, 1], index=0, key="dog_clock")

    if st.button("Compare delay of game vs timeout"):
        dog_situation = pd.DataFrame(
            {
                "qtr": [qtr2],
                "game_seconds_remaining": [game_seconds_remaining2],
                "half_seconds_remaining": [half_seconds_remaining2],
                "score_differential": [score_diff2],
                "down": [down2],
                "ydstogo": [ydstogo2],
                "yardline_100": [yardline_100_2],
                "posteam_timeouts_remaining": [posteam_to2],
                "defteam_timeouts_remaining": [defteam_to2],
                "posteam": [posteam2],
                "defteam": [defteam2],
                "play_type": [play_type2],
                "clock_stop": [clock_stop2],
            }
        )

        dog_probs = add_delay_vs_timeout_probs(dog_situation, xgb_model, FEATURE_COLUMNS)
        dog_table = make_delay_vs_timeout_table(dog_probs)

        st.subheader("Results")
        st.dataframe(
            dog_table.style.format(
                {
                    "wp_penalty": "{:.3f}",
                    "wp_timeout": "{:.3f}",
                    "dog_timeout_boost": "{:+.3f}",
                }
            )
        )

        rec2 = dog_table["recommendation"].iloc[0]
        st.success(rec2)
