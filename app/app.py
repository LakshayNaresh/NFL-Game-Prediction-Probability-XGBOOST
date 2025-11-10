
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

st.title("NFL Win Predictor (2023 → cutoff)")

@st.cache_data
def load_feats():
    return pd.read_parquet(ART_DIR / "features.parquet")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")

feats = load_feats()
teams = sorted(set(feats["home_team"]).union(set(feats["away_team"])))

col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams, index=teams.index("DAL") if "DAL" in teams else 0)
away = col2.selectbox("Away team", teams, index=teams.index("PHI") if "PHI" in teams else 1)
season = st.number_input("Season", min_value=int(feats["season"].min()), max_value=int(feats["season"].max()), value=int(feats["season"].max()), step=1)
week = st.number_input("Week", min_value=int(feats["week"].min()), max_value=int(feats["week"].max()), value=int(feats["week"].max()), step=1)

if st.button("Predict"):
    df = feats
    q = (df["home_team"]==home) & (df["away_team"]==away) & (df["season"]==season) & (df["week"]==week)
    cand = df[q]
    if cand.empty:
        st.error("Matchup not found in features. Make sure you've built features through this week.")
    else:
        X = cand.iloc[-1][[c for c in df.columns if c.startswith("diff_")]].to_frame().T.fillna(0.0)
        model = load_model()
        prob = float(model.predict_proba(X)[:,1][0])
        st.metric(label=f"P({home} beats {away})", value=f"{prob:.1%}")
        with st.expander("Feature snapshot"):
            st.dataframe(cand[[c for c in df.columns if c.startswith("diff_")]].T.rename(columns={cand.index[-1]:"value"}))
