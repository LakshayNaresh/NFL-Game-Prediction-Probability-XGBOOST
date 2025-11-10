import argparse
import pandas as pd
import joblib
import numpy as np
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--home", required=True, help="Home team (abbr, e.g. KC)")
parser.add_argument("--away", required=True, help="Away team (abbr, e.g. SF)")
args = parser.parse_args()

# Load model
model = joblib.load("model_xgb.pkl")

# Load play-by-play data (expects in src/)
files = glob.glob(os.path.join("src", "play_by_play_*.parquet"))
pbp = pd.concat([pd.read_parquet(f) for f in files])

# Compute simple team stats (expand later with rolling windows)
team_stats = (
    pbp.groupby(["posteam"])
    .agg(
        plays=("play_id", "count"),
        yards=("yards_gained", "mean"),
        epa=("epa", "mean"),
        success=("success", "mean"),
    )
    .reset_index()
)

# Get home and away team stats
home = team_stats[team_stats.posteam == args.home]
away = team_stats[team_stats.posteam == args.away]

if home.empty or away.empty:
    raise ValueError("One of the teams has no data in PBP files!")

# Build feature vector = home - away
features = home.iloc[0, 1:].values - away.iloc[0, 1:].values
X_matchup = features.reshape(1, -1)

# Predict probabilities
proba = model.predict_proba(X_matchup)[0]
print(f"{args.home} win probability: {proba[1]:.2%}")
print(f"{args.away} win probability: {proba[0]:.2%}")
