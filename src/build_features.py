import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

pbp_files = [
    DATA_DIR / "play_by_play_2023.parquet",
    DATA_DIR / "play_by_play_2024.parquet",
    DATA_DIR / "play_by_play_2025.parquet",
]

teamstat_files = [
    DATA_DIR / "stats_team_week_2023.parquet",
    DATA_DIR / "stats_team_week_2024.parquet",
    DATA_DIR / "stats_team_week_2025.parquet",
]

print("📂 Loading play-by-play and team stats...")
pbp = pd.concat([pd.read_parquet(f) for f in pbp_files])
team_stats = pd.concat([pd.read_parquet(f) for f in teamstat_files])

# --- Step 1. Aggregate play-by-play to team-game level ---
print("🔧 Aggregating team-game stats from pbp...")
team_games = []

for side in ["home", "away"]:
    df = pbp.groupby(["game_id", "season", "week", f"{side}_team"]).mean(numeric_only=True).reset_index()
    df = df.rename(columns={f"{side}_team": "team"})
    df["is_home"] = 1 if side == "home" else 0
    team_games.append(df)

team_games = pd.concat(team_games, ignore_index=True)

# --- Step 2. Merge team weekly advanced stats ---
print("➕ Merging team weekly stats...")
team_stats = team_stats.rename(columns={"team": "team", "week": "week", "season": "season"})
merged = team_games.merge(
    team_stats,
    on=["season", "week", "team"],
    how="left",
    suffixes=("", "_adv")
)

# --- Step 3. Rolling averages & deltas ---
print("📈 Building rolling averages & deltas...")
merged = merged.sort_values(["team", "season", "week"])

numeric_cols = merged.select_dtypes(include=[np.number]).columns.difference(
    ["season", "week", "is_home"]
)

rolling_features = []
for window in [1, 3, 5]:
    rolled = (
        merged.groupby("team")[numeric_cols]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index()
    )
    rolled = rolled.drop(columns="team")
    rolled.columns = [f"{c}_L{window}" for c in rolled.columns]
    rolling_features.append(rolled)

season_avg = (
    merged.groupby(["team", "season"]).expanding()[numeric_cols].mean().reset_index()
)
season_avg = season_avg.drop(columns=["team", "season"])
season_avg.columns = [f"{c}_STD" for c in season_avg.columns]

# Merge back
merged = pd.concat([merged.reset_index(drop=True)] + rolling_features + [season_avg], axis=1)

# --- Step 4. Directionality deltas ---
for c in numeric_cols:
    if f"{c}_L3" in merged.columns and f"{c}_STD" in merged.columns:
        merged[f"{c}_delta_L3"] = merged[f"{c}_L3"] - merged[f"{c}_STD"]
    if f"{c}_L5" in merged.columns and f"{c}_STD" in merged.columns:
        merged[f"{c}_delta_L5"] = merged[f"{c}_L5"] - merged[f"{c}_STD"]

# --- Step 5. Build matchup dataset ---
print("⚔️ Building matchup dataset...")
matchups = pbp[["game_id", "season", "week", "home_team", "away_team"]].drop_duplicates()

home_stats = merged[merged["is_home"] == 1].drop(columns=["is_home"])
away_stats = merged[merged["is_home"] == 0].drop(columns=["is_home"])

df = matchups.merge(home_stats, left_on=["game_id", "home_team"], right_on=["game_id", "team"])
df = df.merge(away_stats, left_on=["game_id", "away_team"], right_on=["game_id", "team"], suffixes=("_home", "_away"))

# --- Step 6. Differentials ---
for c in numeric_cols:
    for feat in ["L1", "L3", "L5", "STD", "delta_L3", "delta_L5"]:
        h = f"{c}_{feat}_home"
        a = f"{c}_{feat}_away"
        if h in df.columns and a in df.columns:
            df[f"{c}_{feat}_diff"] = df[h] - df[a]

# --- Step 7. Add results ---
print("🏆 Adding results...")
df["result"] = (pbp.groupby("game_id").last()["home_score"] >
                pbp.groupby("game_id").last()["away_score"]).astype(int).reindex(df["game_id"]).values

# --- Save ---
out_path = DATA_DIR.parent / "data" / "features.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"✅ Features saved to {out_path}")
