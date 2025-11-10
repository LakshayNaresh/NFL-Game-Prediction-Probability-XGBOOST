
import argparse
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def fetch(start: int, end: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seasons = list(range(start, end + 1))
    print(f"Fetching seasons: {seasons}")
    sched = nfl.import_schedules(seasons)
    sched.to_parquet(DATA_DIR / "schedules.parquet", index=False)
    pbp = nfl.import_pbp_data(seasons, downcast=True, cache=True)
    # keep only needed columns
    cols = [
        "season","week","game_id","home_team","away_team","posteam","defteam",
        "pass","rush","play_type","epa","success","qb_hit","sack","interception",
        "touchdown","fumble_lost","air_yards","yards_gained","yardline_100",
        "down","ydstogo"
    ]
    pbp = pbp[[c for c in cols if c in pbp.columns]]
    pbp.to_parquet(DATA_DIR / "pbp.parquet", index=False)
    print("Saved schedules.parquet and pbp.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()
    fetch(args.start, args.end)
