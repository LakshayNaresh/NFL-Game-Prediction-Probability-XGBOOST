
from pathlib import Path
import yaml
import pandas as pd

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"

if __name__ == "__main__":
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yml"))
    name = cfg.get("dataset_name","dataset_export")
    feats = pd.read_parquet(ART_DIR / "features.parquet")
    # keep label + ids + diff_* features for modeling
    cols = ["season","week","game_id","home_team","away_team","home_win"] + [c for c in feats.columns if c.startswith("diff_")]
    dataset = feats[cols].copy()
    dataset.to_parquet(ART_DIR / f"{name}.parquet", index=False)
    dataset.to_csv(ART_DIR / f"{name}.csv", index=False)
    print(f"Wrote {ART_DIR / f'{name}.parquet'} and .csv with {dataset.shape[0]} rows, {dataset.shape[1]} cols.")
