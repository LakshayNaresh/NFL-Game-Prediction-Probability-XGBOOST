
import subprocess, sys
from pathlib import Path
import yaml

if __name__ == "__main__":
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yml"))
    start = cfg.get("start_season", 2023)
    end = cfg.get("end_season", 2025)
    subprocess.run([sys.executable, "src/fetch_data.py", "--start", str(start), "--end", str(end)], check=True)
    subprocess.run([sys.executable, "src/build_features.py"], check=True)
    subprocess.run([sys.executable, "src/freeze_dataset.py"], check=True)
    subprocess.run([sys.executable, "src/train.py"], check=True)
