# NFL-Game-Prediction-Probability-XGBOOST
Machine learning model for NFL game outcome prediction — utilizes XGBoost with play-by-play, advanced team stats, and momentum directionality to forecast win probabilities.

NFL Win Predictor — Advanced (2023 → Week 10, 2025)

**Author:** Lakshay Naresh  
**Last Updated:** Week 10 – 2025 NFL Season  

---

### Overview
Machine-learning pipeline that predicts NFL game outcomes using play-by-play data, advanced team statistics, and momentum directionality.

Built with:
- **nflverse** datasets (2023 – 2025)
- **Feature-rich, leakage-safe engineering**
- **XGBoost** classifier trained on historical performance
- **Automatic weekly updates** to stay current each NFL week  

---

## Key Features

| Category | Description |
|-----------|-------------|
| **Efficiency** | EPA / play & success rate (offense & defense) |
| **Situational** | Red-zone EPA / success rates; early vs late downs |
| **Play-Type Balance** | Pass / run rates; sack / INT / fumble rates |
| **Explosiveness** | 20 + yard plays (off & def) |
| **Momentum / Form** | Rolling windows (3, 6, 10 games) with directionality |
| **Rest & Context** | Rest days, home-field flag, divisional indicator |
| **Extendable** | Add roof, surface, weather, QB identity |

---

## Quick Start

```bash
# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch data (2023 – 2025)
python src/fetch_data.py --start 2023 --end 2025

# Build and freeze dataset (default: Week 10 2025)
python src/build_features.py
python src/freeze_dataset.py

# Train the model
python src/train.py

# Predict via CLI
python src/predict.py --home DAL --away PHI --season 2025 --week 11

# Optional Streamlit UI
streamlit run app/app.py
```
---

## Methodology

- **Leakage-Safe Rolling Windows** – uses past 3/6/10 games (shifted forward)  
- **Directionality Metrics** – captures team trend momentum  
- **Matchup Differentials** – home − away features to learn relative strength  
- **Rest-Day Feature** – adjusts for travel / recovery fatigue  
- **Continuous Retraining** – integrates new weeks automatically  

---

## Example Usage

```bash
python src/predict.py --home KC --away SF --season 2025 --week 11
```

```
KC win probability: 68.3 %
SF win probability: 31.7 %
```

---

## Project Structure

```
nfl-win-prob-advanced/
│
├── src/
│   ├── build_features.py
│   ├── fetch_data.py
│   ├── freeze_dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── play_by_play_*.parquet
│   └── stats_team_week_*.parquet
│
├── data/
│   └── features.csv
│
├── models/
│   └── model_xgb.pkl
│
├── config.yml
├── requirements.txt
└── README.md
```

---

## Roadmap
- Player-level EPA per dropback / target share  
- Injury & roster tracking  
- Coaching tendencies (Pass Rate Over Expected, blitz %)  
- Environment features (weather, surface, travel)  
- Automated feature importance & tuning  

---

## Data Sources
- [nflverse Data Releases](https://github.com/nflverse/nflverse-data/releases)  
- [nfl_data_py](https://github.com/nflverse/nfl_data_py)  
- [FTN Charting Data](https://www.ftnfantasy.com/)  

---

**Lakshay Naresh**  
