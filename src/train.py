import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import glob
import os



# Look for all play_by_play parquet files inside src/
files = glob.glob(os.path.join("src", "play_by_play_*.parquet"))
print("Found files:", files)

if not files:
    raise FileNotFoundError("No play_by_play parquet files found in src/")

pbp = pd.concat([pd.read_parquet(f) for f in files])



# 2. Build schedules (final score per game)
schedules = (
    pbp.groupby(["game_id", "season", "week", "home_team", "away_team"])
    .agg({"home_score": "max", "away_score": "max"})
    .reset_index()
)
schedules["home_win"] = (schedules["home_score"] > schedules["away_score"]).astype(int)

# 3. Simple team stats (expand with rolling EPA etc. later)
team_stats = (
    pbp.groupby(["game_id", "posteam"])
    .agg(
        plays=("play_id", "count"),
        yards=("yards_gained", "sum"),
        epa=("epa", "mean"),
        success=("success", "mean"),
    )
    .reset_index()
)

# 4. Build matchup dataset (home-away differences)
dataset = []
for _, row in schedules.iterrows():
    gid = row["game_id"]
    home = team_stats[(team_stats.game_id == gid) & (team_stats.posteam == row.home_team)]
    away = team_stats[(team_stats.game_id == gid) & (team_stats.posteam == row.away_team)]
    if home.empty or away.empty:
        continue
    diff = (home.iloc[0, 2:].values - away.iloc[0, 2:].values)  # features
    dataset.append(np.concatenate([diff, [row.home_win]]))

X = np.array([d[:-1] for d in dataset])
y = np.array([d[-1] for d in dataset])

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 6. XGBoost with tuning
params = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
grid = GridSearchCV(model, params, scoring="neg_log_loss", cv=3, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# 7. Evaluate
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, (y_pred_proba > 0.5)))
print("LogLoss:", log_loss(y_test, y_pred_proba))
print("AUC:", roc_auc_score(y_test, y_pred_proba))

# 8. Save model + feature importance
joblib.dump(best_model, "model_xgb.pkl")

xgb.plot_importance(best_model)
plt.tight_layout()
plt.savefig("feature_importance.png")

print("✅ Training complete. Model saved as model_xgb.pkl")
