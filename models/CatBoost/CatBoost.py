import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor

# =========================
# PATHS
# =========================

BASE_DIR = Path(r"C:\Users\ip471\Documents\year 2 research project")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "CatBoost"
OUTPUT_DIR = BASE_DIR / "outputs"

SYNTH_DIR = MODEL_DIR / "synthetic_training"
REAL_DIR = MODEL_DIR / "real_training"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================

TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

FEATURES = ["S", "K", "T", "r", "sigma", "maturity_days"]

# =========================
# MODEL
# =========================

def build_model():
    return CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        verbose=100,
        random_seed=42
    )

# =========================
# TRAINING
# =========================

def train_model(df):
    X = df[FEATURES]
    y = df["target_price"]

    model = build_model()
    model.fit(X, y)

    return model

# =========================
# PREDICTION
# =========================

def predict(model, df):
    X = df[FEATURES]
    preds = model.predict(X)
    return np.array(preds).flatten()

# =========================
# EVALUATION
# =========================

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# =========================
# MAIN PIPELINE
# =========================

def run():

    synth_df = pd.read_csv(DATA_DIR / "synthetic_training_data.csv")
    real_df = pd.read_csv(DATA_DIR / "real_training_data.csv")
    test_df = pd.read_csv(DATA_DIR / "real_test_data.csv")

    all_results = []
    metrics_rows = []

    for ticker in TICKERS:
        for option_type in ["call", "put"]:

            # -------------------------
            # FILTER DATA
            # -------------------------
            synth_sub = synth_df[
                (synth_df["ticker"] == ticker) &
                (synth_df["option_type"] == option_type)
            ]

            real_sub = real_df[
                (real_df["ticker"] == ticker) &
                (real_df["option_type"] == option_type)
            ]

            test_sub = test_df[
                (test_df["ticker"] == ticker) &
                (test_df["option_type"] == option_type)
            ]

            # =========================
            # SYNTHETIC MODEL
            # =========================
            if len(synth_sub) > 100:

                model = train_model(synth_sub)

                model_path = SYNTH_DIR / f"{ticker}_{option_type}.cbm"
                model.save_model(str(model_path))

                preds = predict(model, test_sub)
                y_true = test_sub["target_price"].values

                metrics = evaluate(y_true, preds)

                metrics_rows.append({
                    "model": f"{ticker}_{option_type}_synthetic",
                    **metrics
                })

                tmp = test_sub.copy()
                tmp["model"] = f"{ticker}_{option_type}_synthetic"
                tmp["pred"] = preds

                all_results.append(tmp)

            # =========================
            # REAL MODEL
            # =========================
            if len(real_sub) > 100:

                model = train_model(real_sub)

                model_path = REAL_DIR / f"{ticker}_{option_type}.cbm"
                model.save_model(str(model_path))

                preds = predict(model, test_sub)
                y_true = test_sub["target_price"].values

                metrics = evaluate(y_true, preds)

                metrics_rows.append({
                    "model": f"{ticker}_{option_type}_real",
                    **metrics
                })

                tmp = test_sub.copy()
                tmp["model"] = f"{ticker}_{option_type}_real"
                tmp["pred"] = preds

                all_results.append(tmp)

    # =========================
    # SAVE OUTPUTS
    # =========================

    final_preds = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    final_preds.to_csv(OUTPUT_DIR / "CatBoost_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "CatBoost_metrics.csv", index=False)

    print(final_preds.head())
    print(metrics_df)

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run()