import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(r"C:\Users\ip471\Documents\year 2 research project")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "XGBoost"
OUTPUT_DIR = BASE_DIR / "outputs"

SYNTH_DIR = MODEL_DIR / "synthetic_training"
REAL_DIR = MODEL_DIR / "real_training"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

BASE_FEATURES = ["S", "K", "T", "r", "sigma", "maturity_days"]


def add_features(df):
    df = df.copy()
    df["log_moneyness"] = np.log(df["S"] / df["K"])
    df["sqrt_T"] = np.sqrt(df["T"])
    return df


FEATURES = BASE_FEATURES + ["log_moneyness", "sqrt_T"]


def build_model():
    return XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def train_model(df):
    df = add_features(df)

    X = df[FEATURES].values
    y = np.log1p(df["target_price"].values)

    model = build_model()
    model.fit(X, y)

    return model


def predict(model, df):
    df = add_features(df)

    X = df[FEATURES].values
    preds = model.predict(X)

    return np.expm1(preds)


def run():

    synth_df = pd.read_csv(DATA_DIR / "synthetic_training_data.csv")
    real_df = pd.read_csv(DATA_DIR / "real_training_data.csv")
    test_df = pd.read_csv(DATA_DIR / "real_test_data.csv")

    all_results = []
    metrics_rows = []

    for ticker in TICKERS:
        for option_type in ["call", "put"]:

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


            if len(synth_sub) > 100:

                model = train_model(synth_sub)

                model_path = SYNTH_DIR / f"{ticker}_{option_type}.pkl"
                joblib.dump(model, model_path)

                preds = predict(model, test_sub)

                metrics = evaluate(test_sub["target_price"].values, preds)

                metrics_rows.append({
                    "model": f"{ticker}_{option_type}_synthetic",
                    **metrics
                })

                tmp = test_sub.copy()
                tmp["model"] = f"{ticker}_{option_type}_synthetic"
                tmp["pred"] = preds
                all_results.append(tmp)


            if len(real_sub) > 100:

                model = train_model(real_sub)

                model_path = REAL_DIR / f"{ticker}_{option_type}.pkl"
                joblib.dump(model, model_path)

                preds = predict(model, test_sub)

                metrics = evaluate(test_sub["target_price"].values, preds)

                metrics_rows.append({
                    "model": f"{ticker}_{option_type}_real",
                    **metrics
                })

                tmp = test_sub.copy()
                tmp["model"] = f"{ticker}_{option_type}_real"
                tmp["pred"] = preds
                all_results.append(tmp)

    final_preds = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    final_preds.to_csv(OUTPUT_DIR / "XGBoost_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "XGBoost_metrics.csv", index=False)

    print(final_preds.head())
    print(metrics_df)


if __name__ == "__main__":
    run()