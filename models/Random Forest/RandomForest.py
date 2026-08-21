import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(r"C:\Users\ip471\Documents\year 2 research project")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "Random Forest"
OUTPUT_DIR = BASE_DIR / "outputs"

SYNTH_DIR = MODEL_DIR / "synthetic_training"
REAL_DIR = MODEL_DIR / "real_training"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

FEATURES = ["S", "K", "T", "r", "sigma", "maturity_days"]


def build_model():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )


def prepare_data(df, scaler_X=None, fit_scaler=False):
    X = df[FEATURES].values
    y = df["target_price"].values

    if scaler_X is None:
        scaler_X = StandardScaler()

    if fit_scaler:
        X = scaler_X.fit_transform(X)
    else:
        X = scaler_X.transform(X)

    return X, y, scaler_X


def train_model(df):
    X, y, scaler_X = prepare_data(df, fit_scaler=True)

    model = build_model()
    model.fit(X, y)

    return model, scaler_X


def predict(model, scaler_X, df):
    X = df[FEATURES].values
    X = scaler_X.transform(X)
    return model.predict(X)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


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

                model, scaler_X = train_model(synth_sub)

                model_path = SYNTH_DIR / f"{ticker}_{option_type}.pkl"
                scaler_path = SYNTH_DIR / f"{ticker}_{option_type}_scaler_X.pkl"

                joblib.dump(model, model_path)
                joblib.dump(scaler_X, scaler_path)

                preds = predict(model, scaler_X, test_sub)

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

                model, scaler_X = train_model(real_sub)

                model_path = REAL_DIR / f"{ticker}_{option_type}.pkl"
                scaler_path = REAL_DIR / f"{ticker}_{option_type}_scaler_X.pkl"

                joblib.dump(model, model_path)
                joblib.dump(scaler_X, scaler_path)

                preds = predict(model, scaler_X, test_sub)

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

    final_preds.to_csv(OUTPUT_DIR / "RandomForest_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "RandomForest_metrics.csv", index=False)

    print(final_preds.head())
    print(metrics_df)


if __name__ == "__main__":
    run()