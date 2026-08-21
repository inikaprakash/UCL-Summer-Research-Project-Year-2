import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

BASE_DIR = Path(r"C:\Users\ip471\Documents\year 2 research project")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "LSTM"
OUTPUT_DIR = BASE_DIR / "outputs"

SYNTH_DIR = MODEL_DIR / "synthetic_training"
REAL_DIR = MODEL_DIR / "real_training"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

FEATURES = ["S", "K", "T", "r", "sigma", "maturity_days"]


def build_model():
    model = Sequential([
        LSTM(64, input_shape=(len(FEATURES), 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_scalers(df):
    X = df[FEATURES].values
    y = df["target_price"].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_model(df):
    X, y, scaler_X, scaler_y = prepare_scalers(df)

    model = build_model()
    model.fit(X, y, epochs=5, batch_size=256, verbose=0)

    return model, scaler_X, scaler_y


def predict(model, scaler_X, scaler_y, df):
    X = df[FEATURES].values
    X = scaler_X.transform(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    preds = model.predict(X, verbose=0)
    preds = scaler_y.inverse_transform(preds)

    return preds.flatten()


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
                (synth_df["ticker"] == ticker)
                & (synth_df["option_type"] == option_type)
            ]

            real_sub = real_df[
                (real_df["ticker"] == ticker)
                & (real_df["option_type"] == option_type)
            ]

            test_sub = test_df[
                (test_df["ticker"] == ticker)
                & (test_df["option_type"] == option_type)
            ]

            if len(synth_sub) > 100:

                model, scaler_X, scaler_y = train_model(synth_sub)

                model_path = SYNTH_DIR / f"{ticker}_{option_type}.keras"
                scaler_x_path = SYNTH_DIR / f"{ticker}_{option_type}_scaler_X.pkl"
                scaler_y_path = SYNTH_DIR / f"{ticker}_{option_type}_scaler_y.pkl"

                model.save(model_path)
                joblib.dump(scaler_X, scaler_x_path)
                joblib.dump(scaler_y, scaler_y_path)

                preds = predict(model, scaler_X, scaler_y, test_sub)

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

            if len(real_sub) > 100:

                model, scaler_X, scaler_y = train_model(real_sub)

                model_path = REAL_DIR / f"{ticker}_{option_type}.keras"
                scaler_x_path = REAL_DIR / f"{ticker}_{option_type}_scaler_X.pkl"
                scaler_y_path = REAL_DIR / f"{ticker}_{option_type}_scaler_y.pkl"

                model.save(model_path)
                joblib.dump(scaler_X, scaler_x_path)
                joblib.dump(scaler_y, scaler_y_path)

                preds = predict(model, scaler_X, scaler_y, test_sub)

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

    final_preds = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    final_preds.to_csv(OUTPUT_DIR / "LSTM_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "LSTM_metrics.csv", index=False)

    print(final_preds.head())
    print(metrics_df)


if __name__ == "__main__":
    run()