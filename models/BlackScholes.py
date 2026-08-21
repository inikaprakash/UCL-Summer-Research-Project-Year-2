import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


BASE_DIR = Path(
    r"C:\Users\ip471\Documents\year 2 research project"
)

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def black_scholes_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type
):
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0:
        return np.nan

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (
        sigma * np.sqrt(T)
    )

    d2 = (
        d1
        - sigma * np.sqrt(T)
    )

    if option_type == "call":
        return (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


def main():

    test_file = DATA_DIR / "real_test_data.csv"

    df = pd.read_csv(test_file)

    df["predicted_price"] = df.apply(
        lambda row: black_scholes_price(
            S=row["S"],
            K=row["K"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
            option_type=row["option_type"]
        ),
        axis=1
    )

    df = df.dropna(
        subset=["predicted_price"]
    )

    y_true = df["target_price"]
    y_pred = df["predicted_price"]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / np.maximum(y_true, 1e-8)
            )
        )
        * 100
    )

    metrics_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "MAPE (%)",
            "R²"
        ],
        "Value": [
            mae,
            rmse,
            mape,
            r2
        ]
    })

    predictions_file = (
        OUTPUT_DIR
        / "BlackScholes_predictions.csv"
    )

    metrics_file = (
        OUTPUT_DIR
        / "BlackScholes_metrics.csv"
    )

    df.to_csv(
        predictions_file,
        index=False
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print("\nBlack-Scholes Results")
    print("-" * 40)
    print(f"MAE       : {mae:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"MAPE (%)  : {mape:.4f}")
    print(f"R²        : {r2:.6f}")

    print(
        f"\nPredictions saved to:\n{predictions_file}"
    )

    print(
        f"\nMetrics saved to:\n{metrics_file}"
    )


if __name__ == "__main__":
    main()