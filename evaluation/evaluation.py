import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



BASE_DIR = r"C:\Users\ip471\Documents\year 2 research project"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FILES = {
    "BlackScholes": "BlackScholes_predictions.csv",
    "MonteCarlo": "MonteCarlo_predictions.csv",
    "RF_real": "RandomForest_real_predictions.csv",
    "RF_synth": "RandomForest_synth_predictions.csv",
    "XGB_real": "XGBoost_real_predictions.csv",
    "XGB_synth": "XGBoost_synth_predictions.csv",
    "CatBoost_real": "CatBoost_real_predictions.csv",
    "CatBoost_synth": "CatBoost_synth_predictions.csv",
    "LSTM_real": "LSTM_real_predictions.csv",
    "LSTM_synth": "LSTM_synth_predictions.csv",
}


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def safe_load(file):
    path = os.path.join(OUTPUT_DIR, file)
    if not os.path.exists(path):
        print(f"[WARNING] Missing file: {file}")
        return None
    return pd.read_csv(path)




def evaluate_all_models():
    results = []

    for model_name, file in FILES.items():

        df = safe_load(file)
        if df is None:
            continue

        if "predicted_price" in df.columns:
            y_pred = df["predicted_price"]
        elif "pred" in df.columns:
            y_pred = df["pred"]
        else:
            raise ValueError(f"No prediction column in {file}")

        y_true = df["target_price"]

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        results.append({
            "model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "category": "all"
        })

    return pd.DataFrame(results)




def evaluate_ml_only(df_all):
    ml_df = df_all[df_all["model"].isin([
        "RF_real", "RF_synth",
        "XGB_real", "XGB_synth",
        "CatBoost_real", "CatBoost_synth",
        "LSTM_real", "LSTM_synth"
    ])]

    real = []
    synth = []

    for model in ml_df["model"].unique():

        sub = ml_df[ml_df["model"] == model]

        rmse, mae, r2 = compute_metrics(sub["target_price"], sub["predicted_price"])

        row = {
            "model": model,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        if "real" in model:
            row["regime"] = "real"
            real.append(row)
        else:
            row["regime"] = "synthetic"
            synth.append(row)

    return pd.DataFrame(real), pd.DataFrame(synth)



def domain_shift_analysis(real_df, synth_df):

    results = []

    base_models = ["RF", "XGB", "CatBoost", "LSTM"]

    for m in base_models:

        r = real_df[real_df["model"].str.contains(m)]["RMSE"].values
        s = synth_df[synth_df["model"].str.contains(m)]["RMSE"].values

        if len(r) == 0 or len(s) == 0:
            continue

        delta = float(r[0] - s[0])

        results.append({
            "model": m,
            "RMSE_real": float(r[0]),
            "RMSE_synth": float(s[0]),
            "Delta (Real - Synth)": delta
        })

    return pd.DataFrame(results)


def evaluate_ml_only_from_files():
    ml_files = {
        "RF_real": "RandomForest_real_predictions.csv",
        "RF_synth": "RandomForest_synth_predictions.csv",
        "XGB_real": "XGBoost_real_predictions.csv",
        "XGB_synth": "XGBoost_synth_predictions.csv",
        "CatBoost_real": "CatBoost_real_predictions.csv",
        "CatBoost_synth": "CatBoost_synth_predictions.csv",
        "LSTM_real": "LSTM_real_predictions.csv",
        "LSTM_synth": "LSTM_synth_predictions.csv",
    }

    real, synth = [], []

    for model, file in ml_files.items():
        df = safe_load(file)
        if df is None:
            continue

        y_true = df["target_price"]

        y_pred = (
            df["predicted_price"]
            if "predicted_price" in df.columns
            else df["pred"]
        )

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        row = {
            "model": model,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        if "real" in model:
            row["regime"] = "real"
            real.append(row)
        else:
            row["regime"] = "synthetic"
            synth.append(row)

    return pd.DataFrame(real), pd.DataFrame(synth)

def main():

    print("\n==============================")
    print("RUNNING FULL EVALUATION")
    print("==============================\n")

    all_results = evaluate_all_models()
    print("\n[A] Overall Performance:\n")
    print(all_results.sort_values("RMSE"))

    real_ml, synth_ml = evaluate_ml_only_from_files()

    print("\n[B] ML Real-Trained Models:\n")
    print(real_ml)

    print("\n[B] ML Synthetic-Trained Models:\n")
    print(synth_ml)


    shift_df = domain_shift_analysis(real_ml, synth_ml)

    print("\n[C] Domain Shift (Synthetic → Real):\n")
    print(shift_df)


    all_results.to_csv(os.path.join(OUTPUT_DIR, "A_overall_results.csv"), index=False)
    real_ml.to_csv(os.path.join(OUTPUT_DIR, "B_ml_real.csv"), index=False)
    synth_ml.to_csv(os.path.join(OUTPUT_DIR, "B_ml_synth.csv"), index=False)
    shift_df.to_csv(os.path.join(OUTPUT_DIR, "C_domain_shift.csv"), index=False)

    print("\nSaved all evaluation outputs in /outputs/")


if __name__ == "__main__":
    main()