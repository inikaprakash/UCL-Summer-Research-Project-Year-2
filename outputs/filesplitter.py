import os
import pandas as pd


BASE_DIR = r"C:\Users\ip471\Documents\year 2 research project\outputs"

MODELS = ["LSTM", "RandomForest", "XGBoost", "CatBoost"]


def split_file(file_path, model_name):
    df = pd.read_csv(file_path)


    if "model" not in df.columns:
        raise ValueError(f"'model' column missing in {file_path}")


    df["model"] = df["model"].astype(str)


    real_df = df[~df["model"].str.contains("synthetic", case=False, na=False)]
    synth_df = df[df["model"].str.contains("synthetic", case=False, na=False)]


    real_path = file_path.replace("_predictions.csv", "_real_predictions.csv")
    synth_path = file_path.replace("_predictions.csv", "_synth_predictions.csv")


    real_df.to_csv(real_path, index=False)
    synth_df.to_csv(synth_path, index=False)

    print(f"\n{model_name}")
    print(f"  Real:  {len(real_df)} rows -> {real_path}")
    print(f"  Synth: {len(synth_df)} rows -> {synth_path}")



def main():
    for model in MODELS:

        file_path = os.path.join(BASE_DIR, f"{model}_predictions.csv")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        split_file(file_path, model)


if __name__ == "__main__":
    main()