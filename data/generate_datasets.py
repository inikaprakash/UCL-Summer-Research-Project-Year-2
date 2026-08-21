import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from pathlib import Path



BASE_DIR = Path(r"C:\Users\ip471\Documents\year 2 research project")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "MSFT",
    "AAPL",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA"
]

RISK_FREE_RATE = 0.04

MATURITIES = [7, 30, 90, 240, 365]

STRIKE_MULTIPLIERS = [
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.20
]


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
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":

        return (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )



def generate_synthetic_training_data():

    rows = []

    print("\nGenerating synthetic dataset...")

    for ticker in TICKERS:

        print(f"Processing {ticker}")

        df = yf.download(
            ticker,
            start="2014-01-01",
            end="2025-01-01",
            auto_adjust=True,
            progress=False
        )

        if len(df) < 100:
            continue

        closes = df["Close"]

        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]

        returns = np.log(
            closes / closes.shift(1)
        )

        realized_vol = (
            returns
            .rolling(30)
            .std()
            * np.sqrt(252)
        )

        for date in closes.index:

            S = closes.loc[date]
            sigma = realized_vol.loc[date]

            if pd.isna(S):
                continue

            if pd.isna(sigma):
                continue

            if sigma <= 0:
                continue

            S = float(S)
            sigma = float(sigma)

            for option_type in ["call", "put"]:

                for maturity_days in MATURITIES:

                    T = maturity_days / 365

                    for strike_mult in STRIKE_MULTIPLIERS:

                        K = S * strike_mult

                        price = black_scholes_price(
                            S,
                            K,
                            T,
                            RISK_FREE_RATE,
                            sigma,
                            option_type
                        )

                        rows.append({
                            "ticker": ticker,
                            "date": date,
                            "S": S,
                            "K": K,
                            "T": T,
                            "r": RISK_FREE_RATE,
                            "sigma": sigma,
                            "option_type": option_type,
                            "target_price": price
                        })

    synthetic_df = pd.DataFrame(rows)

    file_path = (
        DATA_DIR /
        "synthetic_training_data.csv"
    )

    synthetic_df.to_csv(
        file_path,
        index=False
    )

    print(
        f"\nSynthetic rows: "
        f"{len(synthetic_df):,}"
    )

    return synthetic_df



def download_real_option_data():

    rows = []

    print("\nDownloading option chains...")

    today = pd.Timestamp.today()

    for ticker in TICKERS:

        try:

            print(f"\n{ticker}")

            tk = yf.Ticker(ticker)

            spot = (
                tk.history(period="1d")
                ["Close"]
                .iloc[-1]
            )

            expiries = tk.options

            for expiry in expiries:

                try:

                    expiry_dt = pd.to_datetime(expiry)

                    T = (
                        expiry_dt - today
                    ).days / 365

                    if T <= 0:
                        continue

                    chain = tk.option_chain(expiry)

                    option_tables = [
                        ("call", chain.calls),
                        ("put", chain.puts)
                    ]

                    for option_type, table in option_tables:

                        for _, row in table.iterrows():

                            strike = row["strike"]

                            market_price = row[
                                "lastPrice"
                            ]

                            iv = row[
                                "impliedVolatility"
                            ]

                            if pd.isna(strike):
                                continue

                            if pd.isna(market_price):
                                continue

                            if pd.isna(iv):
                                continue

                            if market_price <= 0:
                                continue

                            rows.append({

                                "ticker": ticker,

                                "S": float(spot),

                                "K": float(strike),

                                "T": float(T),

                                "r": RISK_FREE_RATE,

                                "sigma": float(iv),

                                "option_type": option_type,

                                "target_price":
                                    float(market_price)

                            })

                except Exception as e:

                    print(
                        f"Expiry error "
                        f"{expiry}: {e}"
                    )

        except Exception as e:

            print(
                f"Ticker error "
                f"{ticker}: {e}"
            )

    return pd.DataFrame(rows)


def create_real_train_test():

    real_df = download_real_option_data()

    print(
        f"\nTotal real rows: "
        f"{len(real_df):,}"
    )

    real_df["strata"] = (
        real_df["ticker"]
        + "_"
        + real_df["option_type"]
    )

    train_df, test_df = train_test_split(

        real_df,

        test_size=0.20,

        random_state=42,

        stratify=real_df["strata"]

    )

    train_df = train_df.drop(
        columns=["strata"]
    )

    test_df = test_df.drop(
        columns=["strata"]
    )

    train_path = (
        DATA_DIR /
        "real_training_data.csv"
    )

    test_path = (
        DATA_DIR /
        "real_test_data.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print(
        f"\nTrain rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    print("\nTrain Distribution")

    print(
        train_df.groupby(
            ["ticker", "option_type"]
        ).size()
    )

    print("\nTest Distribution")

    print(
        test_df.groupby(
            ["ticker", "option_type"]
        ).size()
    )



if __name__ == "__main__":

    generate_synthetic_training_data()

    create_real_train_test()

    print(
        "\nDatasets created successfully."
    )