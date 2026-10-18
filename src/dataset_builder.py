import pandas as pd

from data_loader import download_market_data, TICKERS, BENCHMARK
from features import create_features, create_target

def build_ml_dataset():
    market_data = download_market_data()

    benchmark_data = market_data[BENCHMARK]

    datasets = []
    for ticker in TICKERS:
        stock_data = market_data[ticker]

        stock_features = create_features(stock_data)

        stock_features = create_target(
            stock_features,
            benchmark_data
        )

        stock_features["Ticker"] = ticker

        stock_features = stock_features.reset_index()

        datasets.append(stock_features)

    combined_data = pd.concat(
        datasets,
        ignore_index=True
    )

    columns = [
    "Date",
    "Ticker",
    "Daily_Return",
    "Momentum_5d",
    "Volatility_20d",
    "MA_Ratio_20d",
    "Future_Return_5d",
    "Benchmark_Future_Return_5d",
    "Target"
    ]

    combined_data = combined_data[columns]
    combined_data = combined_data.dropna().copy()

    combined_data["Target"] = (
        combined_data["Target"].astype(int)
    )

    combined_data.to_csv(
    "data/ml_dataset.csv",
    index=False
    )

    return combined_data


if __name__ == "__main__":
    dataset = build_ml_dataset()

    print(dataset.head())

    print("\nDataset shape:")
    print(dataset.shape)

    print("\nStocks:")
    print(dataset["Ticker"].unique())

    print("\nTarget distribution:")
    print(dataset["Target"].value_counts(normalize=True))