import pandas as pd

from data_loader import download_market_data


def create_features(stock_data):
    features = stock_data.copy()

    features["Daily_Return"] = features["Close"].pct_change(fill_method=None)

    features["Momentum_5d"] = features["Close"].pct_change(
        periods=5,
        fill_method=None
    )

    features["Volatility_20d"] = (
        features["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    features["MA_20"] = (
        features["Close"]
        .rolling(window=20)
        .mean()
    )

    features["MA_Ratio_20d"] = (
        features["Close"] / features["MA_20"] - 1
    )

    return features

def create_target(features, benchmark_data):
    features = features.copy()

    features["Future_Return_5d"] = (
        features["Close"].shift(-5) / features["Close"] - 1
    )

    benchmark_future_return = (
        benchmark_data["Close"].shift(-5)
        / benchmark_data["Close"]
        - 1
    )

    features["Benchmark_Future_Return_5d"] = benchmark_future_return

    valid_target = (
        features["Future_Return_5d"].notna()
        & features["Benchmark_Future_Return_5d"].notna()
    )

    features["Target"] = (
        features["Future_Return_5d"]
        > features["Benchmark_Future_Return_5d"]
    ).where(valid_target)

    return features

if __name__ == "__main__":
    market_data = download_market_data()

    apple_data = market_data["AAPL"]
    benchmark_data = market_data["^GSPC"]

    apple_features = create_features(apple_data)
    apple_features = create_target(
        apple_features,
        benchmark_data
    )

    print(
        apple_features[
            [
                "Close",
                "Momentum_5d",
                "Volatility_20d",
                "MA_Ratio_20d",
                "Future_Return_5d",
                "Benchmark_Future_Return_5d",
                "Target"
            ]
        ].tail(10)
    )