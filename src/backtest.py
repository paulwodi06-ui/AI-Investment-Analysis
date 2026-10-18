import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import ttest_1samp

from dataset_builder import build_ml_dataset

import matplotlib.pyplot as plt

def run_backtest():
    dataset = build_ml_dataset()

    features = [
        "Daily_Return",
        "Momentum_5d",
        "Volatility_20d",
        "MA_Ratio_20d"
    ]

    train_end = "2023-12-21"
    test_start = "2024-01-01"

    train_data = dataset[
        dataset["Date"] <= train_end
    ].copy()

    test_data = dataset[
        dataset["Date"] >= test_start
    ].copy()

    X_train = train_data[features]
    y_train = train_data["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_data["Predicted_Probability"] = (
        model.predict_proba(
            test_data[features]
        )[:, 1]
    )

    ranked_data = test_data.sort_values(
        ["Date", "Predicted_Probability"],
        ascending=[True, False]
    )

    top_stocks = (
        ranked_data
        .groupby("Date")
        .head(3)
        .copy()
    )

    test_dates = sorted(
        top_stocks["Date"].unique()
    )

    rebalance_dates = test_dates[::5]

    top_stocks = top_stocks[
        top_stocks["Date"].isin(rebalance_dates)
    ]

    portfolio_returns = (
        top_stocks
        .groupby("Date")
        .agg(
            Portfolio_Return=("Future_Return_5d", "mean"),
            Benchmark_Return=(
                "Benchmark_Future_Return_5d",
                "first"
            )
        )
        .reset_index()
    )

    portfolio_returns = (
    top_stocks
    .groupby("Date")
    .agg(
        Portfolio_Return=("Future_Return_5d", "mean"),
        Benchmark_Return=(
            "Benchmark_Future_Return_5d",
            "first"
        )
    )
    .reset_index()
    )

    portfolio_returns["Excess_Return"] = (
        portfolio_returns["Portfolio_Return"]
        - portfolio_returns["Benchmark_Return"]
    )

    t_stat, p_value = ttest_1samp(
    portfolio_returns["Excess_Return"],
    popmean=0
    )

    portfolio_returns["Portfolio_Growth"] = (
        1 + portfolio_returns["Portfolio_Return"]
    ).cumprod()

    portfolio_returns["Benchmark_Growth"] = (
        1 + portfolio_returns["Benchmark_Return"]
    ).cumprod()

    portfolio_returns["Portfolio_Peak"] = (
    portfolio_returns["Portfolio_Growth"].cummax()
    )

    portfolio_returns["Portfolio_Drawdown"] = (
        portfolio_returns["Portfolio_Growth"]
        / portfolio_returns["Portfolio_Peak"]
        - 1
    )

    portfolio_returns["Benchmark_Peak"] = (
        portfolio_returns["Benchmark_Growth"].cummax()
    )

    portfolio_returns["Benchmark_Drawdown"] = (
        portfolio_returns["Benchmark_Growth"]
        / portfolio_returns["Benchmark_Peak"]
        - 1
    )

    max_drawdown = portfolio_returns["Portfolio_Drawdown"].min()

    benchmark_max_drawdown = (
    portfolio_returns["Benchmark_Drawdown"].min()
    )

    print(
        "Number of rebalance periods:",
        len(portfolio_returns)
    )

    print(
        "Average portfolio 5-day return:",
        round(
            portfolio_returns["Portfolio_Return"].mean(),
            4
        )
    )

    print(
        "Average benchmark 5-day return:",
        round(
            portfolio_returns["Benchmark_Return"].mean(),
            4
        )
    )

    print(
        "Average excess return:",
        round(
            portfolio_returns["Excess_Return"].mean(),
            4
        )
    )

    hit_rate = (
        portfolio_returns["Excess_Return"] > 0
    ).mean()

    print(
        "Portfolio outperformance rate:",
        round(hit_rate, 4)
    )

    total_portfolio_return = (
        portfolio_returns["Portfolio_Growth"].iloc[-1] - 1
    )

    total_benchmark_return = (
        portfolio_returns["Benchmark_Growth"].iloc[-1] - 1
    )

    print(
        "Total portfolio return:",
        round(total_portfolio_return, 4)
    )

    print(
        "Total benchmark return:",
        round(total_benchmark_return, 4)
    )

    print(
    "Portfolio max drawdown:",
    round(max_drawdown, 4)
    )

    print(
    "Benchmark max drawdown:",
    round(benchmark_max_drawdown, 4)
    )

    print(
    "Excess return t-statistic:",
    round(t_stat, 4)
    )

    print(
    "Excess return p-value:",
    round(p_value, 4)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        portfolio_returns["Date"],
        portfolio_returns["Portfolio_Growth"],
        label="Model Portfolio"
    )

    plt.plot(
        portfolio_returns["Date"],
        portfolio_returns["Benchmark_Growth"],
        label="S&P 500"
    )

    plt.title("Model Portfolio vs S&P 500")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
    "results/backtest_performance.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    return portfolio_returns

if __name__ == "__main__":
    run_backtest()