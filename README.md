# AI Investment Analysis

A Python machine learning project investigating whether historical market indicators can help predict short-term equity performance.

The project was inspired by my participation in the JPMorganChase x AmplifyME MarketMind Challenge and explores how quantitative models can complement human judgement in financial-market forecasting.

## Backtest Results

The Random Forest model was used to rank stocks by their predicted probability of outperforming the S&P 500 over the following five trading days. At each rebalance date, the three highest-ranked stocks were selected into an equally weighted portfolio.

Over 100 non-overlapping five-day periods in the 2024–2025 test period:

- Average model portfolio return: 0.62% per five-day period
- Average S&P 500 return: 0.40%
- Average excess return: 0.22%
- Portfolio outperformance rate: 53%
- Model portfolio cumulative return: 76.03%
- S&P 500 cumulative return: 45.40%
- Model portfolio maximum drawdown: -16.16%
- S&P 500 maximum drawdown: -17.07%
- Excess-return p-value: 0.2791

Although the model-selected portfolio produced higher cumulative returns in the backtest, the excess return was not statistically significant. Therefore, the results should be interpreted as encouraging rather than evidence of a reliable market-beating strategy.

![Backtest performance](results/backtest_performance.png)