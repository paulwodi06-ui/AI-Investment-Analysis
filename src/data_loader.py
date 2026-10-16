import yfinance as yf


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "XOM",
    "WMT"
]

BENCHMARK = "^GSPC" # represents the S&P 500

START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

def download_market_data():
    symbols = TICKERS + [BENCHMARK]

    data = yf.download(
        symbols,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )

    return data

if __name__ == "__main__":
    market_data = download_market_data()

    print(market_data.head())

    print("\nDataset shape:")
    print(market_data.shape)

    print("\nApple data:")
    print(market_data["AAPL"].head())