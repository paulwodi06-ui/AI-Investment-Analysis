import pandas as pd
import pytest

from features import create_features, create_target


def test_daily_return():
    data = pd.DataFrame({
        "Close": [100.0, 105.0]
    })

    features = create_features(data)

    assert features["Daily_Return"].iloc[1] == pytest.approx(0.05)


def test_five_day_momentum():
    data = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104, 110]
    })

    features = create_features(data)

    assert features["Momentum_5d"].iloc[5] == pytest.approx(0.10)


def test_moving_average_ratio():
    data = pd.DataFrame({
        "Close": [100.0] * 19 + [110.0]
    })

    features = create_features(data)

    expected_ma = 100.5
    expected_ratio = 110 / expected_ma - 1

    assert features["MA_Ratio_20d"].iloc[-1] == pytest.approx(
        expected_ratio
    )


def test_initial_volatility_is_missing():
    data = pd.DataFrame({
        "Close": [100.0] * 20
    })

    features = create_features(data)

    assert pd.isna(features["Volatility_20d"].iloc[0])


def test_future_return():
    stock_data = pd.DataFrame({
        "Close": [100, 100, 100, 100, 100, 110]
    })

    benchmark_data = pd.DataFrame({
        "Close": [100, 100, 100, 100, 100, 105]
    })

    features = create_features(stock_data)
    result = create_target(features, benchmark_data)

    assert result["Future_Return_5d"].iloc[0] == pytest.approx(0.10)
    assert result["Benchmark_Future_Return_5d"].iloc[0] == pytest.approx(0.05)
    assert result["Target"].iloc[0] == True


def test_final_five_targets_are_missing():
    stock_data = pd.DataFrame({
        "Close": [100] * 10
    })

    benchmark_data = pd.DataFrame({
        "Close": [100] * 10
    })

    features = create_features(stock_data)
    result = create_target(features, benchmark_data)

    assert result["Target"].tail(5).isna().all()