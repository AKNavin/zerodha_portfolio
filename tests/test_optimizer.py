import numpy as np
import pandas as pd

from zerodha_markowitz.optimizer import optimize_portfolios


def test_optimize_portfolios_shapes():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 130, len(dates)),
            "BBB": np.linspace(120, 140, len(dates)) + np.sin(np.arange(len(dates))),
            "CCC": np.linspace(90, 150, len(dates)) + np.cos(np.arange(len(dates))),
        },
        index=dates,
    )

    user_weights = np.array([0.4, 0.3, 0.3])
    result = optimize_portfolios(prices=prices, user_weights=user_weights, risk_free_rate=0.06)

    assert len(result.symbols) == 3
    assert result.frontier_returns.shape == result.frontier_vols.shape
    assert np.isclose(result.tangency_weights.sum(), 1.0)
    assert np.isclose(result.user_weights.sum(), 1.0)
