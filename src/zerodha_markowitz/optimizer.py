"""Markowitz optimizer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .exceptions import OptimizationError

TRADING_DAYS = 252


@dataclass
class FrontierResult:
    symbols: list[str]
    mean_returns: pd.Series
    covariance: pd.DataFrame
    frontier_returns: np.ndarray
    frontier_vols: np.ndarray
    tangency_weights: np.ndarray
    tangency_return: float
    tangency_vol: float
    tangency_sharpe: float
    user_weights: np.ndarray
    user_return: float
    user_vol: float
    user_sharpe: float

    def tangency_weights_dict(self) -> Dict[str, float]:
        return {s: float(w) for s, w in zip(self.symbols, self.tangency_weights)}

    def user_weights_dict(self) -> Dict[str, float]:
        return {s: float(w) for s, w in zip(self.symbols, self.user_weights)}


def optimize_portfolios(
    prices: pd.DataFrame,
    user_weights: np.ndarray,
    risk_free_rate: float = 0.06,
    points: int = 70,
    long_only: bool = True,
) -> FrontierResult:
    """Compute efficient frontier and tangency portfolio from price matrix."""
    if prices.shape[1] < 2:
        raise OptimizationError("At least 2 assets are required for frontier optimization")

    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise OptimizationError("Insufficient return observations")

    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n = len(mu)

    user_weights = np.asarray(user_weights, dtype=float)
    if user_weights.shape != (n,):
        raise OptimizationError("user_weights must match number of symbols")
    if np.any(user_weights < 0):
        raise OptimizationError("user_weights must be non-negative")
    if not np.isclose(user_weights.sum(), 1.0, atol=1e-6):
        raise OptimizationError("user_weights must sum to 1")

    bounds = [(0.0, 1.0) for _ in range(n)] if long_only else [(-1.0, 1.0) for _ in range(n)]

    tangency_weights = _solve_tangency(mu.values, cov.values, risk_free_rate, bounds)

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    target_returns = np.linspace(min_ret, max_ret, points)
    frontier_vols = np.array(
        [
            _solve_min_vol_for_target(mu.values, cov.values, target, bounds)
            for target in target_returns
        ]
    )

    tangency_return, tangency_vol = _portfolio_stats(tangency_weights, mu.values, cov.values)
    user_return, user_vol = _portfolio_stats(user_weights, mu.values, cov.values)

    tangency_sharpe = (tangency_return - risk_free_rate) / tangency_vol if tangency_vol > 0 else np.nan
    user_sharpe = (user_return - risk_free_rate) / user_vol if user_vol > 0 else np.nan

    return FrontierResult(
        symbols=list(prices.columns),
        mean_returns=mu,
        covariance=cov,
        frontier_returns=target_returns,
        frontier_vols=frontier_vols,
        tangency_weights=tangency_weights,
        tangency_return=tangency_return,
        tangency_vol=tangency_vol,
        tangency_sharpe=tangency_sharpe,
        user_weights=user_weights,
        user_return=user_return,
        user_vol=user_vol,
        user_sharpe=user_sharpe,
    )


def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol


def _solve_tangency(mu: np.ndarray, cov: np.ndarray, rf: float, bounds: list[tuple[float, float]]) -> np.ndarray:
    n = len(mu)

    def objective(w: np.ndarray) -> float:
        ret, vol = _portfolio_stats(w, mu, cov)
        if vol <= 0:
            return 1e9
        return -((ret - rf) / vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.repeat(1.0 / n, n)
    res = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")

    if not res.success:
        raise OptimizationError(f"Tangency optimization failed: {res.message}")

    weights = np.asarray(res.x, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / weights.sum()
    return weights


def _solve_min_vol_for_target(
    mu: np.ndarray,
    cov: np.ndarray,
    target_ret: float,
    bounds: list[tuple[float, float]],
) -> float:
    n = len(mu)

    def objective(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_ret},
    ]

    x0 = np.repeat(1.0 / n, n)
    res = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")

    if not res.success:
        return np.nan

    return float(res.fun)
