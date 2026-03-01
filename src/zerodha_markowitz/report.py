"""HTML report generator for Markowitz portfolio comparison."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import HistoricalDataClient
from .optimizer import FrontierResult, optimize_portfolios


def generate_markowitz_report(
    kite: Any,
    portfolio_quantities: Dict[str, float],
    start_date: str,
    output_html: Optional[str] = None,
    output_file_name: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "NSE",
    interval: str = "day",
    risk_free_rate: float = 0.06,
    batch_days: int = 2000,
    throttle_seconds: float = 0.5,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Generate an HTML report comparing user portfolio vs tangency portfolio."""
    if not portfolio_quantities:
        raise ValueError("portfolio_quantities cannot be empty")
    if output_html is not None and output_file_name is not None:
        raise ValueError("Provide either output_html or output_file_name, not both")

    clean_qty = {str(k).strip().upper(): float(v) for k, v in portfolio_quantities.items()}
    symbols = list(clean_qty.keys())

    client = HistoricalDataClient(
        kite,
        batch_days=batch_days,
        throttle_seconds=throttle_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )

    prices = client.fetch_close_prices(
        symbols=symbols,
        from_date=start_date,
        to_date=end_date,
        exchange=exchange,
        interval=interval,
    )

    start_prices = prices.iloc[0]
    user_weights = _compute_user_weights(clean_qty, start_prices)

    result = optimize_portfolios(
        prices=prices,
        user_weights=user_weights,
        risk_free_rate=risk_free_rate,
    )

    fig = _build_frontier_figure(result, risk_free_rate)
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    user_table = _build_portfolio_table(
        symbols=result.symbols,
        quantities=clean_qty,
        start_prices=start_prices,
        weights=result.user_weights,
        title="User Portfolio (from quantities)",
    )

    best_table = _build_best_table(
        symbols=result.symbols,
        weights=result.tangency_weights,
        mean_returns=result.mean_returns,
    )

    analysis_html = _build_analysis_html(result)

    html = _wrap_report_html(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date=str(prices.index.min().date()),
        end_date=str(prices.index.max().date()),
        risk_free_rate=risk_free_rate,
        plot_html=plot_html,
        user_table_html=user_table,
        best_table_html=best_table,
        analysis_html=analysis_html,
    )

    if output_html is None:
        file_name = output_file_name or "markowitz_report.html"
        if not file_name.lower().endswith(".html"):
            file_name = f"{file_name}.html"
        output_html = str(_default_run_dir() / file_name)

    out_path = Path(output_html).expanduser()
    if not out_path.is_absolute():
        # Store relative outputs in the caller's current working directory.
        out_path = Path.cwd() / out_path
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {
        "output_html": str(out_path),
        "symbols": result.symbols,
        "user_weights": result.user_weights_dict(),
        "tangency_weights": result.tangency_weights_dict(),
        "user_return": result.user_return,
        "user_vol": result.user_vol,
        "user_sharpe": result.user_sharpe,
        "tangency_return": result.tangency_return,
        "tangency_vol": result.tangency_vol,
        "tangency_sharpe": result.tangency_sharpe,
    }


def _default_run_dir() -> Path:
    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else Path.cwd()
    if script.suffix:
        return script.resolve().parent
    return Path.cwd().resolve()


def _compute_user_weights(quantities: Dict[str, float], start_prices: pd.Series) -> np.ndarray:
    values = np.array([quantities[s] * float(start_prices[s]) for s in start_prices.index], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Total portfolio market value at start date must be > 0")
    return values / total


def _build_frontier_figure(result: FrontierResult, rf: float) -> go.Figure:
    max_vol = float(np.nanmax(result.frontier_vols))
    cml_x = np.linspace(0.0, max(max_vol, result.tangency_vol) * 1.05, 80)
    cml_y = rf + ((result.tangency_return - rf) / result.tangency_vol) * cml_x

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.frontier_vols,
            y=result.frontier_returns,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#0b7285", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cml_x,
            y=cml_y,
            mode="lines",
            name="CML",
            line=dict(color="#2f9e44", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.user_vol],
            y=[result.user_return],
            mode="markers+text",
            name="User Portfolio",
            text=["User"],
            textposition="top center",
            marker=dict(color="#e8590c", size=12),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.tangency_vol],
            y=[result.tangency_return],
            mode="markers+text",
            name="Best (Tangency)",
            text=["Tangency"],
            textposition="top center",
            marker=dict(color="#1c7ed6", size=12, symbol="diamond"),
        )
    )

    fig.update_layout(
        title="Markowitz Efficient Frontier vs User Portfolio",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_portfolio_table(
    symbols: list[str],
    quantities: Dict[str, float],
    start_prices: pd.Series,
    weights: np.ndarray,
    title: str,
) -> str:
    records = []
    for idx, symbol in enumerate(symbols):
        qty = float(quantities[symbol])
        px = float(start_prices[symbol])
        value = qty * px
        wt = float(weights[idx])
        records.append(
            f"<tr><td>{symbol}</td><td>{qty:.4f}</td><td>{px:.2f}</td><td>{value:.2f}</td><td>{wt:.2%}</td></tr>"
        )

    rows = "".join(records)
    return (
        f"<h3>{title}</h3>"
        "<table><thead><tr><th>Stock</th><th>Qty</th><th>Start Close</th><th>Value</th><th>Weight</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_best_table(symbols: list[str], weights: np.ndarray, mean_returns: pd.Series) -> str:
    min_display_weight = 1e-6
    entries = []
    for idx, symbol in enumerate(symbols):
        wt = float(weights[idx])
        if wt <= min_display_weight:
            continue
        ann_ret = float(mean_returns[symbol])
        entries.append((symbol, wt, ann_ret))

    entries.sort(key=lambda x: x[1], reverse=True)
    rows = [
        f"<tr><td>{symbol}</td><td>{wt:.2%}</td><td>{ann_ret:.2%}</td></tr>"
        for symbol, wt, ann_ret in entries
    ]

    return (
        "<h3>Best Portfolio (Tangency on CML)</h3>"
        "<table><thead><tr><th>Stock</th><th>Weight</th><th>Asset Ann. Return</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _build_analysis_html(result: FrontierResult) -> str:
    delta_return = result.tangency_return - result.user_return
    delta_vol = result.tangency_vol - result.user_vol
    delta_sharpe = result.tangency_sharpe - result.user_sharpe

    return (
        "<h3>Comparison Summary</h3>"
        "<ul>"
        f"<li>User: return={result.user_return:.2%}, vol={result.user_vol:.2%}, sharpe={result.user_sharpe:.3f}</li>"
        f"<li>Best: return={result.tangency_return:.2%}, vol={result.tangency_vol:.2%}, sharpe={result.tangency_sharpe:.3f}</li>"
        f"<li>Delta (Best - User): return={delta_return:.2%}, vol={delta_vol:.2%}, sharpe={delta_sharpe:.3f}</li>"
        "</ul>"
    )


def _wrap_report_html(
    generated_at: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    plot_html: str,
    user_table_html: str,
    best_table_html: str,
    analysis_html: str,
) -> str:
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Markowitz Portfolio Report</title>
  <style>
    :root {{
      --bg: #f8f9fa;
      --panel: #ffffff;
      --line: #dee2e6;
      --text: #212529;
      --muted: #495057;
      --accent: #0b7285;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #e7f5ff, var(--bg));
      padding: 20px;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    .header {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .plot {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    ul {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"header\">
      <h1>Markowitz Efficient Frontier Report</h1>
      <div class=\"meta\">Generated: {generated_at} | Data window: {start_date} to {end_date} | Risk-free rate: {risk_free_rate:.2%}</div>
    </div>
    <div class=\"plot\">{plot_html}</div>
    <div class=\"grid\">
      <div class=\"card\">{user_table_html}</div>
      <div class=\"card\">{best_table_html}{analysis_html}</div>
    </div>
  </div>
</body>
</html>
""".strip()
