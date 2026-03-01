# zerodha_portfolio

Zerodha portfolio analytics toolkit for practical equity allocation decisions:

1. Markowitz efficient frontier report for your current holdings
2. Fama-French factor-tilted stock selection followed by Markowitz optimization

It is built for users who already think in terms of stocks and quantities and want a visual, explainable report they can use for research, review meetings, or resume/project demonstration.

## Install

```bash
pip install zerodha-portfolio
```

For local development:

```bash
pip install -e .[dev]
```

## What You Input

Both workflows start from a dictionary of stocks and quantities:

```python
portfolio = {
    "RELIANCE": 18,
    "TCS": 24,
    "INFY": 32,
    "HDFCBANK": 20,
}
```

You also provide:

- `start_date` (required)
- `end_date` (optional)
- `risk_free_rate` (optional, default `0.06`)

## What You Get Back

Both APIs return a Python dictionary with key metrics and a saved HTML report path:

- `output_html`
- `symbols`
- `user_weights`
- `tangency_weights`
- `user_return`, `user_vol`, `user_sharpe`
- `tangency_return`, `tangency_vol`, `tangency_sharpe`

The HTML report includes:

- Efficient Frontier curve
- Capital Market Line (CML)
- Your current portfolio point
- Tangency (max-Sharpe) portfolio point
- Portfolio composition tables
- Comparison summary

## Workflow 1: Markowitz Report

Use this when you already have a portfolio and want to compare it against the mathematically optimal long-only allocation from the same stock universe.

### Example Code

```python
from kiteconnect import KiteConnect
from zerodha_portfolio import generate_markowitz_report

kite = KiteConnect(api_key="your_api_key")
kite.set_access_token("your_access_token")

result = generate_markowitz_report(
    kite=kite,
    portfolio_quantities={
        "RELIANCE": 18,
        "TCS": 24,
        "INFY": 32,
        "HDFCBANK": 20,
    },
    start_date="2022-01-01",
    end_date="2026-02-28",
    output_file_name="markowitz_report",
    risk_free_rate=0.06,
)

print(result["output_html"])
```

### Example Script and Sample Report

- Script: `examples/example_report.py`
- Sample output: `examples/markowitz_report.html`

## Workflow 2: Fama-French -> Markowitz Report

Use this when you want factor-aware stock selection first, and only then run mean-variance optimization on the selected basket.

This workflow:

1. Loads Indian Fama-French + Momentum factors (downloaded or local file)
2. Estimates each stock's factor betas
3. Scores stocks using your factor preference
4. Picks top stocks (`top_n`)
5. Runs Markowitz report on that selected set

### SMB, HML, WML Meaning

- `SMB` (Small Minus Big): size factor
- Positive means tilt toward small-cap stocks
- Negative means tilt toward large-cap stocks

- `HML` (High Minus Low): value factor
- Positive means tilt toward value stocks (high book-to-market)
- Negative means tilt toward growth stocks

- `WML` (Winners Minus Losers): momentum factor
- Positive means tilt toward recent outperformers
- Negative means tilt toward anti-momentum / reversal

### Preference Input Format

Provide factor preferences using `-1` to `+1`:

```python
preference = {
    "beta_smb": 1.0,   # small-cap tilt
    "beta_hml": 1.0,   # value tilt
    "beta_wml": 1.0,   # momentum tilt
}
```

Examples:

- Small + Value + Momentum: `{"beta_smb": 1.0, "beta_hml": 1.0, "beta_wml": 1.0}`
- Large + Growth: `{"beta_smb": -1.0, "beta_hml": -1.0}`
- Neutral on momentum: `{"beta_smb": 0.5, "beta_hml": 0.5, "beta_wml": 0.0}`

### Example Code

```python
from kiteconnect import KiteConnect
from zerodha_portfolio import generate_fama_french_markowitz_report

kite = KiteConnect(api_key="your_api_key", timeout=30)
kite.set_access_token("your_access_token")

result = generate_fama_french_markowitz_report(
    kite=kite,
    candidate_quantities={
        "RELIANCE": 18,
        "TCS": 24,
        "INFY": 32,
        "HDFCBANK": 20,
        "ICICIBANK": 28,
        "SBIN": 45,
    },
    start_date="2022-01-01",
    end_date="2026-02-28",
    frequency="daily",
    preference={"beta_smb": -1.0, "beta_hml": 1.0, "beta_wml": 1.0},
    top_n=12,
    output_file_name="ff_markowitz_report",
    risk_free_rate=0.06,
    # Optional fallback if download URL changes:
    # factor_file_path=r"C:\\path\\to\\ff_file.csv",
)

print(result["output_html"])
print(result["ff_selected_weights"])
```

### Example Script and Sample Report

- Script: `examples/example_ff_markowitz.py`
- Sample output: `examples/ff_markowitz_report.html`

## API Summary

- `generate_markowitz_report(...)`
- `generate_fama_french_markowitz_report(...)`
- `download_ff_factor_files(...)`
- `load_ff_factors(...)`
- `estimate_stock_factor_betas(...)`
- `build_factor_tilt_portfolio(...)`

Backward-compatible import is preserved:

```python
from zerodha_markowitz import generate_markowitz_report
```

## Output File Behavior

- `output_html`: full output path (highest priority)
- `output_file_name`: file name only (saved in script run directory)
- If neither is given, a default name is used:
  - Markowitz: `markowitz_report.html`
  - FF-Markowitz: `ff_markowitz_report.html`

## Data Source

- IIMA Indian Fama-French + Momentum dataset:
  https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/

## License

MIT
