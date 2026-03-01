"""Fama-French helpers for Indian factors + momentum.

This module downloads factor files from IIMA and uses factor-beta tilts to
choose a portfolio subset, which can then be optimized via Markowitz.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urljoin
from urllib.error import URLError
from urllib.request import urlopen
import re

import numpy as np
import pandas as pd

from .data import HistoricalDataClient

FF_DATA_INDEX_URL = "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
FF_DATA_URLS = {
    "daily": "resolved-from-index",
    "monthly": "resolved-from-index",
    "yearly": "resolved-from-index",
}

_DEFAULT_BATCH_DAYS = 2000
_DEFAULT_THROTTLE_SECONDS = 0.5
_DEFAULT_RETRIES = 5
_DEFAULT_BACKOFF_SECONDS = 2.0
_DEFAULT_MIN_OBSERVATIONS = 40
_DEFAULT_MIN_CALENDAR_OVERLAP_DAYS = 90


def download_ff_factor_files(
    frequencies: Iterable[str] = ("daily", "monthly", "yearly"),
    download_dir: str | Path = ".",
    overwrite: bool = False,
) -> Dict[str, str]:
    """Download IIMA Fama-French factor files into download_dir.

    Relative download_dir is interpreted from the caller's current working directory.
    """
    out_dir = Path(download_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    available_links = _discover_factor_download_links(FF_DATA_INDEX_URL)

    outputs: Dict[str, str] = {}
    for freq in frequencies:
        key = freq.lower().strip()
        if key not in FF_DATA_URLS:
            raise ValueError(f"Unsupported frequency: {freq}")

        target = out_dir / f"ff_india_{key}.csv"
        if target.exists() and not overwrite:
            outputs[key] = str(target.resolve())
            continue

        url = available_links.get(key)
        if not url:
            raise RuntimeError(
                f"Could not find '{key}' FF link under Survivorship-Bias Adjusted section at {FF_DATA_INDEX_URL}"
            )

        try:
            _download_to_file(url, target)
            outputs[key] = str(target.resolve())
        except (URLError, OSError):
            fallback = _find_local_factor_file(key, out_dir)
            if fallback is None:
                raise RuntimeError(
                    f"Failed to download {key} factors from discovered link: {url} and no local fallback file found in {out_dir}"
                )
            outputs[key] = str(fallback.resolve())

    return outputs


def load_ff_factors(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalize factor CSV into Date-indexed decimal return series.

    Returns columns among: MKT_RF, RF, SMB, HML, WML
    """
    path = Path(csv_path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Factor CSV is empty")

    cols = list(df.columns)
    date_col = cols[0]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    lookup = {_norm(c): c for c in cols[1:]}

    def pick(*candidates: str) -> Optional[str]:
        for c in candidates:
            for n, original in lookup.items():
                if c in n:
                    return original
        return None

    col_mkt_rf = pick("mktrf", "mkt_rf", "mkt-rf", "rmrf", "marketexcess", "mf", "marketfactor")
    col_rf = pick("rf", "riskfree")
    col_smb = pick("smb")
    col_hml = pick("hml")
    col_wml = pick("wml", "mom", "umd", "momentum")

    mapping = {
        "MKT_RF": col_mkt_rf,
        "RF": col_rf,
        "SMB": col_smb,
        "HML": col_hml,
        "WML": col_wml,
    }

    for target, source in mapping.items():
        if source is None:
            continue
        out[target] = pd.to_numeric(df[source], errors="coerce")

    out = out.dropna(subset=["date"]).set_index("date").sort_index()

    # Convert likely percentage values to decimal returns.
    # IIMA files are typically in percent units (e.g., 1.2 == 1.2%), but RF can be
    # small enough to evade per-column heuristics. Detect scale globally, then apply
    # to all factor columns consistently.
    ref_cols = [c for c in ["MKT_RF", "SMB", "HML", "WML"] if c in out.columns]
    ref_values = out[ref_cols].to_numpy(dtype=float).ravel()
    ref_values = ref_values[~np.isnan(ref_values)]
    if ref_values.size > 0:
        likely_percent_units = (
            float(np.median(np.abs(ref_values))) >= 0.2
            or float(np.quantile(np.abs(ref_values), 0.95)) > 1.0
        )
        if likely_percent_units:
            out[out.columns] = out[out.columns] / 100.0

    required = {"MKT_RF", "RF", "SMB", "HML", "WML"}
    missing = required.difference(set(out.columns))
    if missing:
        raise ValueError(f"Missing required factor columns: {sorted(missing)}")

    return out[["MKT_RF", "RF", "SMB", "HML", "WML"]].dropna(how="any")


def estimate_stock_factor_betas(
    kite: Any,
    symbols: list[str],
    start_date: str,
    end_date: Optional[str],
    factors: pd.DataFrame,
    exchange: str = "NSE",
    interval: str = "day",
) -> pd.DataFrame:
    """Estimate stock loadings to FF factors via OLS on excess returns."""
    client = HistoricalDataClient(
        kite,
        batch_days=_DEFAULT_BATCH_DAYS,
        throttle_seconds=_DEFAULT_THROTTLE_SECONDS,
        retries=_DEFAULT_RETRIES,
        backoff_seconds=_DEFAULT_BACKOFF_SECONDS,
    )
    rows = []
    for symbol in symbols:
        prices = client.fetch_close_prices(
            symbols=[symbol],
            from_date=start_date,
            to_date=end_date,
            exchange=exchange,
            interval=interval,
        )
        stock_rets = prices.pct_change().dropna(how="any")
        aligned = stock_rets.join(factors, how="inner")
        if len(aligned) < _DEFAULT_MIN_OBSERVATIONS:
            continue

        X = aligned[["MKT_RF", "SMB", "HML", "WML"]].values
        X = np.column_stack([np.ones(len(X)), X])
        y = (aligned[symbol] - aligned["RF"]).values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append(
            {
                "symbol": symbol,
                "alpha": float(beta[0]),
                "beta_mkt": float(beta[1]),
                "beta_smb": float(beta[2]),
                "beta_hml": float(beta[3]),
                "beta_wml": float(beta[4]),
            }
        )

    if not rows:
        raise ValueError(
            "Too few overlapping observations for factor regression across all symbols. "
            "Try older start_date or daily frequency."
        )

    return pd.DataFrame(rows).set_index("symbol")


def build_factor_tilt_portfolio(
    betas: pd.DataFrame,
    preference: Dict[str, float],
    top_n: int = 12,
) -> Dict[str, float]:
    """Build stock weights from desired factor tilts.

    preference examples:
    - {"beta_smb": 1, "beta_hml": 1, "beta_wml": 1}  -> small/value/momentum tilt
    - {"beta_smb": -1, "beta_hml": -1}                -> large/growth tilt
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    score = pd.Series(0.0, index=betas.index)
    for k, v in preference.items():
        if k not in betas.columns:
            raise ValueError(f"Unknown preference key: {k}")
        if float(v) < -1.0 or float(v) > 1.0:
            raise ValueError(f"Preference for {k} must be within [-1, 1]")
        z = (betas[k] - betas[k].mean()) / (betas[k].std(ddof=0) + 1e-12)
        score = score + float(v) * z

    score = score.sort_values(ascending=False).head(min(top_n, len(score)))
    positive = score - score.min() + 1e-6
    weights = positive / positive.sum()
    return {sym: float(w) for sym, w in weights.items()}


def generate_fama_french_markowitz_report(
    kite: Any,
    candidate_quantities: Dict[str, float],
    start_date: str,
    end_date: Optional[str] = None,
    frequency: str = "daily",
    preference: Optional[Dict[str, float]] = None,
    top_n: int = 12,
    factor_download_dir: Optional[str | Path] = None,
    output_html: Optional[str] = None,
    output_file_name: Optional[str] = None,
    exchange: str = "NSE",
    risk_free_rate: float = 0.06,
    factor_file_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Download factors, build factor-tilted portfolio, then run Markowitz report."""
    if not candidate_quantities:
        raise ValueError("candidate_quantities cannot be empty")
    if output_html is not None and output_file_name is not None:
        raise ValueError("Provide either output_html or output_file_name, not both")

    pref = preference or {"beta_smb": 1.0, "beta_hml": 1.0, "beta_wml": 1.0}
    symbols = [str(s).strip().upper() for s in candidate_quantities.keys()]
    run_dir = _default_run_dir()
    if factor_download_dir is None:
        factor_download_dir = run_dir
    if output_html is None:
        file_name = output_file_name or "ff_markowitz_report.html"
        if not str(file_name).lower().endswith(".html"):
            file_name = f"{file_name}.html"
        output_html = str(run_dir / file_name)

    freq_key = frequency.lower().strip()
    if factor_file_path is not None:
        factor_path = Path(factor_file_path).expanduser()
        if not factor_path.is_absolute():
            factor_path = Path.cwd() / factor_path
        if not factor_path.exists():
            raise ValueError(f"factor_file_path does not exist: {factor_path}")
        paths = {freq_key: str(factor_path.resolve())}
    else:
        paths = download_ff_factor_files(
            (freq_key,),
            download_dir=factor_download_dir,
        )

    factors = load_ff_factors(paths[freq_key])
    factor_max_date = factors.index.max()
    start_ts = pd.to_datetime(start_date)

    if end_date is None:
        effective_end = factor_max_date
    else:
        requested_end = pd.to_datetime(end_date)
        effective_end = min(requested_end, factor_max_date)

    if (effective_end - start_ts) < timedelta(days=_DEFAULT_MIN_CALENDAR_OVERLAP_DAYS):
        min_start = (factor_max_date - timedelta(days=_DEFAULT_MIN_CALENDAR_OVERLAP_DAYS)).date()
        raise ValueError(
            "Insufficient factor overlap window. "
            f"Factor data ends on {factor_max_date.date()}, so use start_date on or before {min_start} "
            f"for at least {_DEFAULT_MIN_CALENDAR_OVERLAP_DAYS} calendar days of overlap."
        )

    betas = estimate_stock_factor_betas(
        kite=kite,
        symbols=symbols,
        start_date=start_date,
        end_date=str(effective_end.date()),
        factors=factors,
        exchange=exchange,
    )

    ff_weights = build_factor_tilt_portfolio(betas=betas, preference=pref, top_n=top_n)

    total_qty = float(sum(abs(float(v)) for v in candidate_quantities.values()))
    if total_qty <= 0:
        total_qty = float(len(ff_weights))

    selected_quantities = {
        sym: max(1.0, ff_weights[sym] * total_qty)
        for sym in ff_weights.keys()
    }

    from .report import generate_markowitz_report

    pref_title = (
        "FF Efficient Portfolio | "
        f"SMB {pref.get('beta_smb', 0.0):+0.2f} | "
        f"HML {pref.get('beta_hml', 0.0):+0.2f} | "
        f"WML {pref.get('beta_wml', 0.0):+0.2f}"
    )

    report = generate_markowitz_report(
        kite=kite,
        portfolio_quantities=selected_quantities,
        start_date=start_date,
        end_date=str(effective_end.date()),
        output_html=output_html,
        report_title=pref_title,
        exchange=exchange,
        risk_free_rate=risk_free_rate,
    )

    report["ff_selected_weights"] = ff_weights
    report["ff_factor_betas"] = betas.loc[list(ff_weights.keys())].to_dict(orient="index")
    report["ff_factor_file"] = paths[freq_key]
    report["ff_factor_max_date"] = str(factor_max_date.date())
    report["ff_effective_end_date"] = str(effective_end.date())
    report["ff_tilt_analysis"] = _build_ff_tilt_analysis(pref)
    report["ff_tilt_levels"] = [1.0, 0.75, 0.5, 0.25, 0.0]

    return report


def _norm(col: str) -> str:
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def _discover_factor_download_links(index_url: str) -> Dict[str, str]:
    html = _read_text_url(index_url)
    hrefs = re.findall(r"""href=["']([^"']+)["']""", html, flags=re.IGNORECASE)

    by_freq: Dict[str, str] = {}
    for href in hrefs:
        href_lower = href.lower()
        if "survivorshipbiasadjusted" not in href_lower:
            continue
        if "fourfactors_and_market_returns" not in href_lower:
            continue

        abs_url = urljoin(index_url, href)
        if "daily" in href_lower and "daily" not in by_freq:
            by_freq["daily"] = abs_url
        elif "monthly" in href_lower and "monthly" not in by_freq:
            by_freq["monthly"] = abs_url
        elif "yearly" in href_lower and "yearly" not in by_freq:
            by_freq["yearly"] = abs_url

    return by_freq


def _download_to_file(url: str, target: Path, timeout: int = 20) -> None:
    data = _read_binary_url(url, timeout=timeout)
    target.write_bytes(data)


def _read_text_url(url: str, timeout: int = 20) -> str:
    data = _read_binary_url(url, timeout=timeout)
    return data.decode("utf-8", errors="ignore")


def _read_binary_url(url: str, timeout: int = 20) -> bytes:
    with urlopen(url, timeout=timeout) as response:
        return response.read()


def _find_local_factor_file(frequency: str, folder: Path) -> Optional[Path]:
    freq = frequency.lower().strip()
    exts = ("*.csv", "*.xlsx", "*.xls")
    candidates: list[Path] = []
    for ext in exts:
        candidates.extend(sorted(folder.glob(ext)))

    def score(path: Path) -> int:
        name = path.name.lower()
        s = 0
        if "fourfactors" in name:
            s += 5
        if "market_returns" in name:
            s += 4
        if "survivorshipbiasadjusted" in name:
            s += 3
        if freq in name:
            s += 2
        return s

    ranked = [p for p in candidates if score(p) > 0]
    if not ranked:
        return None
    ranked.sort(key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    return ranked[0]


def _default_run_dir() -> Path:
    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else Path.cwd()
    if script.suffix:
        return script.resolve().parent
    return Path.cwd().resolve()


def _build_ff_tilt_analysis(preference: Dict[str, float]) -> Dict[str, Any]:
    levels = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]

    def explain_factor(name: str, val: float, pos: str, neg: str) -> str:
        strength = abs(float(val))
        if strength >= 0.99:
            intensity = "very strong"
        elif strength >= 0.74:
            intensity = "strong"
        elif strength >= 0.49:
            intensity = "moderate"
        elif strength >= 0.24:
            intensity = "light"
        else:
            intensity = "neutral"

        if val > 0:
            direction = pos
        elif val < 0:
            direction = neg
        else:
            direction = "neutral"

        return f"{name}: {val:+.2f} -> {intensity} tilt toward {direction}"

    smb = float(preference.get("beta_smb", 0.0))
    hml = float(preference.get("beta_hml", 0.0))
    wml = float(preference.get("beta_wml", 0.0))

    return {
        "selected": {
            "beta_smb": smb,
            "beta_hml": hml,
            "beta_wml": wml,
        },
        "meaning": {
            "SMB": "Size factor. Positive tilts toward small-cap; negative toward large-cap.",
            "HML": "Value factor. Positive tilts toward value; negative toward growth.",
            "WML": "Momentum factor. Positive tilts toward recent winners; negative toward losers/reversal.",
        },
        "interpretation": [
            explain_factor("SMB", smb, "small-cap", "large-cap"),
            explain_factor("HML", hml, "value", "growth"),
            explain_factor("WML", wml, "momentum", "anti-momentum"),
        ],
        "level_guide": {
            "allowed_levels": levels,
            "how_to_use": (
                "Use +1/+0.75/+0.5/+0.25/0/-0.25/-0.5/-0.75/-1 for strongest positive to strongest negative exposure."
            ),
        },
    }
