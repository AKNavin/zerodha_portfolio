"""zerodha_portfolio package."""

from .data import HistoricalDataClient
from .optimizer import FrontierResult, optimize_portfolios

__all__ = [
    "HistoricalDataClient",
    "FrontierResult",
    "optimize_portfolios",
    "generate_markowitz_report",
    "FF_DATA_URLS",
    "download_ff_factor_files",
    "load_ff_factors",
    "estimate_stock_factor_betas",
    "build_factor_tilt_portfolio",
    "generate_fama_french_markowitz_report",
]


def generate_markowitz_report(*args, **kwargs):
    from .report import generate_markowitz_report as _fn

    return _fn(*args, **kwargs)


def generate_fama_french_markowitz_report(*args, **kwargs):
    from .fama_french import generate_fama_french_markowitz_report as _fn

    return _fn(*args, **kwargs)


def download_ff_factor_files(*args, **kwargs):
    from .fama_french import download_ff_factor_files as _fn

    return _fn(*args, **kwargs)


def load_ff_factors(*args, **kwargs):
    from .fama_french import load_ff_factors as _fn

    return _fn(*args, **kwargs)


def estimate_stock_factor_betas(*args, **kwargs):
    from .fama_french import estimate_stock_factor_betas as _fn

    return _fn(*args, **kwargs)


def build_factor_tilt_portfolio(*args, **kwargs):
    from .fama_french import build_factor_tilt_portfolio as _fn

    return _fn(*args, **kwargs)


FF_DATA_URLS = {
    "daily": "resolved-from-index",
    "monthly": "resolved-from-index",
    "yearly": "resolved-from-index",
}
