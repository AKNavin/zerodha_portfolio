"""Custom exceptions for zerodha_markowitz."""


class InstrumentNotFoundError(ValueError):
    """Raised when an instrument symbol cannot be resolved on an exchange."""


class HistoricalDataError(RuntimeError):
    """Raised when historical data fetch fails."""


class OptimizationError(RuntimeError):
    """Raised when optimization for frontier/tangency fails."""
