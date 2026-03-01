"""Historical data utilities built on top of KiteConnect."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .exceptions import HistoricalDataError, InstrumentNotFoundError

DateLike = Union[str, date, datetime]


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_seconds: float = 1.0


class HistoricalDataClient:
    """Fetch close prices from Zerodha historical data with batching and retries."""

    def __init__(
        self,
        kite: Any,
        batch_days: int = 2000,
        throttle_seconds: float = 0.5,
        symbol_throttle_seconds: float = 0.0,
        retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.kite = kite
        self.batch_days = batch_days
        self.throttle_seconds = throttle_seconds
        self.symbol_throttle_seconds = symbol_throttle_seconds
        self.retry = RetryConfig(retries=retries, backoff_seconds=backoff_seconds)
        self._instrument_cache: Dict[str, List[Dict[str, Any]]] = {}

    def fetch_close_prices(
        self,
        symbols: List[str],
        from_date: DateLike,
        to_date: Optional[DateLike] = None,
        exchange: str = "NSE",
        interval: str = "day",
    ) -> pd.DataFrame:
        """Return a date-indexed close-price matrix for the requested symbols."""
        if not symbols:
            raise ValueError("symbols cannot be empty")

        start = self._coerce_datetime(from_date)
        end = self._coerce_datetime(to_date or datetime.today())
        if start > end:
            raise ValueError("from_date must be <= to_date")

        series_list: List[pd.Series] = []
        for symbol in symbols:
            token = self.resolve_instrument_token(symbol, exchange=exchange)
            raw = self.fetch_by_token(
                instrument_token=token,
                from_date=start,
                to_date=end,
                interval=interval,
            )
            if raw.empty or "close" not in raw.columns:
                raise HistoricalDataError(f"No close price data returned for {symbol}")

            if "date" not in raw.columns:
                raise HistoricalDataError(f"Missing date column in historical data for {symbol}")

            s = raw[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
            s = s.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
            s = s.set_index("date")["close"]
            s.name = symbol
            series_list.append(s)

            if self.symbol_throttle_seconds > 0:
                time.sleep(self.symbol_throttle_seconds)

        prices = pd.concat(series_list, axis=1).sort_index()
        prices = prices.dropna(how="any")

        if prices.empty:
            raise HistoricalDataError(
                "No overlapping price history after alignment. Try a wider date range."
            )

        return prices

    def fetch_by_token(
        self,
        instrument_token: Union[int, str],
        from_date: DateLike,
        to_date: DateLike,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> pd.DataFrame:
        start = self._coerce_datetime(from_date)
        end = self._coerce_datetime(to_date)
        frames: List[pd.DataFrame] = []
        chunk_days = self._chunk_days_for_interval(interval)

        for window_start, window_end in self._iter_date_windows(start, end, chunk_days=chunk_days):
            candles = self._historical_with_retry(
                instrument_token=instrument_token,
                from_date=window_start,
                to_date=window_end,
                interval=interval,
                continuous=continuous,
                oi=oi,
            )
            if candles:
                frames.append(pd.DataFrame(candles))

            if self.throttle_seconds > 0:
                time.sleep(self.throttle_seconds)

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"]) 
            data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            data = data.reset_index(drop=True)

        return data

    def resolve_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        exchange_upper = exchange.upper()
        records = self._get_instruments(exchange_upper)

        target = symbol.strip().upper()
        matches = [r for r in records if str(r.get("tradingsymbol", "")).upper() == target]
        if not matches:
            raise InstrumentNotFoundError(
                f"Instrument '{symbol}' not found on exchange '{exchange_upper}'."
            )

        token = matches[0].get("instrument_token")
        if token is None:
            raise InstrumentNotFoundError(
                f"Instrument '{symbol}' found but missing instrument_token."
            )

        return int(token)

    def _get_instruments(self, exchange: str) -> List[Dict[str, Any]]:
        if exchange not in self._instrument_cache:
            self._instrument_cache[exchange] = self._instruments_with_retry(exchange)
        return self._instrument_cache[exchange]

    def _instruments_with_retry(self, exchange: str) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None

        for attempt in range(self.retry.retries + 1):
            try:
                return self.kite.instruments(exchange)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.retry.retries:
                    break
                time.sleep(self.retry.backoff_seconds * (2 ** attempt))

        raise HistoricalDataError(
            f"Instrument fetch failed after {self.retry.retries + 1} attempts for exchange '{exchange}'"
        ) from last_error

    def _historical_with_retry(self, **kwargs: Any) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None

        for attempt in range(self.retry.retries + 1):
            try:
                return self.kite.historical_data(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.retry.retries:
                    break
                time.sleep(self.retry.backoff_seconds * (2 ** attempt))

        raise HistoricalDataError(
            f"Historical fetch failed after {self.retry.retries + 1} attempts"
        ) from last_error

    def _iter_date_windows(
        self, start: datetime, end: datetime, chunk_days: int
    ) -> Iterable[Tuple[datetime, datetime]]:
        cursor = start
        if chunk_days <= 0:
            raise ValueError("chunk_days must be > 0")
        delta = timedelta(days=chunk_days - 1)
        while cursor <= end:
            window_end = min(cursor + delta, end)
            yield cursor, window_end
            cursor = window_end + timedelta(days=1)

    def _chunk_days_for_interval(self, interval: str) -> int:
        # Zerodha day historical endpoint is commonly constrained around 2000 candles per call.
        # Keep day interval chunked safely even if caller passes a larger batch_days.
        if str(interval).strip().lower() == "day":
            return min(self.batch_days, 2000)
        return self.batch_days

    @staticmethod
    def _coerce_datetime(value: DateLike) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        return pd.to_datetime(value).to_pydatetime()
