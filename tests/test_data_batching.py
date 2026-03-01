from datetime import datetime

from zerodha_portfolio.data import HistoricalDataClient


class FakeKite:
    def __init__(self):
        self.calls = []

    def instruments(self, exchange):
        return [{"tradingsymbol": "RELIANCE", "instrument_token": 738561}]

    def historical_data(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "date": kwargs["from_date"],
                "open": 1,
                "high": 2,
                "low": 0,
                "close": 1,
                "volume": 100,
            }
        ]


def test_day_interval_caps_chunk_at_2000():
    kite = FakeKite()
    client = HistoricalDataClient(kite, batch_days=5000, throttle_seconds=0)

    _ = client.fetch_by_token(
        instrument_token=738561,
        from_date="2020-01-01",
        to_date="2025-12-31",
        interval="day",
    )

    # More than one chunk expected for ~6 years when day chunk is capped at 2000.
    assert len(kite.calls) >= 2

    # First window should be exactly 2000 calendar days inclusive.
    first_from = kite.calls[0]["from_date"]
    first_to = kite.calls[0]["to_date"]
    assert first_from == datetime(2020, 1, 1)
    assert (first_to - first_from).days == 1999
