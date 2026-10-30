import asyncio
from functools import partial
from unittest.mock import AsyncMock, patch

import arrow
import httpx
import numpy as np
import pandas as pd
import pytest
from diskcache import Cache

from backend import crypto, shared, stock, valuation


@pytest.mark.parametrize("market", ["u", "j", "t", "c"])
def test_provider_dates_survive_midnight_cache_and_valuation(market, tmp_path):
    timezone = shared.MARKET_TO_TIMEZONE[market]
    provider_now = arrow.get("2026-10-01T00:00:01", tzinfo=timezone)
    before_midnight = pd.Timestamp("2026-09-30T23:59:59", tz=timezone)
    dates = pd.date_range(end="2026-10-01", periods=91)
    closes = {
        dates[0].strftime("%Y-%m-%d"): 129.0,
        "2026-09-29": 129.94,
        "2026-09-30": 134.01,
        "2026-10-01": 137.76,
    }
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/klines"):
            rows = [
                [arrow.get(date).int_timestamp * 1000, 0, 0, 0, str(value)]
                for date, value in sorted(closes.items())
            ]
            return httpx.Response(200, json=rows)
        if "/stock_dividend/" in request.url.path:
            return httpx.Response(200, json={"historical": []})
        return httpx.Response(
            200,
            json={
                "historical": [
                    {"date": date, "close": close}
                    for date, close in reversed(closes.items())
                ]
            },
        )

    client = partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    xps = {
        "rps": pd.Series(
            [10.0, 12.0], index=pd.to_datetime(["2026-07-01", "2026-10-01"])
        ),
        "eps": pd.Series(
            [2.0, 1.5], index=pd.to_datetime(["2026-07-01", "2026-10-01"])
        ),
    }
    with (
        Cache(tmp_path / "prices") as cache,
        patch.object(shared, "cache", cache),
        patch.object(stock, "AsyncClient", client),
        patch.object(crypto, "AsyncClient", client),
        patch.object(stock, "get_today_dividend", new=AsyncMock(return_value=0.0)),
        patch.object(arrow, "now", return_value=provider_now) as provider_clock,
        patch.object(pd.Timestamp, "now", return_value=before_midnight) as reader_clock,
        patch.object(valuation, "fetch_xps", new=AsyncMock(return_value=xps)),
        patch.object(
            valuation,
            "fetch_token_terminal_monthly_xps",
            new=AsyncMock(return_value=xps),
        ),
    ):
        # The request begins before midnight; the provider selects its window
        # after midnight. Old undated cache entries must not be reused.
        cache.set(
            (shared.__name__, "get_prices", market, "DATES_TEST", False), np.zeros(91)
        )
        first = asyncio.run(valuation.calc_px(market, "DATES_TEST", 1))
        request_count = len(requests)
        provider_clock.return_value = provider_now.shift(days=1)
        reader_clock.return_value = before_midnight + pd.Timedelta(days=2)
        cached = asyncio.run(valuation.calc_px(market, "DATES_TEST", 1))
        shorter = asyncio.run(shared.get_price_series(market, "DATES_TEST", 3, False))
        array = asyncio.run(shared.get_prices(market, "DATES_TEST", 3, False))
        provider_clock.assert_any_call(timezone)

    expected = pd.Series(closes)
    expected.index = pd.to_datetime(expected.index)
    expected = expected.reindex(dates).ffill()
    pd.testing.assert_index_equal(first.index, dates)
    pd.testing.assert_frame_equal(cached, first)
    assert len(requests) == request_count
    pd.testing.assert_series_equal(shorter, expected.iloc[-3:], check_freq=False)
    assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(array, [129.94, 134.01, 137.76])
    # Missing sessions retain their own calendar dates and carry the last close.
    assert shorter.index.tolist() == list(pd.date_range("2026-09-29", periods=3))
    assert first.loc["2026-09-26", "ps"] == pytest.approx(129.0 / 10)
    assert first.loc["2026-09-27", "ps"] == pytest.approx(129.0 / 10)
    assert first.loc["2026-09-29", "ps"] == pytest.approx(129.94 / 10)
    assert first.loc["2026-09-30", "ps"] == pytest.approx(134.01 / 10)
    assert first.loc["2026-10-01", "ps"] == pytest.approx(137.76 / 12)
    assert first.loc["2026-09-29", "pe"] == pytest.approx(129.94 / 2)
    assert first.loc["2026-10-01", "pe"] == pytest.approx(137.76 / 1.5)
