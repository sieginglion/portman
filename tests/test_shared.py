import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from backend import crypto, shared, stock


def dated_prices(values, end="2026-09-30"):
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values)))


class TestGetPrices:
    @pytest.mark.parametrize(
        ("fetch_day", "later_day"),
        [("2026-09-30", "2026-10-01"), ("2026-10-30", "2026-11-02")],
    )
    def test_cached_prices_keep_their_dates_after_midnight(self, fetch_day, later_day):
        key = (shared.__name__, "get_price_series", "u", "CACHE_DATES", False, True)
        shared.cache.delete(key)
        fetch_prices = AsyncMock(
            return_value=dated_prices([129.94, 134.01], end=fetch_day)
        )
        try:
            with (
                patch.object(stock, "get_price_series", new=fetch_prices),
                patch.object(
                    shared.pd.Timestamp,
                    "now",
                    return_value=pd.Timestamp(fetch_day, tz="America/New_York"),
                ) as clock,
            ):
                original = asyncio.run(
                    shared.get_price_series("u", "CACHE_DATES", 2, False)
                )
                clock.return_value = pd.Timestamp(later_day, tz="America/New_York")
                cached = asyncio.run(
                    shared.get_price_series("u", "CACHE_DATES", 1, False)
                )
                array = asyncio.run(shared.get_prices("u", "CACHE_DATES", 2, False))
        finally:
            shared.cache.delete(key)

        pd.testing.assert_series_equal(cached, original.iloc[-1:])
        assert original.index[-1] == pd.Timestamp(fetch_day)
        np.testing.assert_array_equal(array, [129.94, 134.01])
        fetch_prices.assert_awaited_once_with(
            "u", "CACHE_DATES", 2, False, adjust_dividends=True
        )

    def test_reuses_a_larger_cached_price_series_for_a_shorter_request(self):
        cache_key = (
            shared.__name__,
            "get_price_series",
            "c",
            "CACHE_TEST",
            False,
            True,
        )
        shared.cache.delete(cache_key)
        fetch_prices = AsyncMock(return_value=dated_prices([100.0, 101.0]))

        try:
            with patch.object(crypto, "get_price_series", new=fetch_prices):
                full = asyncio.run(shared.get_prices("c", "CACHE_TEST", 2, False))
                latest = asyncio.run(shared.get_prices("c", "CACHE_TEST", 1, False))
        finally:
            shared.cache.delete(cache_key)

        np.testing.assert_allclose(full, [100.0, 101.0])
        np.testing.assert_allclose(latest, [101.0])
        fetch_prices.assert_awaited_once_with("CACHE_TEST", 2)

    def test_refetches_when_the_cached_series_is_too_short(self):
        cache_key = (
            shared.__name__,
            "get_price_series",
            "c",
            "CACHE_REFETCH",
            False,
            True,
        )
        shared.cache.delete(cache_key)
        fetch_prices = AsyncMock(
            side_effect=[
                dated_prices([100.0, 101.0]),
                dated_prices([99.0, 100.0, 101.0]),
            ]
        )

        try:
            with patch.object(crypto, "get_price_series", new=fetch_prices):
                asyncio.run(shared.get_prices("c", "CACHE_REFETCH", 2, False))
                prices = asyncio.run(shared.get_prices("c", "CACHE_REFETCH", 3, False))
        finally:
            shared.cache.delete(cache_key)

        np.testing.assert_allclose(prices, [99.0, 100.0, 101.0])
        assert [call.args for call in fetch_prices.await_args_list] == [
            ("CACHE_REFETCH", 2),
            ("CACHE_REFETCH", 3),
        ]

    def test_does_not_reuse_prices_with_a_different_currency_conversion(self):
        local_key = (
            shared.__name__,
            "get_price_series",
            "t",
            "CACHE_CURRENCY",
            False,
            True,
        )
        usd_key = (
            shared.__name__,
            "get_price_series",
            "t",
            "CACHE_CURRENCY",
            True,
            True,
        )
        shared.cache.delete(local_key)
        shared.cache.delete(usd_key)
        fetch_prices = AsyncMock(
            side_effect=[dated_prices([100.0]), dated_prices([3.0])]
        )

        try:
            with patch.object(stock, "get_price_series", new=fetch_prices):
                local = asyncio.run(shared.get_prices("t", "CACHE_CURRENCY", 1, False))
                usd = asyncio.run(shared.get_prices("t", "CACHE_CURRENCY", 1, True))
        finally:
            shared.cache.delete(local_key)
            shared.cache.delete(usd_key)

        np.testing.assert_allclose(local, [100.0])
        np.testing.assert_allclose(usd, [3.0])
        assert [call.args for call in fetch_prices.await_args_list] == [
            ("t", "CACHE_CURRENCY", 1, False),
            ("t", "CACHE_CURRENCY", 1, True),
        ]

    def test_does_not_reuse_prices_for_a_different_symbol(self):
        first_key = (
            shared.__name__,
            "get_price_series",
            "c",
            "CACHE_SYMBOL_A",
            False,
            True,
        )
        second_key = (
            shared.__name__,
            "get_price_series",
            "c",
            "CACHE_SYMBOL_B",
            False,
            True,
        )
        shared.cache.delete(first_key)
        shared.cache.delete(second_key)
        fetch_prices = AsyncMock(
            side_effect=[dated_prices([100.0]), dated_prices([200.0])]
        )

        try:
            with patch.object(crypto, "get_price_series", new=fetch_prices):
                first = asyncio.run(shared.get_prices("c", "CACHE_SYMBOL_A", 1, False))
                second = asyncio.run(shared.get_prices("c", "CACHE_SYMBOL_B", 1, False))
        finally:
            shared.cache.delete(first_key)
            shared.cache.delete(second_key)

        np.testing.assert_allclose(first, [100.0])
        np.testing.assert_allclose(second, [200.0])
        assert [call.args for call in fetch_prices.await_args_list] == [
            ("CACHE_SYMBOL_A", 1),
            ("CACHE_SYMBOL_B", 1),
        ]

    def test_does_not_reuse_prices_for_a_different_market(self):
        crypto_key = (
            shared.__name__,
            "get_price_series",
            "c",
            "CACHE_MARKET",
            False,
            True,
        )
        stock_key = (
            shared.__name__,
            "get_price_series",
            "t",
            "CACHE_MARKET",
            False,
            True,
        )
        shared.cache.delete(crypto_key)
        shared.cache.delete(stock_key)
        crypto_prices = AsyncMock(return_value=dated_prices([100.0]))
        stock_prices = AsyncMock(return_value=dated_prices([200.0]))

        try:
            with (
                patch.object(crypto, "get_price_series", new=crypto_prices),
                patch.object(stock, "get_price_series", new=stock_prices),
            ):
                crypto_result = asyncio.run(
                    shared.get_prices("c", "CACHE_MARKET", 1, False)
                )
                stock_result = asyncio.run(
                    shared.get_prices("t", "CACHE_MARKET", 1, False)
                )
        finally:
            shared.cache.delete(crypto_key)
            shared.cache.delete(stock_key)

        np.testing.assert_allclose(crypto_result, [100.0])
        np.testing.assert_allclose(stock_result, [200.0])
        crypto_prices.assert_awaited_once_with("CACHE_MARKET", 1)
        stock_prices.assert_awaited_once_with(
            "t", "CACHE_MARKET", 1, False, adjust_dividends=True
        )
