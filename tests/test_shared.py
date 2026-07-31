import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from backend import crypto, shared, stock


class GetPricesTests(unittest.TestCase):
    def test_reuses_a_larger_cached_price_series_for_a_shorter_request(self):
        cache_key = (shared.__name__, "get_prices", "c", "CACHE_TEST", False)
        shared.cache.delete(cache_key)
        fetch_prices = AsyncMock(return_value=np.array([100.0, 101.0]))

        try:
            with patch.object(crypto, "get_prices", new=fetch_prices):
                full = asyncio.run(shared.get_prices("c", "CACHE_TEST", 2, False))
                latest = asyncio.run(shared.get_prices("c", "CACHE_TEST", 1, False))
        finally:
            shared.cache.delete(cache_key)

        np.testing.assert_allclose(full, [100.0, 101.0])
        np.testing.assert_allclose(latest, [101.0])
        fetch_prices.assert_awaited_once_with("CACHE_TEST", 2)

    def test_refetches_when_the_cached_series_is_too_short(self):
        cache_key = (shared.__name__, "get_prices", "c", "CACHE_REFETCH", False)
        shared.cache.delete(cache_key)
        fetch_prices = AsyncMock(
            side_effect=[np.array([100.0, 101.0]), np.array([99.0, 100.0, 101.0])]
        )

        try:
            with patch.object(crypto, "get_prices", new=fetch_prices):
                asyncio.run(shared.get_prices("c", "CACHE_REFETCH", 2, False))
                prices = asyncio.run(shared.get_prices("c", "CACHE_REFETCH", 3, False))
        finally:
            shared.cache.delete(cache_key)

        np.testing.assert_allclose(prices, [99.0, 100.0, 101.0])
        self.assertEqual(
            [call.args for call in fetch_prices.await_args_list],
            [("CACHE_REFETCH", 2), ("CACHE_REFETCH", 3)],
        )

    def test_does_not_reuse_prices_with_a_different_currency_conversion(self):
        local_key = (shared.__name__, "get_prices", "t", "CACHE_CURRENCY", False)
        usd_key = (shared.__name__, "get_prices", "t", "CACHE_CURRENCY", True)
        shared.cache.delete(local_key)
        shared.cache.delete(usd_key)
        fetch_prices = AsyncMock(
            side_effect=[np.array([100.0]), np.array([3.0])]
        )

        try:
            with patch.object(stock, "get_prices", new=fetch_prices):
                local = asyncio.run(shared.get_prices("t", "CACHE_CURRENCY", 1, False))
                usd = asyncio.run(shared.get_prices("t", "CACHE_CURRENCY", 1, True))
        finally:
            shared.cache.delete(local_key)
            shared.cache.delete(usd_key)

        np.testing.assert_allclose(local, [100.0])
        np.testing.assert_allclose(usd, [3.0])
        self.assertEqual(
            [call.args for call in fetch_prices.await_args_list],
            [("t", "CACHE_CURRENCY", 1, False), ("t", "CACHE_CURRENCY", 1, True)],
        )

    def test_does_not_reuse_prices_for_a_different_symbol(self):
        first_key = (shared.__name__, "get_prices", "c", "CACHE_SYMBOL_A", False)
        second_key = (shared.__name__, "get_prices", "c", "CACHE_SYMBOL_B", False)
        shared.cache.delete(first_key)
        shared.cache.delete(second_key)
        fetch_prices = AsyncMock(
            side_effect=[np.array([100.0]), np.array([200.0])]
        )

        try:
            with patch.object(crypto, "get_prices", new=fetch_prices):
                first = asyncio.run(shared.get_prices("c", "CACHE_SYMBOL_A", 1, False))
                second = asyncio.run(shared.get_prices("c", "CACHE_SYMBOL_B", 1, False))
        finally:
            shared.cache.delete(first_key)
            shared.cache.delete(second_key)

        np.testing.assert_allclose(first, [100.0])
        np.testing.assert_allclose(second, [200.0])
        self.assertEqual(
            [call.args for call in fetch_prices.await_args_list],
            [("CACHE_SYMBOL_A", 1), ("CACHE_SYMBOL_B", 1)],
        )

    def test_does_not_reuse_prices_for_a_different_market(self):
        crypto_key = (shared.__name__, "get_prices", "c", "CACHE_MARKET", False)
        stock_key = (shared.__name__, "get_prices", "t", "CACHE_MARKET", False)
        shared.cache.delete(crypto_key)
        shared.cache.delete(stock_key)
        crypto_prices = AsyncMock(return_value=np.array([100.0]))
        stock_prices = AsyncMock(return_value=np.array([200.0]))

        try:
            with (
                patch.object(crypto, "get_prices", new=crypto_prices),
                patch.object(stock, "get_prices", new=stock_prices),
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
        stock_prices.assert_awaited_once_with("t", "CACHE_MARKET", 1, False)
