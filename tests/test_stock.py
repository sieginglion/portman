import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import arrow
import numpy as np

from backend import stock


class AsyncClientStub:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class StockTests(unittest.TestCase):
    def test_daily_values_fills_missing_dates_and_ignores_outside_rows(self):
        values = stock.daily_values(
            arrow.get('2026-01-01'),
            arrow.get('2026-01-03'),
            [
                {'date': '2026-01-03', 'close': 30.0},
                {'date': '2025-12-31', 'close': 99.0},
                {'date': '2026-01-01', 'close': 10.0},
            ],
            'close',
        )

        self.assertEqual(
            values,
            {
                '2026-01-01': 10.0,
                '2026-01-02': 0.0,
                '2026-01-03': 30.0,
            },
        )

    def test_calc_adjusted_handles_no_dividends(self):
        prices = np.array([100.0, 102.0, 101.0])

        adjusted = stock.calc_adjusted(prices, np.zeros(3))

        np.testing.assert_allclose(adjusted, prices)

    def test_calc_adjusted_accumulates_dividends_backwards(self):
        adjusted = stock.calc_adjusted(
            np.array([100.0, 95.0, 90.0]),
            np.array([0.0, 5.0, 9.0]),
        )

        np.testing.assert_allclose(adjusted, [86.0, 86.0, 90.0])

    def test_calc_adjusted_ignores_a_dividend_on_the_first_date(self):
        prices = np.array([100.0, 105.0])

        adjusted = stock.calc_adjusted(prices, np.array([5.0, 0.0]))

        np.testing.assert_allclose(adjusted, prices)

    def test_get_prices_converts_taiwan_prices_to_usd(self):
        client = AsyncClientStub()
        prices = np.array([300.0, 600.0])
        rates = np.array([30.0, 40.0])

        with (
            patch.object(stock, 'AsyncClient', return_value=client),
            patch.object(
                stock, 'get_adjusted', new=AsyncMock(return_value=prices)
            ) as get_adjusted,
            patch.object(stock, 'get_rates', new=AsyncMock(return_value=rates)) as get_rates,
        ):
            result = asyncio.run(stock.get_prices('t', '2330', 2, True))

        np.testing.assert_allclose(result, [10.0, 15.0])
        get_adjusted.assert_awaited_once_with(client, 't', '2330', 2)
        get_rates.assert_awaited_once_with(client, 2)

    def test_get_prices_skips_rates_without_taiwan_usd_conversion(self):
        client = AsyncClientStub()
        prices = np.array([100.0, 101.0])

        for market, to_usd in (('t', False), ('j', True), ('u', True)):
            with (
                patch.object(stock, 'AsyncClient', return_value=client),
                patch.object(
                    stock, 'get_adjusted', new=AsyncMock(return_value=prices)
                ) as get_adjusted,
                patch.object(stock, 'get_rates', new=AsyncMock()) as get_rates,
            ):
                result = asyncio.run(stock.get_prices(market, 'TEST', 2, to_usd))

            np.testing.assert_allclose(result, prices)
            get_adjusted.assert_awaited_once_with(client, market, 'TEST', 2)
            get_rates.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
