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


class ResponseStub:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class RequestClientStub:
    def __init__(self, payload):
        self.response = ResponseStub(payload)
        self.requests = []

    async def get(self, url, params):
        self.requests.append((url, params))
        return self.response


class StockTests(unittest.TestCase):
    def test_daily_values_fills_missing_dates_and_ignores_outside_rows(self):
        values = stock.daily_values(
            arrow.get("2026-01-01"),
            arrow.get("2026-01-03"),
            [
                {"date": "2026-01-03", "close": 30.0},
                {"date": "2025-12-31", "close": 99.0},
                {"date": "2026-01-01", "close": 10.0},
            ],
            "close",
        )

        self.assertEqual(
            values,
            {
                "2026-01-01": 10.0,
                "2026-01-02": 0.0,
                "2026-01-03": 30.0,
            },
        )

    def test_get_fmp_series_fills_gaps_and_applies_taiwan_limit(self):
        client = RequestClientStub(
            {
                "historical": [
                    {"date": "2026-01-01", "close": 100.0},
                    {"date": "2026-01-03", "close": 120.0},
                    {"date": "2026-01-04", "close": 121.0},
                ]
            }
        )
        start = arrow.get("2026-01-01")
        end = arrow.get("2026-01-04")

        result = asyncio.run(
            stock.get_fmp_series(client, "2330.TW", start, end, 4, True)
        )

        np.testing.assert_allclose(result, [100.0, 100.0, 121.0, 121.0])
        self.assertEqual(
            client.requests,
            [
                (
                    "https://financialmodelingprep.com/api/v3/historical-price-full/2330.TW",
                    {"from": "2026-01-01", "serietype": "line"},
                )
            ],
        )

    def test_get_fmp_unadjusted_uses_market_suffix_and_limit(self):
        client = object()
        get_fmp_series = AsyncMock(return_value=np.array([100.0, 101.0]))

        with patch.object(stock, "get_fmp_series", new=get_fmp_series):
            taiwan = asyncio.run(stock.get_fmp_unadjusted(client, "t", "2330", 2))
            japan = asyncio.run(stock.get_fmp_unadjusted(client, "j", "7203", 2))

        np.testing.assert_allclose(taiwan, [100.0, 101.0])
        np.testing.assert_allclose(japan, [100.0, 101.0])
        self.assertEqual(
            [
                (call.args[0], call.args[1], call.args[4], call.args[5])
                for call in get_fmp_series.await_args_list
            ],
            [(client, "2330.TW", 2, True), (client, "7203.T", 2, False)],
        )

    def test_get_fmp_unadjusted_reports_the_symbol_when_data_is_invalid(self):
        get_fmp_series = AsyncMock(side_effect=AssertionError)

        with (
            patch.object(stock, "get_fmp_series", new=get_fmp_series),
            self.assertRaisesRegex(AssertionError, "2330"),
        ):
            asyncio.run(stock.get_fmp_unadjusted(object(), "t", "2330", 2))

    def test_market_window_uses_market_timezone_and_lookback(self):
        now = arrow.get("2026-01-15T12:00:00+08:00")

        for market, timezone in (
            ("j", "Asia/Tokyo"),
            ("t", "Asia/Taipei"),
            ("u", "America/New_York"),
        ):
            with self.subTest(market=market):
                with patch.object(stock.arrow, "now", return_value=now) as get_now:
                    start, end = stock._market_window(market, 2)

                get_now.assert_called_once_with(timezone)
                self.assertEqual(end, now)
                self.assertEqual(start, now.shift(days=-15))

    def test_get_rates_uses_usdtwd_series(self):
        client = object()
        rates = np.array([31.0, 32.0])
        get_fmp_series = AsyncMock(return_value=rates)

        with patch.object(stock, "get_fmp_series", new=get_fmp_series):
            result = asyncio.run(stock.get_rates(client, 2))

        np.testing.assert_allclose(result, rates)
        args = get_fmp_series.await_args.args
        self.assertEqual((args[0], args[1], args[4]), (client, "USDTWD", 2))

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

    def test_get_adjusted_uses_fmp(self):
        client = object()
        prices = np.array([100.0, 101.0])
        dividends = np.zeros(2)
        get_fmp_unadjusted = AsyncMock(return_value=prices)
        get_fmp_dividends = AsyncMock(return_value=dividends)

        with (
            patch.object(stock, "get_fmp_unadjusted", new=get_fmp_unadjusted),
            patch.object(stock, "get_fmp_dividends", new=get_fmp_dividends),
        ):
            result = asyncio.run(stock.get_adjusted(client, "u", "AAPL", 2))

        np.testing.assert_allclose(result, prices)
        get_fmp_unadjusted.assert_awaited_once_with(client, "u", "AAPL", 2)
        get_fmp_dividends.assert_awaited_once_with(client, "u", "AAPL", 2)

    def test_get_prices_converts_taiwan_prices_to_usd(self):
        client = AsyncClientStub()
        prices = np.array([300.0, 600.0])
        rates = np.array([30.0, 40.0])

        with (
            patch.object(stock, "AsyncClient", return_value=client),
            patch.object(
                stock, "get_adjusted", new=AsyncMock(return_value=prices)
            ) as get_adjusted,
            patch.object(
                stock, "get_rates", new=AsyncMock(return_value=rates)
            ) as get_rates,
        ):
            result = asyncio.run(stock.get_prices("t", "2330", 2, True))

        np.testing.assert_allclose(result, [10.0, 15.0])
        get_adjusted.assert_awaited_once_with(client, "t", "2330", 2)
        get_rates.assert_awaited_once_with(client, 2)

    def test_get_prices_skips_rates_without_taiwan_usd_conversion(self):
        client = AsyncClientStub()
        prices = np.array([100.0, 101.0])

        for market, to_usd in (("t", False), ("j", True), ("u", True)):
            with (
                patch.object(stock, "AsyncClient", return_value=client),
                patch.object(
                    stock, "get_adjusted", new=AsyncMock(return_value=prices)
                ) as get_adjusted,
                patch.object(stock, "get_rates", new=AsyncMock()) as get_rates,
            ):
                result = asyncio.run(stock.get_prices(market, "TEST", 2, to_usd))

            np.testing.assert_allclose(result, prices)
            get_adjusted.assert_awaited_once_with(client, market, "TEST", 2)
            get_rates.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
