import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

from backend import valuation


class TestFinMindValuation:
    def test_complete_and_roll_xps_keeps_each_complete_column(self):
        xps = valuation.pd.DataFrame(
            {
                "rps": [1.0, 2.0, 3.0, 4.0],
                "eps": [1.0, None, 3.0, 4.0],
            },
            index=valuation.pd.date_range("2025-01-01", periods=4, freq="QS"),
        )

        result = valuation.complete_and_roll_xps(
            {str(column): values for column, values in xps.items()}, 4
        )

        assert list(result) == ["rps"]
        assert result["rps"].tolist() == [10.0]

    def test_complete_and_roll_xps_rejects_when_neither_column_is_complete(self):
        xps = valuation.pd.DataFrame({"rps": [1.0, None], "eps": [None, 1.0]})

        with pytest.raises(ValueError, match="neither a complete"):
            valuation.complete_and_roll_xps(
                {str(column): values for column, values in xps.items()}, 2
            )

    def test_complete_and_roll_xps_preserves_each_series_dates(self):
        rps = valuation.pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=valuation.pd.date_range("2025-01-01", periods=5, freq="QS"),
        )
        eps = valuation.pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            index=rps.index + valuation.pd.Timedelta(days=15),
        )

        result = valuation.complete_and_roll_xps({"rps": rps, "eps": eps}, 4)

        assert result["rps"].index.equals(rps.index[-2:])
        assert result["eps"].index.equals(eps.index[-2:])
        assert result["rps"].tolist() == [10.0, 14.0]
        assert result["eps"].tolist() == pytest.approx([1.0, 1.4])

    def test_fetch_finmind_taiwan_rows_uses_bearer_authentication(self):
        cached_get = AsyncMock(return_value='{"data": [{"type": "EPS"}]}')

        with patch.object(valuation.shared, "cached_get", cached_get):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_rows(
                    "TaiwanStockFinancialStatements", "2330", "2025-01-01"
                )
            )

        cached_get.assert_awaited_once_with(
            valuation.FINMIND_API_URL,
            {
                "dataset": "TaiwanStockFinancialStatements",
                "data_id": "2330",
                "start_date": "2025-01-01",
            },
            headers={"Authorization": f"Bearer {valuation.FINMIND_API_KEY}"},
        )
        assert rows == [{"type": "EPS"}]

    def test_pivot_finmind_rows_filters_invalid_rows_and_keeps_last_duplicate(self):
        rows = [
            {"date": "2025-03-31", "type": "Revenue", "value": "100"},
            {"date": "2025-03-31", "type": "Revenue", "value": "110"},
            {"date": "2025-03-31", "type": "EPS", "value": "invalid"},
            {"date": None, "type": "Revenue", "value": 120},
            {"date": "2025-06-30", "type": None, "value": 120},
            {"date": "2025-06-30", "type": "Revenue", "value": None},
        ]

        result = valuation.pivot_finmind_taiwan_rows(rows)

        assert result.to_dict("index") == {"2025-03-31": {"Revenue": 110.0}}
        assert valuation.pivot_finmind_taiwan_rows([]).empty

    def test_normalize_income_statement_rows_canonicalizes_dates(self):
        rows = valuation.normalize_income_statement_rows(
            [
                {"date": "2025-09-30T00:00:00.000Z", "revenue": 100},
                {"date": "not-a-date", "revenue": 200},
            ],
            field_map={"revenue": "revenue"},
            date_field="date",
        )

        assert rows == {"2025-09-30": {"revenue": 100}}

    def test_normalize_income_statement_rows_keeps_metadata(self):
        rows = valuation.normalize_income_statement_rows(
            [
                {
                    "date": "2025-09-30",
                    "revenue": 100,
                    "cikValue": "320193",
                    "fiscalQuarter": "Q3",
                }
            ],
            field_map={"revenue": "revenue"},
            date_field="date",
            metadata_fields={"cik": "cikValue", "quarter": "fiscalQuarter"},
        )

        assert rows == {
            "2025-09-30": {"revenue": 100, "cik": "320193", "quarter": "Q3"}
        }

    def test_quarterly_start_date_uses_market_timezone_and_buffers(self):
        now = valuation.pd.Timestamp("2026-07-22 15:30:00", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now) as now_mock:
            assert (
                valuation.quarterly_start_date("u", limit=4, quarter_buffer=1)
                == "2025-04-22"
            )
            assert (
                valuation.quarterly_start_date(
                    "t", limit=4, quarter_buffer=1, month_buffer=2
                )
                == "2025-02-22"
            )

        now_mock.assert_has_calls(
            [
                call(tz=valuation.shared.MARKET_TO_TIMEZONE["u"]),
                call(tz=valuation.shared.MARKET_TO_TIMEZONE["t"]),
            ]
        )

    @pytest.mark.parametrize(
        ("market", "utc_now", "month_buffer", "expected"),
        [
            ("t", "2026-07-21T16:30:00Z", 2, "2025-02-22"),
            ("u", "2026-07-22T00:30:00Z", 0, "2025-04-21"),
            ("t", "2026-01-31T16:30:00Z", 2, "2024-09-01"),
            ("u", "2026-02-01T00:30:00Z", 0, "2024-10-31"),
        ],
    )
    def test_quarterly_start_date_preserves_market_date_at_utc_boundary(
        self, market, utc_now, month_buffer, expected
    ):
        now = valuation.pd.Timestamp(utc_now)
        with patch.object(
            valuation.pd.Timestamp,
            "now",
            side_effect=lambda *, tz: now.tz_convert(tz),
        ) as now_mock:
            result = valuation.quarterly_start_date(
                market, limit=4, quarter_buffer=1, month_buffer=month_buffer
            )

        assert result == expected
        now_mock.assert_called_once_with(tz=valuation.shared.MARKET_TO_TIMEZONE[market])

    def test_build_xps_series_returns_ttm_values_with_quarter_dates(self):
        quarters: valuation.ResolvedQuarters = {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": 10,
                "epsDiluted": 1.0,
            },
            "2025-06-30": {
                "revenue": 110,
                "weightedAverageShsOutDil": 10,
                "epsDiluted": 1.1,
            },
            "2025-09-30": {
                "revenue": 120,
                "weightedAverageShsOutDil": 12,
                "epsDiluted": 1.2,
            },
            "2025-12-31": {
                "revenue": 130,
                "weightedAverageShsOutDil": 13,
                "epsDiluted": 1.3,
            },
            "2026-03-31": {
                "revenue": 140,
                "weightedAverageShsOutDil": 14,
                "epsDiluted": 1.4,
            },
        }

        frame = valuation.build_xps_series(valuation.split_xps_quarters(quarters, True))

        assert list(frame) == ["rps", "eps"]
        for values in frame.values():
            assert valuation.pd.DatetimeIndex(values.index).strftime(
                "%Y-%m-%d"
            ).tolist() == ["2026-01-01", "2026-04-01"]
        assert frame["rps"].tolist() == [41.0, 41.0]
        assert frame["eps"].tolist() == [4.6, 5.0]

        without_eps = valuation.build_xps_series(
            valuation.split_xps_quarters(quarters, False)
        )
        assert list(without_eps) == ["rps"]
        assert valuation.pd.DatetimeIndex(without_eps["rps"].index).strftime(
            "%Y-%m-%d"
        ).tolist() == [
            "2026-01-01",
            "2026-04-01",
        ]
        assert without_eps["rps"].tolist() == [41.0, 41.0]

    def test_build_xps_series_keeps_each_complete_metric_independently(self):
        quarters = {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.0,
            },
            "2025-06-30": {
                "revenue": 110,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.1,
            },
            "2025-09-30": {
                "revenue": 120,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.2,
            },
            "2025-12-31": {
                "revenue": 130,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.3,
            },
        }

        frame = valuation.build_xps_series(valuation.split_xps_quarters(quarters, True))

        assert list(frame) == ["eps"]
        assert frame["eps"].tolist() == [4.6]

    def test_calc_px_aligns_each_series_and_preserves_its_update_dates(self):
        xps = {
            "rps": valuation.pd.Series(
                [10.0, 14.0],
                index=valuation.pd.to_datetime(["2025-12-31", "2026-01-03"]),
            ),
            "eps": valuation.pd.Series(
                [2.0, 3.0],
                index=valuation.pd.to_datetime(["2026-01-01", "2026-01-04"]),
            ),
        }
        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    [20, 24, 28, 30],
                    index=valuation.pd.date_range("2026-01-01", periods=4),
                ),
            ),
            patch.object(valuation, "fetch_xps", return_value=xps),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-01-04", tz="UTC"),
            ),
        ):
            px = asyncio.run(valuation.calc_px("u", "A", 1))

        assert px["ps"].tolist() == pytest.approx([2, 2.4, 2, 30 / 14])
        assert px["pe"].tolist() == [10, 12, 14, 10]
        assert px.attrs["ps_dates"].equals(xps["rps"].index)
        assert px.attrs["pe_dates"].equals(xps["eps"].index)

    def test_calc_px_keeps_cached_price_dates_across_a_quarter_boundary(self):
        prices = valuation.pd.Series(
            [129.94, 134.01, 137.76],
            index=valuation.pd.date_range("2026-09-29", periods=3),
        )
        xps = {
            "rps": valuation.pd.Series(
                [10.0, 12.0],
                index=valuation.pd.to_datetime(["2026-07-01", "2026-10-01"]),
            ),
            "eps": valuation.pd.Series(
                [2.0, 1.5],
                index=valuation.pd.to_datetime(["2026-07-01", "2026-10-01"]),
            ),
        }
        with (
            patch.object(valuation.shared, "get_price_series", return_value=prices),
            patch.object(valuation, "fetch_xps", return_value=xps),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-10-02", tz="UTC"),
            ),
        ):
            px = asyncio.run(valuation.calc_px("u", "NOW", 2))

        assert px.index.equals(prices.index)
        assert px["ps"].tolist() == pytest.approx(
            [129.94 / 10, 134.01 / 10, 137.76 / 12]
        )
        assert px["pe"].tolist() == pytest.approx(
            [129.94 / 2, 134.01 / 2, 137.76 / 1.5]
        )

    def test_calc_px_allows_rps_with_insufficient_initial_coverage(self):
        prices = [20.0, 21.0, 22.0, 23.0]
        xps = valuation.pd.DataFrame(
            {
                "rps": [None, 10.0],
                "eps": [2.0, 2.0],
            },
            index=valuation.pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    prices,
                    index=valuation.pd.date_range("2026-01-01", periods=len(prices)),
                ),
            ),
            patch.object(
                valuation,
                "fetch_xps",
                return_value={str(column): values for column, values in xps.items()},
            ),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-01-04", tz="UTC"),
            ),
        ):
            px = asyncio.run(valuation.calc_px("u", "A", 1))

        assert valuation.pd.isna(px["ps"].iloc[0])
        assert px["ps"].iloc[1:].tolist() == [2.1, 2.2, 2.3]
        assert px["pe"].tolist() == [10.0, 10.5, 11.0, 11.5]
        assert px.attrs["ps_dates"].equals(xps["rps"].index)
        assert px.attrs["pe_dates"].equals(xps["eps"].index)

    def test_calc_px_allows_eps_with_insufficient_initial_coverage(self):
        prices = [20.0, 21.0, 22.0, 23.0]
        xps = valuation.pd.DataFrame(
            {
                "rps": [10.0, 10.0],
                "eps": [None, 2.0],
            },
            index=valuation.pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    prices,
                    index=valuation.pd.date_range("2026-01-01", periods=len(prices)),
                ),
            ),
            patch.object(
                valuation,
                "fetch_xps",
                return_value={str(column): values for column, values in xps.items()},
            ),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-01-04", tz="UTC"),
            ),
        ):
            px = asyncio.run(valuation.calc_px("u", "A", 1))

        assert px["ps"].tolist() == [2.0, 2.1, 2.2, 2.3]
        assert valuation.pd.isna(px["pe"].iloc[0])
        assert px["pe"].iloc[1:].tolist() == [10.5, 11.0, 11.5]

    def test_calc_px_rejects_both_metrics_with_insufficient_initial_coverage(self):
        prices = [20.0, 21.0]
        xps = valuation.pd.DataFrame(
            {"rps": [None, 10.0], "eps": [None, 2.0]},
            index=valuation.pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    prices,
                    index=valuation.pd.date_range("2026-01-01", periods=len(prices)),
                ),
            ),
            patch.object(
                valuation,
                "fetch_xps",
                return_value={str(column): values for column, values in xps.items()},
            ),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-01-02", tz="UTC"),
            ),
            pytest.raises(ValueError, match="initial RPS and EPS values are missing"),
        ):
            asyncio.run(valuation.calc_px("u", "A", 1))

    def test_calc_px_rejects_missing_initial_price(self):
        xps = valuation.pd.DataFrame(
            {"rps": [10.0], "eps": [2.0]},
            index=valuation.pd.to_datetime(["2026-01-01"]),
        )

        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    [None, 21.0],
                    index=valuation.pd.date_range("2026-01-01", periods=2),
                ),
            ),
            patch.object(
                valuation,
                "fetch_xps",
                return_value={str(column): values for column, values in xps.items()},
            ),
            patch.object(
                valuation.pd.Timestamp,
                "now",
                return_value=valuation.pd.Timestamp("2026-01-02", tz="UTC"),
            ),
            pytest.raises(ValueError, match="initial price is missing"),
        ):
            asyncio.run(valuation.calc_px("u", "A", 1))

    def test_income_statement_rows_are_kept_without_balance_sheet_data(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {"date": "2025-03-31", "type": "Revenue", "value": 100},
                {"date": "2025-03-31", "type": "EPS", "value": 1.5},
            ],
            [],
        )

        assert rows == {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.5,
            }
        }

    def test_normalize_keeps_eps_only_rows_for_central_validation(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [{"date": "2025-03-31", "type": "EPS", "value": 1.5}],
            [],
        )

        assert rows == {
            "2025-03-31": {
                "revenue": None,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.5,
            }
        }

    def test_normalize_ignores_balance_sheet_only_dates(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {"date": "2025-03-31", "type": "Revenue", "value": 100},
                {"date": "2025-03-31", "type": "EPS", "value": 1.5},
            ],
            [
                {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
                {"date": "2025-06-30", "type": "OrdinaryShare", "value": 300},
            ],
        )

        assert rows == {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": 20,
                "epsDiluted": 1.5,
            }
        }

    def test_normalize_leaves_endpoint_share_gaps_missing(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {"date": "2025-03-31", "type": "Revenue", "value": 100},
                {"date": "2025-06-30", "type": "Revenue", "value": 110},
                {"date": "2025-09-30", "type": "Revenue", "value": 120},
            ],
            [{"date": "2025-06-30", "type": "OrdinaryShare", "value": 200}],
        )

        assert [row["weightedAverageShsOutDil"] for row in rows.values()] == [
            None,
            20,
            None,
        ]

    def test_share_interpolation_keeps_endpoints_missing(self):
        shares = valuation.pd.Series([None, 1, None, 3, None], dtype=float)

        with patch.object(valuation.logger, "warning") as warning:
            filled = valuation.fill_finmind_missing_ordinary_shares(shares)

        valuation.pd.testing.assert_series_equal(
            filled, valuation.pd.Series([None, 1, 2, 3, None], dtype=float)
        )
        warning.assert_called_once()
        assert warning.call_args.args[1:] == (2, "linear interpolation", 2.0)

    @pytest.mark.parametrize("reverse_rows", [False, True])
    def test_fetch_xps_interpolates_an_isolated_taiwan_share_gap(self, reverse_rows):
        financial_rows = [
            {"date": date, "type": field, "value": value}
            for date, revenue, eps in [
                ("2025-03-31", 100, 1.0),
                ("2025-06-30", 150, 1.5),
                ("2025-09-30", 200, 2.0),
                ("2025-12-31", 200, 2.0),
            ]
            for field, value in [("Revenue", revenue), ("EPS", eps)]
        ]
        balance_rows = [
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
            {"date": "2025-09-30", "type": "OrdinaryShare", "value": 400},
            {"date": "2025-12-31", "type": "OrdinaryShare", "value": 400},
        ]
        if reverse_rows:
            financial_rows.reverse()
            balance_rows.reverse()

        with (
            patch.object(
                valuation,
                "fetch_finmind_taiwan_rows",
                new=AsyncMock(side_effect=[financial_rows, balance_rows]),
            ),
            patch.object(valuation.logger, "warning") as warning,
        ):
            frame = asyncio.run(valuation.fetch_xps("t", "2330", q=1))

        assert frame["rps"].tolist() == [20.0]
        assert frame["eps"].tolist() == [6.5]
        warning.assert_called_once()
        assert warning.call_args.args[1:] == (
            "2025-06-30",
            "linear interpolation",
            300.0,
        )

    @pytest.mark.parametrize(
        "gap_positions", [(0,), (3,), (0, 3), (0, 1), (1, 2), (2, 3)]
    )
    def test_unfilled_share_gaps_keep_complete_taiwan_eps(self, gap_positions):
        dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
        financial_rows = [
            {"date": date, "type": field, "value": value}
            for date in dates
            for field, value in [("Revenue", 100), ("EPS", 1.5)]
        ]
        balance_rows = [
            {"date": date, "type": "OrdinaryShare", "value": 200}
            for index, date in enumerate(dates)
            if index not in gap_positions
        ]

        with patch.object(valuation.logger, "warning") as warning:
            rows = valuation.normalize_finmind_taiwan_income_statement_rows(
                financial_rows, balance_rows
            )
            frame = valuation.build_xps_series(valuation.split_xps_quarters(rows, True))

        assert all(
            rows[dates[index]]["weightedAverageShsOutDil"] is None
            for index in gap_positions
        )
        assert list(frame) == ["eps"]
        assert frame["eps"].tolist() == [6.0]
        warning.assert_not_called()

    @pytest.mark.parametrize("missing_value", [0, -1, float("inf"), float("nan")])
    def test_share_gap_filling_treats_invalid_counts_as_missing(self, missing_value):
        shares = valuation.pd.Series([200.0, missing_value, 400.0])

        filled = valuation.fill_finmind_missing_ordinary_shares(shares)

        assert filled.tolist() == [200.0, 300.0, 400.0]

    def test_fetch_derives_shares_from_finmind_balance_sheet(self):
        finmind_financial_rows = [
            {"date": "2025-03-31", "type": "Revenue", "value": 100},
            {"date": "2025-03-31", "type": "EPS", "value": 1.5},
        ]
        finmind_balance_rows = [
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200}
        ]
        with (
            patch.object(
                valuation,
                "quarterly_start_date",
                return_value="2024-01-01",
            ) as start_date,
            patch.object(
                valuation,
                "fetch_finmind_taiwan_rows",
                new=AsyncMock(
                    side_effect=[finmind_financial_rows, finmind_balance_rows]
                ),
            ) as finmind_fetch,
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements("2330", limit=1)
            )

        start_date.assert_called_once_with(
            "t",
            1,
            valuation.FINMIND_TAIWAN_QUARTER_BUFFER,
            valuation.FINMIND_TAIWAN_MONTH_BUFFER,
        )
        finmind_fetch.assert_any_await(
            "TaiwanStockFinancialStatements", "2330", "2024-01-01"
        )
        finmind_fetch.assert_any_await("TaiwanStockBalanceSheet", "2330", "2024-01-01")
        assert rows == {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": 20,
                "epsDiluted": 1.5,
            }
        }

    def test_fetch_selects_latest_quarters_from_unsorted_finmind_rows(self):
        financial_rows = [
            {"date": "2025-12-31", "type": "EPS", "value": 1.3},
            {"date": "2025-03-31", "type": "Revenue", "value": 100},
            {"date": "2025-09-30", "type": "Revenue", "value": 119},
            {"date": "2025-06-30", "type": "Revenue", "value": 110},
            {"date": "2025-12-31", "type": "Revenue", "value": 130},
            {"date": "2025-09-30", "type": "EPS", "value": 1.2},
            {"date": "2025-03-31", "type": "EPS", "value": 1.0},
            {"date": "2025-06-30", "type": "EPS", "value": 1.1},
            {"date": "2025-09-30", "type": "Revenue", "value": 120},
        ]
        balance_sheet_rows = [
            {"date": "2025-09-30", "type": "OrdinaryShare", "value": 240},
            {"date": "2026-03-31", "type": "OrdinaryShare", "value": 9990},
            {"date": "2025-12-31", "type": "OrdinaryShare", "value": 250},
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
            {"date": "2025-12-31", "type": "OrdinaryShare", "value": 260},
        ]
        with patch.object(
            valuation,
            "fetch_finmind_taiwan_rows",
            new=AsyncMock(side_effect=[financial_rows, balance_sheet_rows]),
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements("2330", limit=2)
            )

        assert list(rows) == ["2025-09-30", "2025-12-31"]
        assert rows == {
            "2025-09-30": {
                "revenue": 120,
                "weightedAverageShsOutDil": 24,
                "epsDiluted": 1.2,
            },
            "2025-12-31": {
                "revenue": 130,
                "weightedAverageShsOutDil": 26,
                "epsDiluted": 1.3,
            },
        }

    def test_fetch_accepts_finmind_rows_missing_eps(self):
        finmind_financial_rows = [
            {"date": "2025-03-31", "type": "Revenue", "value": 100},
        ]
        finmind_balance_rows = [
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200}
        ]
        with (
            patch.object(
                valuation,
                "fetch_finmind_taiwan_rows",
                new=AsyncMock(
                    side_effect=[finmind_financial_rows, finmind_balance_rows]
                ),
            ),
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements("2330", limit=1)
            )

        assert rows == {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": 20,
                "epsDiluted": None,
            }
        }

    def test_fetch_keeps_latest_quarter_when_rps_is_complete(self):
        financial_rows = [
            {"date": "2025-03-31", "type": "Revenue", "value": 100},
            {"date": "2025-03-31", "type": "EPS", "value": 1.5},
            {"date": "2025-06-30", "type": "Revenue", "value": 110},
        ]
        balance_sheet_rows = [
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
            {"date": "2025-06-30", "type": "OrdinaryShare", "value": 220},
        ]
        with patch.object(
            valuation,
            "fetch_finmind_taiwan_rows",
            new=AsyncMock(side_effect=[financial_rows, balance_sheet_rows]),
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements("2330", limit=1)
            )

        assert list(rows) == ["2025-06-30"]

    def test_fetch_keeps_latest_quarter_when_eps_is_missing(self):
        financial_rows = [
            {"date": "2025-03-31", "type": "Revenue", "value": 100},
            {"date": "2025-03-31", "type": "EPS", "value": 1.5},
            {"date": "2025-06-30", "type": "Revenue", "value": 110},
        ]
        balance_sheet_rows = [
            {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
            {"date": "2025-06-30", "type": "OrdinaryShare", "value": 220},
        ]
        with patch.object(
            valuation,
            "fetch_finmind_taiwan_rows",
            new=AsyncMock(side_effect=[financial_rows, balance_sheet_rows]),
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements("2330", limit=1)
            )

        assert list(rows) == ["2025-06-30"]

    @pytest.mark.parametrize("include_eps", [True, False])
    def test_resolved_taiwan_quarters_do_not_forward_include_eps(
        self, include_eps: bool
    ):
        resolved: valuation.ResolvedQuarters = {
            "2025-06-30": {
                "revenue": 110,
                "weightedAverageShsOutDil": 22,
                "epsDiluted": None,
            }
        }
        fetch = AsyncMock(return_value=resolved)

        with patch.object(
            valuation,
            "fetch_finmind_taiwan_income_statements",
            new=fetch,
        ):
            result = asyncio.run(
                valuation.fetch_resolved_income_statement_quarters(
                    "t", "2330", 1, include_eps
                )
            )

        assert result == valuation.split_xps_quarters(resolved, include_eps)
        fetch.assert_awaited_once_with("2330", 1)

    def test_select_latest_required_quarters_does_not_validate_metrics(self):
        rows = {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": 20,
                "epsDiluted": 1.0,
            },
            "2025-06-30": {
                "revenue": 110,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.1,
            },
            "2025-09-30": {
                "revenue": 120,
                "weightedAverageShsOutDil": 24,
                "epsDiluted": 1.2,
            },
        }

        selected = valuation.select_latest_required_quarters(rows, 2)

        assert list(selected) == ["2025-06-30", "2025-09-30"]

    def test_select_latest_required_quarters_keeps_eps_only_latest_row(self):
        rows = {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.0,
            }
        }

        selected = valuation.select_latest_required_quarters(rows, 1)

        assert selected == rows
