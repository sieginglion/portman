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

        result = valuation.complete_and_roll_xps(xps, 4)

        assert result.columns.tolist() == ["rps"]
        assert result["rps"].tolist() == [10.0]

    def test_complete_and_roll_xps_rejects_when_neither_column_is_complete(self):
        xps = valuation.pd.DataFrame({"rps": [1.0, None], "eps": [None, 1.0]})

        with pytest.raises(ValueError, match="neither a complete"):
            valuation.complete_and_roll_xps(xps, 2)

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
            headers={"Authorization": f"Bearer {valuation.FINMIND_KEY}"},
        )
        assert rows == [{"type": "EPS"}]

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

    def test_build_xps_frame_returns_ttm_values_with_quarter_dates(self):
        quarters = {
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

        frame = valuation.build_xps_frame(quarters, include_eps=True)

        assert frame.columns.tolist() == ["rps", "eps"]
        assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-01-01", "2026-04-01"]
        assert frame["rps"].tolist() == [41.0, 41.0]
        assert frame["eps"].tolist() == [4.6, 5.0]

        without_eps = valuation.build_xps_frame(
            {
                date: {
                    "revenue": row["revenue"],
                    "weightedAverageShsOutDil": row["weightedAverageShsOutDil"],
                }
                for date, row in quarters.items()
            },
            include_eps=False,
        )
        assert without_eps.columns.tolist() == ["rps"]
        assert without_eps.index.strftime("%Y-%m-%d").tolist() == [
            "2026-01-01",
            "2026-04-01",
        ]
        assert without_eps["rps"].tolist() == [41.0, 41.0]

    def test_build_xps_frame_keeps_each_complete_metric_independently(self):
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

        frame = valuation.build_xps_frame(quarters, include_eps=True)

        assert frame.columns.tolist() == ["eps"]
        assert frame["eps"].tolist() == [4.6]

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
            patch.object(valuation.shared, "get_prices", return_value=prices),
            patch.object(valuation, "fetch_xps", return_value=xps),
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
        assert px.attrs["xps_dates"].equals(xps.index)

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
            patch.object(valuation.shared, "get_prices", return_value=prices),
            patch.object(valuation, "fetch_xps", return_value=xps),
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
            patch.object(valuation.shared, "get_prices", return_value=prices),
            patch.object(valuation, "fetch_xps", return_value=xps),
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
            patch.object(valuation.shared, "get_prices", return_value=[None, 21.0]),
            patch.object(valuation, "fetch_xps", return_value=xps),
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

    def test_normalize_leaves_missing_shares_for_central_validation(self):
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
                valuation.fetch_finmind_taiwan_income_statements(
                    "2330", limit=1, require_eps=True
                )
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

    def test_fetch_without_eps_accepts_finmind_rows_missing_eps(self):
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
                valuation.fetch_finmind_taiwan_income_statements(
                    "2330", limit=1, require_eps=False
                )
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
                valuation.fetch_finmind_taiwan_income_statements(
                    "2330", limit=1, require_eps=True
                )
            )

        assert list(rows) == ["2025-06-30"]

    def test_fetch_keeps_latest_quarter_when_eps_is_not_required(self):
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
                valuation.fetch_finmind_taiwan_income_statements(
                    "2330", limit=1, require_eps=False
                )
            )

        assert list(rows) == ["2025-06-30"]

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
