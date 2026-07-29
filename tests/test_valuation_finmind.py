import asyncio
import unittest
from unittest.mock import AsyncMock, call, patch

from backend import valuation


class FinMindValuationTests(unittest.TestCase):
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
        self.assertEqual(rows, [{"type": "EPS"}])

    def test_normalize_income_statement_rows_canonicalizes_dates(self):
        rows = valuation.normalize_income_statement_rows(
            [
                {"date": "2025-09-30T00:00:00.000Z", "revenue": 100},
                {"date": "not-a-date", "revenue": 200},
            ],
            field_map={"revenue": "revenue"},
            date_field="date",
        )

        self.assertEqual(rows, {"2025-09-30": {"revenue": 100}})

    def test_quarterly_start_date_uses_market_timezone_and_buffer(self):
        now = valuation.pd.Timestamp("2026-07-22 15:30:00", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now) as now_mock:
            self.assertEqual(
                valuation.quarterly_start_date("u", limit=4, quarter_buffer=1),
                "2025-04-22",
            )
            self.assertEqual(
                valuation.quarterly_start_date("t", limit=4, quarter_buffer=2),
                "2025-01-22",
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

        self.assertEqual(frame.columns.tolist(), ["rps", "eps"])
        self.assertEqual(
            frame.index.strftime("%Y-%m-%d").tolist(),
            ["2026-01-01", "2026-04-01"],
        )
        self.assertEqual(frame["rps"].tolist(), [41.0, 41.0])
        self.assertEqual(frame["eps"].tolist(), [4.6, 5.0])

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
        self.assertEqual(without_eps.columns.tolist(), ["rps"])
        self.assertEqual(
            without_eps.index.strftime("%Y-%m-%d").tolist(),
            ["2026-01-01", "2026-04-01"],
        )
        self.assertEqual(without_eps["rps"].tolist(), [41.0, 41.0])

    def test_income_statement_rows_are_kept_without_balance_sheet_data(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {"date": "2025-03-31", "type": "Revenue", "value": 100},
                {"date": "2025-03-31", "type": "EPS", "value": 1.5},
            ],
            [],
        )

        self.assertEqual(
            rows,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": None,
                    "epsDiluted": 1.5,
                }
            },
        )

    def test_normalize_skips_rows_without_revenue(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [{"date": "2025-03-31", "type": "EPS", "value": 1.5}],
            [],
        )

        self.assertEqual(rows, {})

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

        self.assertEqual(
            rows,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 20,
                    "epsDiluted": 1.5,
                }
            },
        )

    def test_normalize_interpolates_one_missing_balance_sheet_quarter(self):
        with patch.object(valuation.logger, "warning") as warning:
            rows = valuation.normalize_finmind_taiwan_income_statement_rows(
                [
                    {"date": "2025-03-31", "type": "Revenue", "value": 100},
                    {"date": "2025-06-30", "type": "Revenue", "value": 110},
                    {"date": "2025-09-30", "type": "Revenue", "value": 120},
                ],
                [
                    {"date": "2025-03-31", "type": "OrdinaryShare", "value": 200},
                    {"date": "2025-09-30", "type": "OrdinaryShare", "value": 400},
                ],
            )

        self.assertEqual(
            rows["2025-06-30"]["weightedAverageShsOutDil"],
            30,
        )
        warning.assert_called_once()
        self.assertIn("!!! PATCHING FINMIND", warning.call_args.args[0])
        self.assertEqual(
            warning.call_args.args[1:3], ("2025-06-30", "linear interpolation")
        )

    def test_normalize_fills_endpoint_balance_sheet_gaps_with_latest_value(self):
        with patch.object(valuation.logger, "warning") as warning:
            rows = valuation.normalize_finmind_taiwan_income_statement_rows(
                [
                    {"date": "2025-03-31", "type": "Revenue", "value": 100},
                    {"date": "2025-06-30", "type": "Revenue", "value": 110},
                    {"date": "2025-09-30", "type": "Revenue", "value": 120},
                ],
                [
                    {"date": "2025-06-30", "type": "OrdinaryShare", "value": 200},
                ],
            )

        self.assertEqual(
            [row["weightedAverageShsOutDil"] for row in rows.values()],
            [20, 20, 20],
        )
        warning.assert_has_calls(
            [
                call(
                    "!!! PATCHING FINMIND OrdinaryShare gap: "
                    "date={} method={} patched_value={} !!!",
                    "2025-03-31",
                    "latest reported value",
                    200.0,
                ),
                call(
                    "!!! PATCHING FINMIND OrdinaryShare gap: "
                    "date={} method={} patched_value={} !!!",
                    "2025-09-30",
                    "latest reported value",
                    200.0,
                ),
            ]
        )
        self.assertEqual(warning.call_count, 2)

    def test_normalize_rejects_multiple_missing_balance_sheet_quarters(self):
        with patch.object(valuation.logger, "warning") as warning:
            with self.assertRaisesRegex(
                ValueError,
                r"2 consecutive missing OrdinaryShare values",
            ):
                valuation.normalize_finmind_taiwan_income_statement_rows(
                    [
                        {"date": "2025-03-31", "type": "Revenue", "value": 100},
                        {"date": "2025-06-30", "type": "Revenue", "value": 110},
                        {"date": "2025-09-30", "type": "Revenue", "value": 120},
                        {"date": "2025-12-31", "type": "Revenue", "value": 130},
                    ],
                    [
                        {
                            "date": "2025-03-31",
                            "type": "OrdinaryShare",
                            "value": 200,
                        },
                        {
                            "date": "2025-12-31",
                            "type": "OrdinaryShare",
                            "value": 400,
                        },
                    ],
                )

        warning.assert_not_called()

    def test_normalize_rejects_consecutive_endpoint_balance_sheet_gaps(self):
        financial_rows = [
            {"date": date, "type": "Revenue", "value": value}
            for date, value in [
                ("2025-03-31", 100),
                ("2025-06-30", 110),
                ("2025-09-30", 120),
                ("2025-12-31", 130),
            ]
        ]
        for reported_dates in (
            ("2025-09-30", "2025-12-31"),
            ("2025-03-31", "2025-06-30"),
        ):
            with self.subTest(reported_dates=reported_dates):
                balance_sheet_rows = [
                    {"date": date, "type": "OrdinaryShare", "value": 200}
                    for date in reported_dates
                ]
                with self.assertRaisesRegex(
                    ValueError,
                    r"2 consecutive missing OrdinaryShare values",
                ):
                    valuation.normalize_finmind_taiwan_income_statement_rows(
                        financial_rows, balance_sheet_rows
                    )

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
            "t", 1, valuation.FINMIND_TAIWAN_QUARTER_BUFFER
        )
        finmind_fetch.assert_any_await(
            "TaiwanStockFinancialStatements", "2330", "2024-01-01"
        )
        finmind_fetch.assert_any_await("TaiwanStockBalanceSheet", "2330", "2024-01-01")
        self.assertEqual(
            rows,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 20,
                    "epsDiluted": 1.5,
                }
            },
        )

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

        self.assertEqual(
            rows,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 20,
                    "epsDiluted": None,
                }
            },
        )

    def test_fetch_drops_latest_quarter_missing_required_eps(self):
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

        self.assertEqual(list(rows), ["2025-03-31"])

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

        self.assertEqual(list(rows), ["2025-06-30"])

    def test_select_latest_clean_required_quarters_rejects_older_incomplete_rows(self):
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

        with self.assertRaisesRegex(
            ValueError, r"incomplete quarters: \['2025-06-30'\]"
        ):
            valuation.select_latest_clean_required_quarters(
                rows, 2, valuation.required_xps_fields(True)
            )

    def test_select_latest_clean_required_quarters_fails_when_latest_drop_leaves_too_few(
        self,
    ):
        rows = {
            "2025-03-31": {
                "revenue": 100,
                "weightedAverageShsOutDil": None,
                "epsDiluted": 1.0,
            }
        }

        with self.assertRaisesRegex(ValueError, r"need 1 quarters, found 0"):
            valuation.select_latest_clean_required_quarters(
                rows, 1, valuation.required_xps_fields(True)
            )
