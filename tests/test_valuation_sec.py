import asyncio
import math
import unittest
from unittest.mock import ANY, AsyncMock, patch

import pandas as pd
from httpx import HTTPStatusError, Request, Response

from backend import valuation


def sec_frame(rows):
    return pd.DataFrame(rows, columns=valuation.SEC_DEDUPE_COLS)


def selected_sources(*names: str):
    source_by_name = {
        source.name: source for source in valuation.US_INCOME_STATEMENT_SOURCES
    }
    return tuple(source_by_name[name] for name in names)


def patch_xps_diagnostics():
    return patch.object(valuation, "_xps_diagnostics", valuation.new_xps_diagnostics())


def aligned_quarter(fields):
    return valuation.AlignedQuarter(fields=fields)


class DateMatchingTests(unittest.TestCase):
    def test_select_closest_date_key_uses_six_day_limit_and_stable_tiebreaking(self):
        cases = (
            ("2025-03-31", ["2025-03-31"], "2025-03-31"),
            ("2025-04-06", ["2025-03-31"], "2025-03-31"),
            ("2025-04-07", ["2025-03-31"], None),
            ("2025-04-04", ["2025-03-31", "2025-04-05"], "2025-04-05"),
            ("2025-04-03", ["2025-04-05", "2025-04-01"], "2025-04-01"),
        )

        for target_date, date_keys, expected in cases:
            with self.subTest(target_date=target_date, date_keys=date_keys):
                self.assertEqual(
                    valuation.select_closest_date_key(target_date, date_keys),
                    expected,
                )

    def test_eodhd_balance_sheet_lookup_uses_nearby_dates_only(self):
        row = {"commonStockSharesOutstanding": "123"}

        self.assertIs(
            valuation.select_eodhd_balance_sheet_row("2025-03-31", {"2025-04-06": row}),
            row,
        )
        self.assertIsNone(
            valuation.select_eodhd_balance_sheet_row("2025-03-31", {"2025-04-07": row})
        )


class SecValuationTests(unittest.TestCase):
    def test_source_diffs_reject_nonfinite_values(self):
        self.assertEqual(valuation.source_log_diff(math.inf, 1), math.inf)
        self.assertEqual(valuation.source_log_diff(1, math.nan), math.inf)
        self.assertEqual(valuation.source_abs_diff(math.inf, 1), math.inf)
        self.assertEqual(valuation.source_abs_diff(1, math.nan), math.inf)

    def test_consensus_helpers_resolve_agreeing_sources(self):
        source_values = {"fmp": 100, "finnhub": 102, "sec": None}

        values = valuation.usable_source_values(source_values)
        groups = valuation.build_consensus_groups("revenue", values)
        group = valuation.choose_consensus_group(groups)

        self.assertEqual(values, [("fmp", 100), ("finnhub", 102)])
        self.assertEqual(group, (("fmp", 100), ("finnhub", 102)))
        self.assertEqual(
            valuation.resolve_source_consensus("revenue", source_values),
            valuation.Consensus(101, ("fmp", "finnhub")),
        )

    def test_configured_source_order_drives_alignment_and_consensus(self):
        source_rows = {
            "fmp": {"2025-03-31": {"revenue": 100}},
            "finnhub": {"2025-04-05": {"revenue": 102}},
        }
        with patch.object(
            valuation,
            "ENABLED_US_INCOME_STATEMENT_SOURCES",
            selected_sources("finnhub", "fmp"),
        ):
            aligned_quarters = valuation.build_aligned_source_quarters(source_rows)
            consensus = valuation.resolve_source_consensus(
                "revenue", {"fmp": 100, "finnhub": 102}
            )

        self.assertEqual(list(aligned_quarters), ["2025-04-05"])
        self.assertEqual(
            aligned_quarters["2025-04-05"].fields,
            {"revenue": {"finnhub": 102, "fmp": 100}},
        )
        self.assertEqual(consensus, valuation.Consensus(101, ("finnhub", "fmp")))

    def test_consensus_helpers_reject_disagreeing_sources(self):
        source_values = {"fmp": 100, "finnhub": 120}

        values = valuation.usable_source_values(source_values)

        self.assertEqual(valuation.build_consensus_groups("revenue", values), [])
        self.assertIsNone(valuation.choose_consensus_group([]))
        self.assertIsNone(valuation.resolve_source_consensus("revenue", source_values))

    def test_consensus_helpers_keep_transitive_consensus_groups(self):
        source_values = {"fmp": 100, "finnhub": 106, "sec": 112}

        self.assertEqual(
            valuation.resolve_source_consensus("revenue", source_values),
            valuation.Consensus(106, ("fmp", "finnhub", "sec")),
        )

    def test_choose_consensus_group_prefers_sec_in_a_tie(self):
        groups = [
            (("fmp", 100), ("finnhub", 100)),
            (("massive", 200), ("sec", 200)),
        ]

        self.assertEqual(
            valuation.choose_consensus_group(groups),
            (("massive", 200), ("sec", 200)),
        )

    def test_choose_consensus_group_rejects_tie_without_sec(self):
        groups = [
            (("fmp", 100), ("finnhub", 100)),
            (("massive", 200), ("eodhd", 200)),
        ]

        self.assertIsNone(valuation.choose_consensus_group(groups))

    def test_consensus_helpers_apply_eps_absolute_tolerance(self):
        source_values = {"fmp": 0.001, "finnhub": -0.01}

        consensus = valuation.resolve_source_consensus("epsDiluted", source_values)

        self.assertIsNotNone(consensus)
        self.assertAlmostEqual(consensus.value, -0.0045)
        self.assertEqual(consensus.sources, ("fmp", "finnhub"))

    def test_diagnostics_reports_per_vendor_missing_counts(self):
        def row(fmp, finnhub):
            return aligned_quarter(
                {
                    field: {"fmp": fmp, "finnhub": finnhub}
                    for field in valuation.ALL_XPS_FIELDS
                }
            )

        rows = {
            "2025-03-31": row(100, 100),
            "2025-06-30": row(100, None),
            "2025-09-30": row(None, 100),
            "2025-12-31": row(None, None),
        }
        with patch_xps_diagnostics():
            valuation.record_xps_diagnostics("MISSINGNESS-TEST", rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics["total_quarters"], 4)
        revenue = diagnostics["missing"]["revenue"]
        self.assertNotIn("sec", revenue)
        self.assertEqual(revenue["fmp"], 2)
        self.assertEqual(revenue["finnhub"], 2)
        self.assertEqual(diagnostics["missing"]["weightedAverageShsOutDil"], revenue)
        self.assertEqual(diagnostics["missing"]["epsDiluted"], revenue)
        self.assertEqual(diagnostics["consensus_pairs"]["revenue"]["fmp:finnhub"], 1)
        self.assertEqual(
            diagnostics["consensus_pairs"]["weightedAverageShsOutDil"]["fmp:finnhub"],
            1,
        )
        self.assertEqual(diagnostics["consensus_pairs"]["epsDiluted"]["fmp:finnhub"], 1)

    def test_missing_counts_exclude_unavailable_sources(self):
        with patch_xps_diagnostics():
            valuation.record_xps_diagnostics(
                "UNAVAILABLE-SOURCE-TEST",
                {
                    "2025-03-31": aligned_quarter(
                        {"revenue": {"fmp": None, "finnhub": None}}
                    )
                },
                unavailable_sources=frozenset({"fmp"}),
            )
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics["total_quarters"], 1)
        revenue = diagnostics["missing"]["revenue"]
        self.assertEqual(revenue["fmp"], 0)
        self.assertEqual(revenue["finnhub"], 1)
        self.assertTrue(
            all(
                count == 0
                for count in diagnostics["consensus_pairs"]["revenue"].values()
            )
        )

    def test_diagnostics_counts_each_directly_agreeing_pair_once(self):
        rows = {
            "2025-03-31": aligned_quarter(
                {"revenue": {"fmp": 100, "massive": 104, "finnhub": 108}}
            )
        }
        with (
            patch.object(
                valuation,
                "ENABLED_US_INCOME_STATEMENT_SOURCES",
                selected_sources("fmp", "massive", "finnhub"),
            ),
            patch_xps_diagnostics(),
        ):
            valuation.record_xps_diagnostics("PAIR-TEST", rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(
            diagnostics["consensus_pairs"]["revenue"],
            {
                "fmp:massive": 1,
                "fmp:finnhub": 0,
                "massive:finnhub": 1,
            },
        )
        self.assertNotIn("sec", diagnostics["consensus_pairs"]["revenue"])

    def test_diagnostics_excludes_present_sec_values_from_peer_pairs(self):
        rows = {
            "2025-03-31": aligned_quarter(
                {"revenue": {"fmp": 100, "finnhub": 100, "sec": 100}}
            )
        }
        with (
            patch.object(
                valuation,
                "ENABLED_US_INCOME_STATEMENT_SOURCES",
                selected_sources("fmp", "finnhub"),
            ),
            patch_xps_diagnostics(),
        ):
            valuation.record_xps_diagnostics("SEC-EXCLUSION-TEST", rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics["consensus_pairs"]["revenue"], {"fmp:finnhub": 1})

    def test_diagnostics_deduplicates_missing_and_pair_counts(self):
        rows = {
            "2025-03-31": aligned_quarter({"revenue": {"fmp": 100, "finnhub": 100}})
        }
        with (
            patch.object(
                valuation,
                "ENABLED_US_INCOME_STATEMENT_SOURCES",
                selected_sources("fmp", "finnhub"),
            ),
            patch_xps_diagnostics(),
        ):
            valuation.record_xps_diagnostics("DEDUP-TEST", rows)
            valuation.record_xps_diagnostics("DEDUP-TEST", rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics["total_quarters"], 1)
        self.assertEqual(diagnostics["missing"]["revenue"], {"fmp": 0, "finnhub": 0})
        self.assertEqual(diagnostics["consensus_pairs"]["revenue"], {"fmp:finnhub": 1})

    def test_diagnostics_skips_request_that_falls_short_after_latest_drop(self):
        def row(revenue):
            return {
                "revenue": revenue,
                "weightedAverageShsOutDil": 10,
                "epsDiluted": revenue / 100,
            }

        source_rows = {
            "fmp": {
                "2025-03-31": row(100),
                "2025-06-30": row(110),
                "2025-09-30": row(120),
                "2025-12-31": row(130),
            },
            "finnhub": {
                "2025-03-31": row(100),
                "2025-06-30": row(110),
                "2025-09-30": row(120),
            },
        }
        diagnostics_state = valuation.new_xps_diagnostics()
        with (
            patch.object(valuation, "_xps_diagnostics", diagnostics_state),
            patch.object(valuation, "add_sec_reference_values", new=AsyncMock()),
            self.assertRaises(ValueError),
        ):
            asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "SPCX", source_rows, limit=4, include_eps=True
                )
            )

        self.assertEqual(diagnostics_state.seen, set())
        self.assertEqual(diagnostics_state.missing["revenue"]["finnhub"], 0)

    def test_resolve_us_income_statement_quarters_accepts_source_rows(self):
        source_rows = {
            # Insertion order intentionally differs from the source registry.
            "finnhub": {
                "2025-04-05": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                }
            },
            "fmp": {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                }
            },
        }

        with patch.object(valuation, "add_sec_reference_values", new=AsyncMock()):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "AAPL", source_rows, 1, include_eps=True
                )
            )

        self.assertEqual(
            resolved,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_without_eps(self):
        source_rows = {
            "fmp": {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                }
            },
            "finnhub": {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                }
            },
        }

        with (
            patch.object(
                valuation, "add_sec_reference_values", new=AsyncMock()
            ) as add_sec_values,
            patch.object(valuation, "record_xps_diagnostics") as diagnostics,
        ):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "AAPL", source_rows, limit=1, include_eps=False
                )
            )

        add_sec_values.assert_awaited_once_with(
            "AAPL",
            ANY,
            ["revenue", "weightedAverageShsOutDil"],
        )
        diagnostics.assert_not_called()
        self.assertEqual(
            resolved,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_drops_latest_single_source_row(
        self,
    ):
        source_rows = {
            "fmp": {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                },
                "2025-06-30": {
                    "revenue": 120,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1.2,
                },
            },
            "finnhub": {
                "2025-04-05": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                }
            },
        }

        with patch.object(
            valuation, "add_sec_reference_values", new=AsyncMock()
        ) as add_sec_values:
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "AAPL", source_rows, limit=1, include_eps=True
                )
            )

        add_sec_values.assert_awaited_once()
        self.assertEqual(
            resolved,
            {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_merges_sec_before_dropping_latest(
        self,
    ):
        source_rows = {
            "fmp": {
                "2025-06-30": {
                    "revenue": 120,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1.2,
                }
            },
            "finnhub": {"2025-06-30": {"revenue": 121}},
        }

        async def add_sec_fields(symbol, aligned_quarters, required_fields):
            self.assertEqual(symbol, "AAPL")
            self.assertEqual(
                required_fields,
                ["revenue", "weightedAverageShsOutDil", "epsDiluted"],
            )
            latest_quarter = aligned_quarters["2025-06-30"]
            latest_quarter.fields["weightedAverageShsOutDil"]["sec"] = 10
            latest_quarter.fields["epsDiluted"]["sec"] = 1.2

        with patch.object(
            valuation,
            "add_sec_reference_values",
            new=AsyncMock(side_effect=add_sec_fields),
        ) as add_sec_values:
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "AAPL", source_rows, limit=1, include_eps=True
                )
            )

        add_sec_values.assert_awaited_once()
        self.assertEqual(
            resolved,
            {
                "2025-06-30": {
                    "revenue": 120.5,
                    "weightedAverageShsOutDil": 10,
                    "epsDiluted": 1.2,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_records_selected_quarters_only(
        self,
    ):
        def row(revenue):
            return {
                "revenue": revenue,
                "weightedAverageShsOutDil": 10,
                "epsDiluted": revenue / 100,
            }

        source_rows = {
            "fmp": {
                "2025-03-31": row(100),
                "2025-06-30": row(110),
            },
            "finnhub": {
                "2025-03-31": row(100),
                "2025-06-30": row(110),
            },
        }
        diagnostics_state = valuation.new_xps_diagnostics()
        with (
            patch.object(valuation, "_xps_diagnostics", diagnostics_state),
            patch.object(valuation, "add_sec_reference_values", new=AsyncMock()),
        ):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    "AAPL", source_rows, limit=1, include_eps=True
                )
            )

        self.assertEqual(list(resolved), ["2025-06-30"])
        self.assertEqual(diagnostics_state.seen, {("AAPL", "2025-06-30")})

    def test_select_latest_required_quarters_reports_symbol_and_available_dates(self):
        with self.assertRaisesRegex(
            ValueError,
            (
                r"need 5 aligned quarters, found 2 for SPCX; "
                r"available=\['2025-03-31', '2026-03-31'\]"
            ),
        ):
            valuation.select_latest_required_quarters(
                {
                    "2026-03-31": {},
                    "2025-03-31": {},
                },
                5,
                symbol="SPCX",
                quarter_label="aligned quarters",
            )

    def test_format_sec_cik_normalizes_values(self):
        self.assertEqual(valuation.format_sec_cik("320193"), "0000320193")
        self.assertEqual(valuation.format_sec_cik(320193.0), "0000320193")
        self.assertEqual(valuation.format_sec_cik("CIK 0000320193"), "0000320193")
        self.assertIsNone(valuation.format_sec_cik(None))
        self.assertIsNone(valuation.format_sec_cik("not-a-cik"))

    def test_normalize_massive_fiscal_quarter(self):
        self.assertEqual(valuation.normalize_massive_fiscal_quarter(1), "Q1")
        self.assertEqual(valuation.normalize_massive_fiscal_quarter("q4"), "Q4")
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter(5))
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter("FY"))
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter(None))

    def test_dedupe_sec_rows_keeps_latest_supported_filing_per_period(self):
        rows = [
            {
                "form": "10-Q",
                "filed": "2025-04-20",
                "val": 10,
                "start": "2025-01-01",
                "end": "2025-03-31",
            },
            {
                "form": "10-Q/A",
                "filed": "2025-05-01",
                "val": 11,
                "start": "2025-01-01",
                "end": "2025-03-31",
            },
            {
                "form": "8-K",
                "filed": "2025-05-02",
                "val": 99,
                "start": "2025-01-01",
                "end": "2025-03-31",
            },
            {
                "form": "10-Q",
                "filed": "2025-05-03",
                "val": 12,
                "start": None,
                "end": "2025-03-31",
            },
        ]

        df = valuation.dedupe_sec_rows(rows)

        self.assertEqual(
            df.to_dict("records"),
            [
                {
                    "filed": "2025-05-01",
                    "val": 11,
                    "start": "2025-01-01",
                    "end": "2025-03-31",
                }
            ],
        )

    def test_dedupe_sec_rows_returns_expected_empty_shape(self):
        self.assertEqual(
            valuation.dedupe_sec_rows([{"form": "8-K"}]).columns.tolist(),
            valuation.SEC_DEDUPE_COLS,
        )
        self.assertTrue(valuation.dedupe_sec_rows([{"form": "8-K"}]).empty)

    def test_sec_fact_window_builders_use_expected_date_ranges(self):
        cases = [
            (
                "quarter",
                valuation.sec_quarter_fact_window,
                pd.Timestamp("2025-03-31"),
                valuation.SecFactWindow(
                    min_start="2024-11-30",
                    max_start="2025-01-31",
                    min_end="2025-02-28",
                    max_end="2025-04-30",
                ),
            ),
            (
                "annual",
                valuation.sec_annual_fact_window,
                pd.Timestamp("2025-12-31"),
                valuation.SecFactWindow(
                    min_start="2024-11-30",
                    max_start="2025-01-31",
                    min_end="2025-11-30",
                    max_end="2026-01-31",
                ),
            ),
            (
                "Q1-Q3",
                valuation.sec_q1_to_q3_fact_window,
                pd.Timestamp("2025-01-01"),
                valuation.SecFactWindow(
                    min_start="2024-12-01",
                    max_start="2025-02-01",
                    min_end="2025-09-01",
                    max_end="2025-11-01",
                ),
            ),
        ]

        for label, build_window, date, expected in cases:
            with self.subTest(window=label):
                self.assertEqual(build_window(date), expected)

    def test_select_sec_fact_requires_exactly_one_date_window_match(self):
        rows = sec_frame(
            [
                {
                    "filed": "2025-04-20",
                    "val": 10,
                    "start": "2025-01-01",
                    "end": "2025-03-31",
                }
            ]
        )

        match = valuation.select_sec_fact(
            rows,
            "single match",
            valuation.SecFactWindow(
                min_start="2024-12-31",
                max_start="2025-02-01",
                min_end="2025-03-01",
                max_end="2025-04-30",
            ),
        )

        self.assertEqual(match["val"], 10)
        self.assertIsNone(
            valuation.select_sec_fact(
                rows,
                "strict boundary",
                valuation.SecFactWindow(
                    min_start="2025-01-01",
                    max_start="2025-02-01",
                    min_end="2025-03-01",
                    max_end="2025-04-30",
                ),
                log_errors=False,
            )
        )

    def test_select_sec_fact_rejects_ambiguous_matches(self):
        rows = sec_frame(
            [
                {
                    "filed": "2025-04-20",
                    "val": 10,
                    "start": "2025-01-01",
                    "end": "2025-03-31",
                },
                {
                    "filed": "2025-04-21",
                    "val": 11,
                    "start": "2025-01-15",
                    "end": "2025-03-31",
                },
            ]
        )

        self.assertIsNone(
            valuation.select_sec_fact(
                rows,
                "ambiguous",
                valuation.SecFactWindow(
                    min_start="2024-12-31",
                    max_start="2025-02-01",
                    min_end="2025-03-01",
                    max_end="2025-04-30",
                ),
                log_errors=False,
            )
        )

    def test_select_sec_quarter_fact_uses_quarter_date_window(self):
        rows = sec_frame(
            [
                {
                    "filed": "2025-04-20",
                    "val": 10,
                    "start": "2025-01-01",
                    "end": "2025-03-31",
                }
            ]
        )

        match = valuation.select_sec_quarter_fact(
            rows, "quarter", pd.Timestamp("2025-03-31")
        )

        self.assertEqual(match["val"], 10)

    def test_derive_sec_q4_value_uses_annual_less_q1_to_q3(self):
        rows = sec_frame(
            [
                {
                    "filed": "2026-02-01",
                    "val": 100,
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
                {
                    "filed": "2025-11-01",
                    "val": 60,
                    "start": "2025-01-01",
                    "end": "2025-09-30",
                },
            ]
        )

        self.assertEqual(
            valuation.derive_sec_q4_value(
                "flow", rows, "q4 EPS", pd.Timestamp("2025-12-31")
            ),
            40,
        )

    def test_derive_sec_q4_value_handles_average_share_count(self):
        rows = sec_frame(
            [
                {
                    "filed": "2026-02-01",
                    "val": 25,
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
                {
                    "filed": "2025-11-01",
                    "val": 20,
                    "start": "2025-01-01",
                    "end": "2025-09-30",
                },
            ]
        )

        self.assertEqual(
            valuation.derive_sec_q4_value(
                "average",
                rows,
                "q4 shares",
                pd.Timestamp("2025-12-31"),
            ),
            40,
        )

    def test_lookup_sec_field_value_prefers_exact_q4_fact_over_derived_value(self):
        rows = sec_frame(
            [
                {
                    "filed": "2026-01-20",
                    "val": 25,
                    "start": "2025-10-01",
                    "end": "2025-12-31",
                },
                {
                    "filed": "2026-02-01",
                    "val": 100,
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
                {
                    "filed": "2025-11-01",
                    "val": 60,
                    "start": "2025-01-01",
                    "end": "2025-09-30",
                },
            ]
        )
        metadata = valuation.SecQuarterMetadata("2025-12-31", "Q4")

        self.assertEqual(
            valuation.lookup_sec_field_value(
                "revenue", "0000320193", metadata, rows, q4_value_kind="flow"
            ),
            25,
        )

    def test_lookup_sec_field_value_derives_q4_when_exact_fact_missing(self):
        rows = sec_frame(
            [
                {
                    "filed": "2026-02-01",
                    "val": 100,
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
                {
                    "filed": "2025-11-01",
                    "val": 60,
                    "start": "2025-01-01",
                    "end": "2025-09-30",
                },
            ]
        )
        metadata = valuation.SecQuarterMetadata("2025-12-31", "Q4")

        self.assertEqual(
            valuation.lookup_sec_field_value(
                "revenue", "0000320193", metadata, rows, q4_value_kind="flow"
            ),
            40,
        )

    def test_select_sec_cik_prefers_fmp_and_falls_back_to_massive(self):
        fmp_rows = {"2025-03-31": {"cik": "320193"}}
        massive_rows = {"2025-03-31": {"cik": "789019"}}
        with patch.object(
            valuation,
            "ENABLED_US_INCOME_STATEMENT_SOURCES",
            selected_sources("fmp", "massive"),
        ):
            self.assertEqual(
                valuation.select_sec_cik(
                    valuation.build_aligned_source_quarters(
                        {"fmp": fmp_rows, "massive": massive_rows}
                    )
                ),
                "0000320193",
            )
            self.assertEqual(
                valuation.select_sec_cik(
                    valuation.build_aligned_source_quarters(
                        {
                            "fmp": {"2025-03-31": {}},
                            "massive": massive_rows,
                        }
                    )
                ),
                "0000789019",
            )
            self.assertIsNone(
                valuation.select_sec_cik(
                    valuation.build_aligned_source_quarters(
                        {
                            "fmp": {"2025-03-31": {}},
                            "massive": {"2025-03-31": {}},
                        }
                    )
                )
            )

    def test_alignment_retains_provider_date_for_sec_metadata(self):
        source_rows = {
            "finnhub": {"2025-06-30": {"revenue": 100}},
            "fmp": {
                "2025-06-29": {
                    "revenue": 100,
                    "cik": "320193",
                    "quarter": "Q2",
                }
            },
        }

        with patch.object(
            valuation,
            "ENABLED_US_INCOME_STATEMENT_SOURCES",
            selected_sources("finnhub", "fmp"),
        ):
            aligned_quarters = valuation.build_aligned_source_quarters(source_rows)

        self.assertEqual(list(aligned_quarters), ["2025-06-30"])
        self.assertEqual(
            valuation.build_sec_quarter_metadata(aligned_quarters),
            {"2025-06-30": valuation.SecQuarterMetadata("2025-06-29", "Q2")},
        )

    def test_sec_cik_uses_later_source_row_while_metadata_uses_first(self):
        source_rows = {
            "fmp": {
                "2025-03-31": {"quarter": "Q1"},
                "2025-04-05": {"cik": "320193", "quarter": "Q1"},
            }
        }
        aligned_quarters = valuation.build_aligned_source_quarters(source_rows)

        self.assertEqual(valuation.select_sec_cik(aligned_quarters), "0000320193")
        self.assertEqual(
            valuation.build_sec_quarter_metadata(aligned_quarters),
            {"2025-03-31": valuation.SecQuarterMetadata("2025-03-31", "Q1")},
        )

    def test_build_sec_quarter_metadata_uses_fmp_then_massive_priority(self):
        aligned_quarters = {
            "2025-03-31": valuation.AlignedQuarter(),
            "2025-06-30": valuation.AlignedQuarter(),
            "2025-09-30": valuation.AlignedQuarter(),
        }
        aligned_quarters["2025-03-31"].source_periods["fmp"] = [
            ("2025-03-31", {"quarter": "Q1"}),
        ]
        aligned_quarters["2025-06-30"].source_periods.update(
            {
                "fmp": [("2025-06-29", {"quarter": "Q2"})],
                "massive": [("2025-06-30", {"quarter": 3})],
            }
        )
        aligned_quarters["2025-09-30"].source_periods["massive"] = [
            ("2025-09-30", {"quarter": 3}),
        ]

        metadata = valuation.build_sec_quarter_metadata(aligned_quarters)

        self.assertEqual(
            metadata,
            {
                "2025-03-31": valuation.SecQuarterMetadata("2025-03-31", "Q1"),
                "2025-06-30": valuation.SecQuarterMetadata("2025-06-29", "Q2"),
                "2025-09-30": valuation.SecQuarterMetadata("2025-09-30", "Q3"),
            },
        )


class FetchSecFieldRowsTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_empty_concept_frame(self):
        empty_rows = sec_frame([])

        with patch.object(
            valuation,
            "fetch_sec_concept_rows",
            return_value=empty_rows,
        ):
            result = await valuation.fetch_sec_field_rows(
                "0000320193", "weightedAverageShsOutDil"
            )

        self.assertIs(result, empty_rows)
        self.assertTrue(result.empty)

    async def test_returns_empty_dataframe_when_concept_processing_fails(self):
        with patch.object(
            valuation,
            "fetch_sec_concept_rows",
            side_effect=RuntimeError("bad SEC response"),
        ):
            result = await valuation.fetch_sec_field_rows(
                "0000320193", "weightedAverageShsOutDil"
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    async def test_returns_empty_dataframe_when_concept_is_not_found(self):
        request = Request("GET", "https://data.sec.gov/api/xbrl/companyconcept")
        error = HTTPStatusError(
            "Not found", request=request, response=Response(404, request=request)
        )

        with patch.object(
            valuation,
            "fetch_sec_company_concept_raw",
            side_effect=error,
        ):
            result = await valuation.fetch_sec_field_rows(
                "0000320193", "weightedAverageShsOutDil"
            )

        self.assertTrue(result.empty)


class FetchSecValuesForQuartersTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_required_repair_fields_and_returns_only_usable_values(self):
        metadata_by_quarter = {
            "2025-03-31": valuation.SecQuarterMetadata("2025-03-31", "Q1"),
            "2025-06-30": valuation.SecQuarterMetadata("2025-06-30", "Q2"),
        }
        frames_by_field = {
            "weightedAverageShsOutDil": sec_frame([{}]),
            "epsDiluted": sec_frame([{}]),
        }
        values_by_field_and_date = {
            ("weightedAverageShsOutDil", "2025-03-31"): 12,
            ("weightedAverageShsOutDil", "2025-06-30"): 0,
            ("epsDiluted", "2025-03-31"): 1.2,
            ("epsDiluted", "2025-06-30"): float("nan"),
        }
        fetched_fields = []
        q4_value_kinds = []

        async def fake_fetch_sec_field_rows(cik, field):
            self.assertEqual(cik, "0000320193")
            fetched_fields.append(field)
            return frames_by_field[field]

        def fake_lookup_sec_field_value(
            field, cik, metadata, sec_rows, *, q4_value_kind
        ):
            q4_value_kinds.append((field, q4_value_kind))
            return values_by_field_and_date[(field, metadata.date)]

        with (
            patch.object(
                valuation,
                "fetch_sec_field_rows",
                side_effect=fake_fetch_sec_field_rows,
            ),
            patch.object(
                valuation,
                "lookup_sec_field_value",
                side_effect=fake_lookup_sec_field_value,
            ),
        ):
            values = await valuation.fetch_sec_values_for_quarters(
                "0000320193",
                metadata_by_quarter,
                [
                    "revenue",
                    "weightedAverageShsOutDil",
                    "epsDiluted",
                    "revenue",
                ],
            )

        self.assertEqual(fetched_fields, ["weightedAverageShsOutDil", "epsDiluted"])
        self.assertEqual(
            q4_value_kinds,
            [
                ("weightedAverageShsOutDil", "average"),
                ("weightedAverageShsOutDil", "average"),
                ("epsDiluted", "flow"),
                ("epsDiluted", "flow"),
            ],
        )
        self.assertEqual(
            values,
            {
                "2025-03-31": {
                    "weightedAverageShsOutDil": 12,
                    "epsDiluted": 1.2,
                }
            },
        )

    async def test_skips_an_empty_field_frame(self):
        metadata_by_quarter = {
            "2025-03-31": valuation.SecQuarterMetadata("2025-03-31", "Q1"),
        }

        with (
            patch.object(
                valuation,
                "fetch_sec_field_rows",
                return_value=sec_frame([]),
            ),
            patch.object(
                valuation,
                "lookup_sec_field_value",
                side_effect=AssertionError("SEC lookup should not run"),
            ),
        ):
            values = await valuation.fetch_sec_values_for_quarters(
                "0000320193",
                metadata_by_quarter,
                ["weightedAverageShsOutDil"],
            )

        self.assertEqual(values, {})

    async def test_continues_when_a_field_has_no_sec_data(self):
        metadata_by_quarter = {
            "2025-03-31": valuation.SecQuarterMetadata("2025-03-31", "Q1"),
        }

        async def fake_fetch_sec_field_rows(cik, field):
            if field == "weightedAverageShsOutDil":
                return sec_frame([])
            return sec_frame([{}])

        with (
            patch.object(
                valuation,
                "fetch_sec_field_rows",
                side_effect=fake_fetch_sec_field_rows,
            ),
            patch.object(valuation, "lookup_sec_field_value", return_value=1.2),
        ):
            values = await valuation.fetch_sec_values_for_quarters(
                "0000320193",
                metadata_by_quarter,
                ["weightedAverageShsOutDil", "epsDiluted"],
            )

        self.assertEqual(values, {"2025-03-31": {"epsDiluted": 1.2}})


class AddSecReferenceValuesTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_sec_reference_values_adds_supported_repair_fields(self):
        source_rows = {
            "fmp": {
                "2025-03-31": {
                    "revenue": 100,
                    "weightedAverageShsOutDil": 10,
                    "cik": "320193",
                    "quarter": "Q1",
                }
            }
        }
        aligned_quarters = valuation.build_aligned_source_quarters(source_rows)
        frames_by_field = {
            "weightedAverageShsOutDil": sec_frame(
                [
                    {
                        "filed": "2025-04-20",
                        "val": 12,
                        "start": "2025-01-01",
                        "end": "2025-03-31",
                    }
                ]
            ),
        }

        fetched_fields = []

        async def fake_fetch_sec_field_rows(cik, field):
            self.assertEqual(cik, "0000320193")
            fetched_fields.append(field)
            return frames_by_field[field]

        with patch.object(
            valuation,
            "fetch_sec_field_rows",
            side_effect=fake_fetch_sec_field_rows,
        ):
            await valuation.add_sec_reference_values(
                "AAPL",
                aligned_quarters,
                ["revenue", "weightedAverageShsOutDil", "revenue"],
            )

        self.assertEqual(fetched_fields, ["weightedAverageShsOutDil"])
        self.assertNotIn("sec", aligned_quarters["2025-03-31"].fields["revenue"])
        self.assertEqual(
            aligned_quarters["2025-03-31"].fields["weightedAverageShsOutDil"]["sec"],
            12,
        )

    async def test_add_sec_reference_values_skips_without_cik_or_metadata(self):
        without_cik = valuation.build_aligned_source_quarters(
            {"fmp": {"2025-03-31": {"revenue": 100}}}
        )
        without_metadata = valuation.build_aligned_source_quarters(
            {"fmp": {"2025-03-31": {"revenue": 100, "cik": "320193"}}}
        )

        async def fail_if_called(cik, field):
            raise AssertionError("SEC fetch should not run")

        with patch.object(
            valuation, "fetch_sec_field_rows", side_effect=fail_if_called
        ):
            await valuation.add_sec_reference_values("AAPL", without_cik, ["revenue"])
            await valuation.add_sec_reference_values(
                "AAPL",
                without_metadata,
                ["revenue"],
            )

        self.assertEqual(
            without_cik["2025-03-31"].fields,
            {"revenue": {"fmp": 100}},
        )
        self.assertEqual(
            without_metadata["2025-03-31"].fields,
            {"revenue": {"fmp": 100}},
        )


if __name__ == "__main__":
    unittest.main()
