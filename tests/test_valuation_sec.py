import asyncio
import unittest
from unittest.mock import ANY, AsyncMock, patch

import pandas as pd
from backend import valuation


def sec_frame(rows):
    return pd.DataFrame(rows, columns=valuation.SEC_DEDUPE_COLS)


class SecValuationTests(unittest.TestCase):
    def test_consensus_helpers_select_and_average_agreeing_sources(self):
        source_values = {'fmp': 100, 'finnhub': 102, 'sec': None}

        values = valuation.usable_source_values(source_values)
        groups = valuation.build_consensus_groups('revenue', values)
        group = valuation.choose_consensus_group(groups)

        self.assertEqual(values, [('fmp', 100), ('finnhub', 102)])
        self.assertEqual(group, (('fmp', 100), ('finnhub', 102)))
        self.assertEqual(valuation.average_consensus_group(group), 101)
        self.assertEqual(
            valuation.select_consensus_source_value('revenue', source_values),
            (101, ('fmp', 'finnhub')),
        )

    def test_consensus_helpers_reject_disagreeing_sources(self):
        source_values = {'fmp': 100, 'finnhub': 120}

        values = valuation.usable_source_values(source_values)

        self.assertEqual(valuation.build_consensus_groups('revenue', values), [])
        self.assertIsNone(valuation.choose_consensus_group([]))
        self.assertEqual(
            valuation.select_consensus_source_value('revenue', source_values),
            (None, None),
        )

    def test_consensus_helpers_keep_transitive_consensus_groups(self):
        source_values = {'fmp': 100, 'finnhub': 106, 'sec': 112}

        self.assertEqual(
            valuation.select_consensus_source_value('revenue', source_values),
            (106, ('fmp', 'finnhub', 'sec')),
        )

    def test_choose_consensus_group_prefers_sec_in_a_tie(self):
        groups = [
            (('fmp', 100), ('finnhub', 100)),
            (('massive', 200), ('sec', 200)),
        ]

        self.assertEqual(
            valuation.choose_consensus_group(groups),
            (('massive', 200), ('sec', 200)),
        )

    def test_choose_consensus_group_rejects_tie_without_sec(self):
        groups = [
            (('fmp', 100), ('finnhub', 100)),
            (('massive', 200), ('eodhd', 200)),
        ]

        self.assertIsNone(valuation.choose_consensus_group(groups))

    def test_consensus_helpers_apply_eps_absolute_tolerance(self):
        source_values = {'fmp': 0.001, 'finnhub': -0.01}

        value, sources = valuation.select_consensus_source_value(
            'epsDiluted', source_values
        )

        self.assertAlmostEqual(value, -0.0045)
        self.assertEqual(sources, ('fmp', 'finnhub'))

    def test_diagnostics_reports_per_vendor_missing_counts(self):
        def row(fmp, finnhub):
            return {
                field: {'fmp': fmp, 'finnhub': finnhub}
                for field in valuation.ALL_XPS_FIELDS
            }

        rows = {
            '2025-03-31': row(100, 100),
            '2025-06-30': row(100, None),
            '2025-09-30': row(None, 100),
            '2025-12-31': row(None, None),
        }
        with (
            patch.object(valuation, '_xps_diagnostics_seen', set()),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
        ):
            valuation.record_xps_diagnostics('MISSINGNESS-TEST', rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics['total_quarters'], 4)
        revenue = diagnostics['missing']['revenue']
        self.assertNotIn('sec', revenue)
        self.assertEqual(revenue['fmp'], 2)
        self.assertEqual(revenue['finnhub'], 2)
        self.assertEqual(diagnostics['missing']['weightedAverageShsOutDil'], revenue)
        self.assertEqual(diagnostics['missing']['epsDiluted'], revenue)
        self.assertEqual(diagnostics['consensus_pairs']['revenue']['fmp:finnhub'], 1)
        self.assertEqual(
            diagnostics['consensus_pairs']['weightedAverageShsOutDil']['fmp:finnhub'],
            1,
        )
        self.assertEqual(diagnostics['consensus_pairs']['epsDiluted']['fmp:finnhub'], 1)

    def test_missing_counts_exclude_unavailable_sources(self):
        with (
            patch.object(valuation, '_xps_diagnostics_seen', set()),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
        ):
            valuation.record_xps_diagnostics(
                'UNAVAILABLE-SOURCE-TEST',
                {'2025-03-31': {'revenue': {'fmp': None, 'finnhub': None}}},
                unavailable_sources=frozenset({'fmp'}),
            )
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics['total_quarters'], 1)
        revenue = diagnostics['missing']['revenue']
        self.assertEqual(revenue['fmp'], 0)
        self.assertEqual(revenue['finnhub'], 1)
        self.assertTrue(
            all(
                count == 0
                for count in diagnostics['consensus_pairs']['revenue'].values()
            )
        )

    def test_diagnostics_counts_each_directly_agreeing_pair_once(self):
        rows = {'2025-03-31': {'revenue': {'fmp': 100, 'massive': 104, 'finnhub': 108}}}
        with (
            patch.object(valuation, 'BASE_SOURCE_ORDER', ('fmp', 'massive', 'finnhub')),
            patch.object(valuation, '_xps_diagnostics_seen', set()),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
        ):
            valuation.record_xps_diagnostics('PAIR-TEST', rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(
            diagnostics['consensus_pairs']['revenue'],
            {
                'fmp:massive': 1,
                'fmp:finnhub': 0,
                'massive:finnhub': 1,
            },
        )
        self.assertNotIn('sec', diagnostics['consensus_pairs']['revenue'])

    def test_diagnostics_deduplicates_missing_and_pair_counts(self):
        rows = {
            '2025-03-31': {
                'revenue': {'fmp': 100, 'finnhub': 100},
            }
        }
        with (
            patch.object(valuation, 'BASE_SOURCE_ORDER', ('fmp', 'finnhub')),
            patch.object(valuation, '_xps_diagnostics_seen', set()),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
        ):
            valuation.record_xps_diagnostics('DEDUP-TEST', rows)
            valuation.record_xps_diagnostics('DEDUP-TEST', rows)
            diagnostics = valuation.get_xps_diagnostics()

        self.assertEqual(diagnostics['total_quarters'], 1)
        self.assertEqual(diagnostics['missing']['revenue'], {'fmp': 0, 'finnhub': 0})
        self.assertEqual(diagnostics['consensus_pairs']['revenue'], {'fmp:finnhub': 1})

    def test_diagnostics_skips_request_that_falls_short_after_latest_drop(self):
        def row(revenue):
            return {
                'revenue': revenue,
                'weightedAverageShsOutDil': 10,
                'epsDiluted': revenue / 100,
            }

        source_rows = {
            'fmp': {
                '2025-03-31': row(100),
                '2025-06-30': row(110),
                '2025-09-30': row(120),
                '2025-12-31': row(130),
            },
            'finnhub': {
                '2025-03-31': row(100),
                '2025-06-30': row(110),
                '2025-09-30': row(120),
            },
        }
        seen = set()
        with (
            patch.object(valuation, '_xps_diagnostics_seen', seen),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
            patch.object(valuation, 'merge_sec_fields', new=AsyncMock()),
            self.assertRaises(ValueError),
        ):
            asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'SPCX', source_rows, limit=4, include_eps=True
                )
            )

        self.assertEqual(seen, set())
        diagnostics = valuation.get_xps_diagnostics()
        self.assertEqual(diagnostics['total_quarters'], 0)
        self.assertEqual(diagnostics['missing']['revenue']['finnhub'], 0)

    def test_resolve_us_income_statement_quarters_accepts_source_rows(self):
        source_rows = {
            # Insertion order intentionally differs from BASE_SOURCE_ORDER.
            'finnhub': {
                '2025-04-05': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                }
            },
            'fmp': {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                }
            },
        }

        with patch.object(valuation, 'merge_sec_fields', new=AsyncMock()):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'AAPL', source_rows, 1, include_eps=True
                )
            )

        self.assertEqual(
            resolved,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_without_eps(self):
        source_rows = {
            'fmp': {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                }
            },
            'finnhub': {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                }
            },
        }

        with (
            patch.object(valuation, 'merge_sec_fields', new=AsyncMock()) as merge_sec,
            patch.object(valuation, 'record_xps_diagnostics') as diagnostics,
        ):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'AAPL', source_rows, limit=1, include_eps=False
                )
            )

        merge_sec.assert_awaited_once_with(
            'AAPL',
            ANY,
            source_rows['fmp'],
            {},
            ['revenue', 'weightedAverageShsOutDil'],
        )
        diagnostics.assert_not_called()
        self.assertEqual(
            resolved,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_drops_latest_single_source_row(
        self,
    ):
        source_rows = {
            'fmp': {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                },
                '2025-06-30': {
                    'revenue': 120,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1.2,
                },
            },
            'finnhub': {
                '2025-04-05': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                }
            },
        }

        with patch.object(
            valuation, 'merge_sec_fields', new=AsyncMock()
        ) as merge_sec_fields:
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'AAPL', source_rows, limit=1, include_eps=True
                )
            )

        merge_sec_fields.assert_awaited_once()
        self.assertEqual(
            resolved,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_merges_sec_before_dropping_latest(
        self,
    ):
        source_rows = {
            'fmp': {
                '2025-06-30': {
                    'revenue': 120,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1.2,
                }
            },
            'finnhub': {'2025-06-30': {'revenue': 121}},
        }

        async def add_sec_fields(
            symbol, aligned_quarters, fmp_rows, massive_rows, required_fields
        ):
            self.assertEqual(symbol, 'AAPL')
            self.assertIs(fmp_rows, source_rows['fmp'])
            self.assertEqual(massive_rows, {})
            self.assertEqual(
                required_fields,
                ['revenue', 'weightedAverageShsOutDil', 'epsDiluted'],
            )
            latest_quarter = aligned_quarters['2025-06-30']
            latest_quarter['weightedAverageShsOutDil']['sec'] = 10
            latest_quarter['epsDiluted']['sec'] = 1.2

        with patch.object(
            valuation,
            'merge_sec_fields',
            new=AsyncMock(side_effect=add_sec_fields),
        ) as merge_sec_fields:
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'AAPL', source_rows, limit=1, include_eps=True
                )
            )

        merge_sec_fields.assert_awaited_once()
        self.assertEqual(
            resolved,
            {
                '2025-06-30': {
                    'revenue': 120.5,
                    'weightedAverageShsOutDil': 10,
                    'epsDiluted': 1.2,
                }
            },
        )

    def test_resolve_us_income_statement_quarters_records_selected_quarters_only(
        self,
    ):
        def row(revenue):
            return {
                'revenue': revenue,
                'weightedAverageShsOutDil': 10,
                'epsDiluted': revenue / 100,
            }

        source_rows = {
            'fmp': {
                '2025-03-31': row(100),
                '2025-06-30': row(110),
            },
            'finnhub': {
                '2025-03-31': row(100),
                '2025-06-30': row(110),
            },
        }
        seen = set()
        with (
            patch.object(valuation, '_xps_diagnostics_seen', seen),
            patch.object(
                valuation,
                '_xps_missing_counts',
                valuation.new_xps_missing_counts(),
            ),
            patch.object(
                valuation,
                '_xps_consensus_pair_counts',
                valuation.new_xps_consensus_pair_counts(),
            ),
            patch.object(valuation, 'merge_sec_fields', new=AsyncMock()),
        ):
            resolved = asyncio.run(
                valuation.resolve_us_income_statement_quarters(
                    'AAPL', source_rows, limit=1, include_eps=True
                )
            )

        self.assertEqual(list(resolved), ['2025-06-30'])
        self.assertEqual(seen, {('AAPL', '2025-06-30')})

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
                    '2026-03-31': {},
                    '2025-03-31': {},
                },
                5,
                symbol='SPCX',
                quarter_label='aligned quarters',
            )

    def test_format_sec_cik_normalizes_values(self):
        self.assertEqual(valuation.format_sec_cik('320193'), '0000320193')
        self.assertEqual(valuation.format_sec_cik(320193.0), '0000320193')
        self.assertEqual(valuation.format_sec_cik('CIK 0000320193'), '0000320193')
        self.assertIsNone(valuation.format_sec_cik(None))
        self.assertIsNone(valuation.format_sec_cik('not-a-cik'))

    def test_normalize_massive_fiscal_quarter(self):
        self.assertEqual(valuation.normalize_massive_fiscal_quarter(1), 'Q1')
        self.assertEqual(valuation.normalize_massive_fiscal_quarter('q4'), 'Q4')
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter(5))
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter('FY'))
        self.assertIsNone(valuation.normalize_massive_fiscal_quarter(None))

    def test_dedupe_sec_rows_keeps_latest_supported_filing_per_period(self):
        rows = [
            {
                'form': '10-Q',
                'filed': '2025-04-20',
                'val': 10,
                'start': '2025-01-01',
                'end': '2025-03-31',
            },
            {
                'form': '10-Q/A',
                'filed': '2025-05-01',
                'val': 11,
                'start': '2025-01-01',
                'end': '2025-03-31',
            },
            {
                'form': '8-K',
                'filed': '2025-05-02',
                'val': 99,
                'start': '2025-01-01',
                'end': '2025-03-31',
            },
            {
                'form': '10-Q',
                'filed': '2025-05-03',
                'val': 12,
                'start': None,
                'end': '2025-03-31',
            },
        ]

        df = valuation.dedupe_sec_rows(rows)

        self.assertEqual(
            df.to_dict('records'),
            [
                {
                    'filed': '2025-05-01',
                    'val': 11,
                    'start': '2025-01-01',
                    'end': '2025-03-31',
                }
            ],
        )

    def test_dedupe_sec_rows_returns_expected_empty_shape(self):
        self.assertEqual(
            valuation.dedupe_sec_rows([{'form': '8-K'}]).columns.tolist(),
            valuation.SEC_DEDUPE_COLS,
        )
        self.assertTrue(valuation.dedupe_sec_rows([{'form': '8-K'}]).empty)

    def test_select_sec_fact_requires_exactly_one_date_window_match(self):
        rows = sec_frame(
            [
                {
                    'filed': '2025-04-20',
                    'val': 10,
                    'start': '2025-01-01',
                    'end': '2025-03-31',
                }
            ]
        )

        match = valuation.select_sec_fact(
            rows,
            'single match',
            min_start='2024-12-31',
            max_start='2025-02-01',
            min_end='2025-03-01',
            max_end='2025-04-30',
        )

        self.assertEqual(match['val'], 10)
        self.assertIsNone(
            valuation.select_sec_fact(
                rows,
                'strict boundary',
                min_start='2025-01-01',
                max_start='2025-02-01',
                min_end='2025-03-01',
                max_end='2025-04-30',
                log_errors=False,
            )
        )

    def test_select_sec_fact_rejects_ambiguous_matches(self):
        rows = sec_frame(
            [
                {
                    'filed': '2025-04-20',
                    'val': 10,
                    'start': '2025-01-01',
                    'end': '2025-03-31',
                },
                {
                    'filed': '2025-04-21',
                    'val': 11,
                    'start': '2025-01-15',
                    'end': '2025-03-31',
                },
            ]
        )

        self.assertIsNone(
            valuation.select_sec_fact(
                rows,
                'ambiguous',
                min_start='2024-12-31',
                max_start='2025-02-01',
                min_end='2025-03-01',
                max_end='2025-04-30',
                log_errors=False,
            )
        )

    def test_select_sec_quarter_fact_uses_quarter_date_window(self):
        rows = sec_frame(
            [
                {
                    'filed': '2025-04-20',
                    'val': 10,
                    'start': '2025-01-01',
                    'end': '2025-03-31',
                }
            ]
        )

        match = valuation.select_sec_quarter_fact(
            rows, 'quarter', pd.Timestamp('2025-03-31')
        )

        self.assertEqual(match['val'], 10)

    def test_derive_sec_q4_value_uses_annual_less_q1_to_q3(self):
        rows = sec_frame(
            [
                {
                    'filed': '2026-02-01',
                    'val': 100,
                    'start': '2025-01-01',
                    'end': '2025-12-31',
                },
                {
                    'filed': '2025-11-01',
                    'val': 60,
                    'start': '2025-01-01',
                    'end': '2025-09-30',
                },
            ]
        )

        self.assertEqual(
            valuation.derive_sec_q4_value(
                'revenue', rows, 'q4 revenue', pd.Timestamp('2025-12-31')
            ),
            40,
        )

    def test_derive_sec_q4_value_handles_average_share_count(self):
        rows = sec_frame(
            [
                {
                    'filed': '2026-02-01',
                    'val': 25,
                    'start': '2025-01-01',
                    'end': '2025-12-31',
                },
                {
                    'filed': '2025-11-01',
                    'val': 20,
                    'start': '2025-01-01',
                    'end': '2025-09-30',
                },
            ]
        )

        self.assertEqual(
            valuation.derive_sec_q4_value(
                'weightedAverageShsOutDil',
                rows,
                'q4 shares',
                pd.Timestamp('2025-12-31'),
            ),
            40,
        )

    def test_lookup_sec_field_value_prefers_exact_q4_fact_over_derived_value(self):
        rows = sec_frame(
            [
                {
                    'filed': '2026-01-20',
                    'val': 25,
                    'start': '2025-10-01',
                    'end': '2025-12-31',
                },
                {
                    'filed': '2026-02-01',
                    'val': 100,
                    'start': '2025-01-01',
                    'end': '2025-12-31',
                },
                {
                    'filed': '2025-11-01',
                    'val': 60,
                    'start': '2025-01-01',
                    'end': '2025-09-30',
                },
            ]
        )
        metadata = {'cik': '0000320193', 'date': '2025-12-31', 'period': 'Q4'}

        self.assertEqual(
            valuation.lookup_sec_field_value('revenue', metadata, [rows]),
            25,
        )

    def test_lookup_sec_field_value_derives_q4_when_exact_fact_missing(self):
        rows = sec_frame(
            [
                {
                    'filed': '2026-02-01',
                    'val': 100,
                    'start': '2025-01-01',
                    'end': '2025-12-31',
                },
                {
                    'filed': '2025-11-01',
                    'val': 60,
                    'start': '2025-01-01',
                    'end': '2025-09-30',
                },
            ]
        )
        metadata = {'cik': '0000320193', 'date': '2025-12-31', 'period': 'Q4'}

        self.assertEqual(
            valuation.lookup_sec_field_value('revenue', metadata, [rows]),
            40,
        )

    def test_select_sec_cik_prefers_fmp_and_falls_back_to_massive(self):
        fmp_rows = {'2025-03-31': {'cik': '320193'}}
        massive_rows = {'2025-03-31': {'cik': '789019'}}
        self.assertEqual(valuation.select_sec_cik(fmp_rows, massive_rows), '0000320193')
        self.assertEqual(
            valuation.select_sec_cik({'x': {}}, massive_rows), '0000789019'
        )
        self.assertIsNone(valuation.select_sec_cik({'x': {}}, {'y': {}}))

    def test_build_sec_quarter_metadata_uses_fmp_then_massive_priority(self):
        aligned_quarters = {
            '2025-03-31': {},
            '2025-06-30': {},
            '2025-09-30': {},
        }
        fmp_rows = {
            '2025-03-31': {'quarter': 'Q1'},
            '2025-06-29': {'quarter': 'Q2'},
        }
        massive_rows = {
            '2025-06-30': {'quarter': 3},
            '2025-09-30': {'quarter': 3},
            '2025-10-08': {'quarter': 4},
        }

        metadata = valuation.build_sec_quarter_metadata(
            aligned_quarters,
            fmp_rows,
            massive_rows,
            '0000320193',
        )

        self.assertEqual(
            metadata,
            {
                '2025-03-31': {
                    'cik': '0000320193',
                    'date': '2025-03-31',
                    'period': 'Q1',
                },
                '2025-06-30': {
                    'cik': '0000320193',
                    'date': '2025-06-29',
                    'period': 'Q2',
                },
                '2025-09-30': {
                    'cik': '0000320193',
                    'date': '2025-09-30',
                    'period': 'Q3',
                },
            },
        )


class FetchSecValuesForQuartersTests(unittest.IsolatedAsyncioTestCase):
    async def test_skips_revenue_and_returns_only_usable_values(self):
        metadata_by_quarter = {
            '2025-03-31': {'date': '2025-03-31', 'period': 'Q1'},
            '2025-06-30': {'date': '2025-06-30', 'period': 'Q2'},
        }
        frames_by_field = {
            'weightedAverageShsOutDil': [sec_frame([])],
            'epsDiluted': [sec_frame([])],
        }
        values_by_field_and_date = {
            ('weightedAverageShsOutDil', '2025-03-31'): 12,
            ('weightedAverageShsOutDil', '2025-06-30'): 0,
            ('epsDiluted', '2025-03-31'): 1.2,
            ('epsDiluted', '2025-06-30'): float('nan'),
        }
        fetched_fields = []

        async def fake_fetch_sec_field_rows(cik, field):
            self.assertEqual(cik, '0000320193')
            fetched_fields.append(field)
            return frames_by_field[field]

        def fake_lookup_sec_field_value(field, metadata, frames):
            return values_by_field_and_date[(field, metadata['date'])]

        with (
            patch.object(
                valuation,
                'fetch_sec_field_rows',
                side_effect=fake_fetch_sec_field_rows,
            ),
            patch.object(
                valuation,
                'lookup_sec_field_value',
                side_effect=fake_lookup_sec_field_value,
            ),
        ):
            values = await valuation.fetch_sec_values_for_quarters(
                'AAPL',
                '0000320193',
                metadata_by_quarter,
                [
                    'revenue',
                    'weightedAverageShsOutDil',
                    'epsDiluted',
                    'revenue',
                ],
            )

        self.assertEqual(fetched_fields, ['weightedAverageShsOutDil', 'epsDiluted'])
        self.assertEqual(
            values,
            {
                '2025-03-31': {
                    'weightedAverageShsOutDil': 12,
                    'epsDiluted': 1.2,
                }
            },
        )

    async def test_continues_when_a_field_fetch_fails(self):
        metadata_by_quarter = {
            '2025-03-31': {'date': '2025-03-31', 'period': 'Q1'},
        }

        async def fake_fetch_sec_field_rows(cik, field):
            if field == 'weightedAverageShsOutDil':
                raise RuntimeError('temporary SEC error')
            return [sec_frame([])]

        with (
            patch.object(
                valuation,
                'fetch_sec_field_rows',
                side_effect=fake_fetch_sec_field_rows,
            ),
            patch.object(valuation, 'lookup_sec_field_value', return_value=1.2),
        ):
            values = await valuation.fetch_sec_values_for_quarters(
                'AAPL',
                '0000320193',
                metadata_by_quarter,
                ['weightedAverageShsOutDil', 'epsDiluted'],
            )

        self.assertEqual(values, {'2025-03-31': {'epsDiluted': 1.2}})


class MergeSecFieldsTests(unittest.IsolatedAsyncioTestCase):
    async def test_merge_sec_fields_skips_revenue_but_adds_other_fields(self):
        aligned_quarters = {
            '2025-03-31': {
                'revenue': {'fmp': 100},
                'weightedAverageShsOutDil': {'fmp': 10},
            }
        }
        fmp_rows = {'2025-03-31': {'cik': '320193', 'quarter': 'Q1'}}
        frames_by_field = {
            'weightedAverageShsOutDil': [
                sec_frame(
                    [
                        {
                            'filed': '2025-04-20',
                            'val': 12,
                            'start': '2025-01-01',
                            'end': '2025-03-31',
                        }
                    ]
                )
            ],
        }

        fetched_fields = []

        async def fake_fetch_sec_field_rows(cik, field):
            self.assertEqual(cik, '0000320193')
            fetched_fields.append(field)
            return frames_by_field[field]

        with patch.object(
            valuation,
            'fetch_sec_field_rows',
            side_effect=fake_fetch_sec_field_rows,
        ):
            await valuation.merge_sec_fields(
                'AAPL',
                aligned_quarters,
                fmp_rows,
                {},
                ['revenue', 'weightedAverageShsOutDil', 'revenue'],
            )

        self.assertEqual(fetched_fields, ['weightedAverageShsOutDil'])
        self.assertNotIn('sec', aligned_quarters['2025-03-31']['revenue'])
        self.assertEqual(
            aligned_quarters['2025-03-31']['weightedAverageShsOutDil']['sec'], 12
        )

    async def test_merge_sec_fields_skips_when_cik_or_metadata_is_unavailable(self):
        aligned_quarters = {'2025-03-31': {'revenue': {'fmp': 100}}}

        async def fail_if_called(cik, field):
            raise AssertionError('SEC fetch should not run')

        with patch.object(
            valuation, 'fetch_sec_field_rows', side_effect=fail_if_called
        ):
            await valuation.merge_sec_fields(
                'AAPL', aligned_quarters, {}, {}, ['revenue']
            )
            await valuation.merge_sec_fields(
                'AAPL',
                aligned_quarters,
                {'2025-03-31': {'cik': '320193'}},
                {},
                ['revenue'],
            )

        self.assertEqual(aligned_quarters, {'2025-03-31': {'revenue': {'fmp': 100}}})


if __name__ == '__main__':
    unittest.main()
