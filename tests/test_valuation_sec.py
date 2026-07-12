import unittest
from unittest.mock import patch

import pandas as pd
from backend import valuation


def sec_frame(rows):
    return pd.DataFrame(rows, columns=valuation.SEC_DEDUPE_COLS)


class SecValuationTests(unittest.TestCase):
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
