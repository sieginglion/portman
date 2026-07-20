import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend import valuation


class FinMindValuationTests(unittest.TestCase):
    def test_build_xps_frame_returns_ttm_values_with_quarter_dates(self):
        quarters = {
            '2025-03-31': {
                'revenue': 100,
                'weightedAverageShsOutDil': 10,
                'epsDiluted': 1.0,
            },
            '2025-06-30': {
                'revenue': 110,
                'weightedAverageShsOutDil': 10,
                'epsDiluted': 1.1,
            },
            '2025-09-30': {
                'revenue': 120,
                'weightedAverageShsOutDil': 12,
                'epsDiluted': 1.2,
            },
            '2025-12-31': {
                'revenue': 130,
                'weightedAverageShsOutDil': 13,
                'epsDiluted': 1.3,
            },
            '2026-03-31': {
                'revenue': 140,
                'weightedAverageShsOutDil': 14,
                'epsDiluted': 1.4,
            },
        }

        frame = valuation.build_xps_frame(quarters, include_eps=True)

        self.assertEqual(frame.columns.tolist(), ['rps', 'eps'])
        self.assertEqual(
            frame.index.strftime('%Y-%m-%d').tolist(),
            ['2026-01-01', '2026-04-01'],
        )
        self.assertEqual(frame['rps'].tolist(), [41.0, 41.0])
        self.assertEqual(frame['eps'].tolist(), [4.6, 5.0])

        without_eps = valuation.build_xps_frame(
            {
                date: {
                    'revenue': row['revenue'],
                    'weightedAverageShsOutDil': row['weightedAverageShsOutDil'],
                }
                for date, row in quarters.items()
            },
            include_eps=False,
        )
        self.assertEqual(without_eps.columns.tolist(), ['rps'])
        self.assertEqual(
            without_eps.index.strftime('%Y-%m-%d').tolist(),
            ['2026-01-01', '2026-04-01'],
        )
        self.assertEqual(without_eps['rps'].tolist(), [41.0, 41.0])

    def test_income_statement_rows_are_kept_without_balance_sheet_data(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {'date': '2025-03-31', 'type': 'Revenue', 'value': 100},
                {'date': '2025-03-31', 'type': 'EPS', 'value': 1.5},
            ],
            [],
        )

        self.assertEqual(
            rows,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': None,
                    'epsDiluted': 1.5,
                }
            },
        )

    def test_merge_preferred_xps_rows_uses_nearby_dates_only(self):
        baseline_rows = {
            '2025-03-31': {
                'revenue': 100,
                'weightedAverageShsOutDil': 20,
                'epsDiluted': 1.0,
            }
        }
        preferred_rows = {
            '2025-04-06': {
                'revenue': 101,
                'weightedAverageShsOutDil': None,
                'epsDiluted': 1.1,
            },
            '2025-04-07': {
                'revenue': 999,
                'weightedAverageShsOutDil': 30,
                'epsDiluted': 9.9,
            },
        }

        valuation.merge_preferred_xps_rows(baseline_rows, preferred_rows)

        self.assertEqual(
            baseline_rows,
            {
                '2025-03-31': {
                    'revenue': 101,
                    'weightedAverageShsOutDil': 20,
                    'epsDiluted': 1.1,
                }
            },
        )

    def test_fetch_keeps_fmp_shares_when_finmind_balance_sheet_is_empty(self):
        finmind_financial_rows = [
            {'date': '2025-03-31', 'type': 'Revenue', 'value': 100},
            {'date': '2025-03-31', 'type': 'EPS', 'value': 1.5},
        ]
        fmp_rows = {
            '2025-03-31': {
                'revenue': 90,
                'weightedAverageShsOutDil': 20,
                'epsDiluted': 1.0,
            }
        }
        with (
            patch.object(
                valuation,
                'fetch_finmind_taiwan_rows',
                new=AsyncMock(side_effect=[finmind_financial_rows, []]),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value=fmp_rows),
            ),
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements(
                    '2330', limit=1, require_eps=True
                )
            )

        self.assertEqual(
            rows,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 20,
                    'epsDiluted': 1.5,
                }
            },
        )

    def test_fetch_without_eps_accepts_finmind_rows_missing_eps(self):
        finmind_financial_rows = [
            {'date': '2025-03-31', 'type': 'Revenue', 'value': 100},
        ]
        fmp_rows = {
            '2025-03-31': {
                'revenue': 90,
                'weightedAverageShsOutDil': 20,
                'epsDiluted': None,
            }
        }
        with (
            patch.object(
                valuation,
                'fetch_finmind_taiwan_rows',
                new=AsyncMock(side_effect=[finmind_financial_rows, []]),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value=fmp_rows),
            ) as fmp_fetch,
        ):
            rows = asyncio.run(
                valuation.fetch_finmind_taiwan_income_statements(
                    '2330', limit=1, require_eps=False
                )
            )

        fmp_fetch.assert_awaited_once_with('t', '2330', 2)
        self.assertEqual(
            rows,
            {
                '2025-03-31': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 20,
                    'epsDiluted': None,
                }
            },
        )
