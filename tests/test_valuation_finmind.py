import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend import valuation


class FinMindValuationTests(unittest.TestCase):
    def test_income_statement_rows_are_kept_without_balance_sheet_data(self):
        rows = valuation.normalize_finmind_taiwan_income_statement_rows(
            [
                {'date': '2025-03-31', 'type': 'Revenue', 'value': 100},
                {'date': '2025-03-31', 'type': 'EPS', 'value': 1.5},
            ],
            [],
            include_eps=True,
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
