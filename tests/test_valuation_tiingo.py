import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from httpx import HTTPStatusError, Request, Response

from backend import valuation


class TiingoValuationTests(unittest.TestCase):
    def test_tiingo_is_disabled_by_default(self):
        self.assertFalse(valuation.shared.ENABLE_TIINGO_FUNDAMENTALS)
        self.assertNotIn('tiingo', valuation.BASE_SOURCE_ORDER)

    def test_disabled_tiingo_is_not_fetched(self):
        with (
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value={}),
            ),
            patch.object(
                valuation,
                'fetch_massive_income_statements',
                new=AsyncMock(return_value={}),
            ),
            patch.object(
                valuation,
                'fetch_eodhd_income_statements',
                new=AsyncMock(return_value={}),
            ),
            patch.object(
                valuation,
                'fetch_finnhub_income_statements',
                new=AsyncMock(return_value={}),
            ),
            patch.object(
                valuation,
                'fetch_tiingo_income_statements',
                new=AsyncMock(),
            ) as tiingo_fetch,
        ):
            source_rows = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8, True)
            )

        self.assertEqual(len(source_rows), 4)
        tiingo_fetch.assert_not_awaited()

    def test_source_fetch_error_redacts_request_url_and_token(self):
        request = Request(
            'GET',
            'https://api.tiingo.com/tiingo/fundamentals/HOOD/statements?token=secret',
        )
        error = HTTPStatusError(
            'Client error', request=request, response=Response(400, request=request)
        )

        self.assertEqual(valuation.format_source_fetch_error(error), 'HTTP 400')

    def test_normalize_tiingo_income_statement_rows_uses_quarterly_income_fields(self):
        statements = [
            {
                'date': '2025-12-31T00:00:00.000Z',
                'quarter': 0,
                'year': 2025,
                'statementData': {
                    'balanceSheet': [],
                    'cashFlow': [],
                    'incomeStatement': [
                        {'dataCode': 'revenue', 'value': 999},
                    ],
                    'overview': [],
                },
            },
            {
                'date': '2025-09-30T00:00:00.000Z',
                'quarter': 3,
                'year': 2025,
                'statementData': {
                    'balanceSheet': [],
                    'cashFlow': [],
                    'incomeStatement': [
                        {'dataCode': 'revenue', 'value': 100},
                        {'dataCode': 'shareswaDil', 'value': 20},
                        {'dataCode': 'epsDil', 'value': 1.5},
                    ],
                    'overview': [],
                },
            },
        ]

        rows = valuation.normalize_tiingo_income_statement_rows(
            statements, include_eps=True
        )

        self.assertEqual(
            rows,
            {
                '2025-09-30': {
                    'revenue': 100,
                    'weightedAverageShsOutDil': 20,
                    'epsDiluted': 1.5,
                }
            },
        )

    def test_normalize_tiingo_income_statement_rows_skips_eps_when_unneeded(self):
        rows = valuation.normalize_tiingo_income_statement_rows(
            [
                {
                    'date': '2025-09-30',
                    'quarter': 'Q3',
                    'statementData': {
                        'incomeStatement': [
                            {'dataCode': 'revenue', 'value': 100},
                            {
                                'dataCode': 'shareswaDil',
                                'value': 20,
                            },
                            {'dataCode': 'epsDil', 'value': 1.5},
                        ]
                    },
                }
            ],
            include_eps=False,
        )

        self.assertEqual(
            rows['2025-09-30'],
            {'revenue': 100, 'weightedAverageShsOutDil': 20},
        )
