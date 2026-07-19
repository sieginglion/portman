import asyncio
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import valuation
from httpx import HTTPStatusError, Request, Response


def selected_sources(*names: str):
    source_by_name = {
        source.name: source for source in valuation.US_INCOME_STATEMENT_SOURCES
    }
    return tuple(source_by_name[name] for name in names)


class TiingoValuationTests(unittest.TestCase):
    def test_disabled_sources_do_not_require_keys_at_startup(self):
        project_root = Path(__file__).resolve().parents[1]
        base_env = {
            **os.environ,
            'FMP_KEY': 'test',
            'FINMIND_KEY': 'test',
            'FINNHUB_API_KEY': 'test',
            'MASSIVE_API_KEY': 'test',
            'EODHD_API_KEY': 'test',
            'FROM_COINGECKO': '',
            'FROM_YAHOO': '',
            'ENABLE_TIINGO_FUNDAMENTALS': 'false',
        }
        for flag, api_key, source in (
            ('ENABLE_MASSIVE_FUNDAMENTALS', 'MASSIVE_API_KEY', 'massive'),
            ('ENABLE_EODHD_FUNDAMENTALS', 'EODHD_API_KEY', 'eodhd'),
        ):
            with self.subTest(source=source):
                env = {
                    **base_env,
                    'ENABLE_MASSIVE_FUNDAMENTALS': 'true',
                    'ENABLE_EODHD_FUNDAMENTALS': 'true',
                    flag: 'false',
                }
                env.pop(api_key)
                code = f'''\
import sys
import types

dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda: False
sys.modules['dotenv'] = dotenv

from backend import shared, valuation

assert not shared.{flag}
assert shared.{api_key} == ''
assert '{source}' not in valuation.us_income_statement_source_order()
assert '{source}' not in valuation.us_income_statement_source_order(include_sec=True)
'''
                result = subprocess.run(
                    [sys.executable, '-c', code],
                    cwd=project_root,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f'{result.stdout}\n{result.stderr}',
                )

    def test_tiingo_defaults_to_disabled_without_environment_flag(self):
        project_root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            'FMP_KEY': 'test',
            'FINMIND_KEY': 'test',
            'FINNHUB_API_KEY': 'test',
            'MASSIVE_API_KEY': 'test',
            'EODHD_API_KEY': 'test',
            'FROM_COINGECKO': '',
            'FROM_YAHOO': '',
            'ENABLE_MASSIVE_FUNDAMENTALS': 'false',
            'ENABLE_EODHD_FUNDAMENTALS': 'false',
        }
        env.pop('ENABLE_TIINGO_FUNDAMENTALS', None)
        code = '''\
import sys
import types

dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda: False
sys.modules['dotenv'] = dotenv

from backend import shared, valuation

assert not shared.ENABLE_TIINGO_FUNDAMENTALS
assert 'tiingo' not in valuation.us_income_statement_source_order()
assert 'tiingo' not in valuation.us_income_statement_source_order(include_sec=True)
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, f'{result.stdout}\n{result.stderr}')

    def test_all_sources_enabled_at_startup_match_the_registry(self):
        project_root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            'FMP_KEY': 'test',
            'FINMIND_KEY': 'test',
            'FINNHUB_API_KEY': 'test',
            'MASSIVE_API_KEY': 'test',
            'EODHD_API_KEY': 'test',
            'FROM_COINGECKO': '',
            'FROM_YAHOO': '',
            'ENABLE_MASSIVE_FUNDAMENTALS': 'true',
            'ENABLE_EODHD_FUNDAMENTALS': 'true',
            'ENABLE_TIINGO_FUNDAMENTALS': 'true',
        }
        code = '''\
import sys
import types

dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda: False
sys.modules['dotenv'] = dotenv

from backend import valuation

expected = ('fmp', 'massive', 'eodhd', 'finnhub', 'tiingo')
assert tuple(source.name for source in valuation.ENABLED_US_INCOME_STATEMENT_SOURCES) == expected
assert valuation.us_income_statement_source_order() == expected
assert valuation.us_income_statement_source_order(include_sec=True) == (*expected, 'sec')
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, f'{result.stdout}\n{result.stderr}')

    def test_disabled_tiingo_is_not_fetched(self):
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp', 'massive', 'eodhd', 'finnhub'),
            ),
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
            fetched = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8)
            )

        self.assertEqual(
            set(fetched.rows), {'fmp', 'massive', 'eodhd', 'finnhub'}
        )
        tiingo_fetch.assert_not_awaited()

    def test_enabled_sources_are_fetched_in_registry_order(self):
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp', 'massive', 'eodhd', 'finnhub', 'tiingo'),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value={}),
            ) as fmp_fetch,
            patch.object(
                valuation,
                'fetch_massive_income_statements',
                new=AsyncMock(return_value={}),
            ) as massive_fetch,
            patch.object(
                valuation,
                'fetch_eodhd_income_statements',
                new=AsyncMock(return_value={}),
            ) as eodhd_fetch,
            patch.object(
                valuation,
                'fetch_finnhub_income_statements',
                new=AsyncMock(return_value={}),
            ) as finnhub_fetch,
            patch.object(
                valuation,
                'fetch_tiingo_income_statements',
                new=AsyncMock(return_value={}),
            ) as tiingo_fetch,
        ):
            fetched = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8)
            )

        self.assertEqual(
            list(fetched.rows),
            ['fmp', 'massive', 'eodhd', 'finnhub', 'tiingo'],
        )
        fmp_fetch.assert_awaited_once_with('u', 'HOOD', 8)
        massive_fetch.assert_awaited_once_with('HOOD', 8)
        eodhd_fetch.assert_awaited_once_with('HOOD', 8)
        finnhub_fetch.assert_awaited_once_with('HOOD', 8)
        tiingo_fetch.assert_awaited_once_with('HOOD', 8)

    def test_failed_source_is_empty_and_marked_unavailable(self):
        fmp_rows = {'2025-03-31': {'revenue': 100}}
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp', 'massive'),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value=fmp_rows),
            ),
            patch.object(
                valuation,
                'fetch_massive_income_statements',
                new=AsyncMock(side_effect=RuntimeError),
            ),
        ):
            fetched = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8)
            )

        self.assertEqual(fetched.rows, {'fmp': fmp_rows, 'massive': {}})
        self.assertEqual(fetched.unavailable_sources, frozenset({'massive'}))

    def test_cancelled_source_propagates(self):
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp'),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(side_effect=asyncio.CancelledError),
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(
                    valuation.fetch_us_income_statement_sources('HOOD', 8)
                )

    def test_us_resolution_receives_fetched_rows_and_availability(self):
        rows = {'fmp': {'2025-03-31': {'revenue': 100}}}
        fetched = valuation.IncomeStatementFetch(
            rows, frozenset({'massive'})
        )
        resolved = {'2025-03-31': {'revenue': 100}}
        with (
            patch.object(
                valuation,
                'fetch_us_income_statement_sources',
                new=AsyncMock(return_value=fetched),
            ) as fetch_sources,
            patch.object(
                valuation,
                'resolve_us_income_statement_quarters',
                new=AsyncMock(return_value=resolved),
            ) as resolve_quarters,
        ):
            result = asyncio.run(
                valuation.fetch_resolved_income_statement_quarters(
                    'u', 'HOOD', 8, True
                )
            )

        self.assertEqual(result, resolved)
        fetch_sources.assert_awaited_once_with('HOOD', 9)
        resolve_quarters.assert_awaited_once_with(
            'HOOD',
            rows,
            8,
            True,
            unavailable_sources=frozenset({'massive'}),
        )

    def test_us_resolution_preserves_disabled_eps_after_fetch(self):
        rows = {'fmp': {'2025-03-31': {'revenue': 100}}}
        fetched = valuation.IncomeStatementFetch(rows, frozenset())
        with (
            patch.object(
                valuation,
                'fetch_us_income_statement_sources',
                new=AsyncMock(return_value=fetched),
            ) as fetch_sources,
            patch.object(
                valuation,
                'resolve_us_income_statement_quarters',
                new=AsyncMock(return_value={}),
            ) as resolve_quarters,
        ):
            asyncio.run(
                valuation.fetch_resolved_income_statement_quarters(
                    'u', 'HOOD', 8, False
                )
            )

        fetch_sources.assert_awaited_once_with('HOOD', 9)
        resolve_quarters.assert_awaited_once_with(
            'HOOD', rows, 8, False, unavailable_sources=frozenset()
        )

    def test_disabled_massive_is_not_fetched(self):
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp', 'eodhd', 'finnhub'),
            ),
            patch.object(
                valuation,
                'fetch_fmp_income_statements',
                new=AsyncMock(return_value={}),
            ),
            patch.object(
                valuation,
                'fetch_massive_income_statements',
                new=AsyncMock(),
            ) as massive_fetch,
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
        ):
            fetched = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8)
            )

        self.assertNotIn('massive', fetched.rows)
        massive_fetch.assert_not_awaited()

    def test_disabled_eodhd_is_not_fetched(self):
        with (
            patch.object(
                valuation,
                'ENABLED_US_INCOME_STATEMENT_SOURCES',
                selected_sources('fmp', 'massive', 'finnhub'),
            ),
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
                new=AsyncMock(),
            ) as eodhd_fetch,
            patch.object(
                valuation,
                'fetch_finnhub_income_statements',
                new=AsyncMock(return_value={}),
            ),
        ):
            fetched = asyncio.run(
                valuation.fetch_us_income_statement_sources('HOOD', 8)
            )

        self.assertNotIn('eodhd', fetched.rows)
        eodhd_fetch.assert_not_awaited()

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

        rows = valuation.normalize_tiingo_income_statement_rows(statements)

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

    def test_normalize_tiingo_income_statement_rows_keeps_eps(self):
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
        )

        self.assertEqual(
            rows['2025-09-30'],
            {'revenue': 100, 'weightedAverageShsOutDil': 20, 'epsDiluted': 1.5},
        )
