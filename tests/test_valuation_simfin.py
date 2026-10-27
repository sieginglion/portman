import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import HTTPStatusError, Request, Response

from backend import valuation


def company_rows(rows, company_id=111052):
    return [
        {
            "id": company_id,
            "statements": [{"statement": "PL", "data": rows}],
        }
    ]


def income_row(**overrides):
    return {
        "Fiscal Period": "Q2",
        "Fiscal Year": 2025,
        "Report Date": "2025-03-31",
        "TTM": 0,
        "Revenue": 100,
        "Net Income Available to Common Shareholders": 15,
        **overrides,
    }


def share_row(**overrides):
    return {
        "pid": 111052,
        "endDate": "2025-03-29",
        "fyear": 2025,
        "period": "Q2",
        "basic": 10,
        "diluted": 20,
        **overrides,
    }


def test_simfin_joins_by_company_and_fiscal_period_and_excludes_cumulative_data():
    companies = company_rows(
        [
            income_row(),
            income_row(**{"Fiscal Period": "FY", "Revenue": 999}),
            income_row(TTM=1, Revenue=888),
        ]
    )
    companies[0]["statements"].append(
        {"statement": "BS", "data": [income_row(Revenue=777)]}
    )
    shares = [
        share_row(),
        share_row(period="H1", diluted=999),
        share_row(period="FY", diluted=888),
        share_row(pid=999, diluted=777),
        share_row(fyear=2024, diluted=666),
    ]

    assert valuation.normalize_simfin_income_statement_rows(companies, shares) == {
        "2025-03-31": {
            "revenue": 100,
            "weightedAverageShsOutDil": 20,
            "epsDiluted": 0.75,
            "quarter": "Q2",
        }
    }


def test_simfin_uses_net_revenue_for_insurance_statements():
    row = income_row(**{"Net Revenue": 100})
    row.pop("Revenue")
    companies = company_rows([row])
    companies[0]["template"] = "INSURANCES"

    result = valuation.normalize_simfin_income_statement_rows(companies, [share_row()])

    assert result["2025-03-31"]["revenue"] == 100


@pytest.mark.parametrize("eps", [-0.5, 0, 0.5])
def test_simfin_prefers_provider_diluted_eps_and_matches_derived_periods(eps):
    companies = company_rows([income_row()])
    derived_row = {
        "Fiscal Year": 2025,
        "Fiscal Period": "Q2",
        "TTM": 0,
        "Earnings Per Share, Basic": 999,
        "Earnings Per Share, Diluted": eps,
    }
    companies[0]["statements"].append(
        {
            "statement": "DERIVED",
            "data": [
                derived_row,
                {**derived_row, "TTM": 1, "Earnings Per Share, Diluted": 888},
                {
                    **derived_row,
                    "Fiscal Period": "FY",
                    "Earnings Per Share, Diluted": 777,
                },
                {
                    **derived_row,
                    "Fiscal Year": 2024,
                    "Earnings Per Share, Diluted": 666,
                },
            ],
        }
    )

    result = valuation.normalize_simfin_income_statement_rows(companies, [share_row()])

    assert result["2025-03-31"]["epsDiluted"] == eps


def test_simfin_keeps_provider_eps_when_diluted_shares_are_missing():
    companies = company_rows([income_row()])
    companies[0]["statements"].append(
        {
            "statement": "DERIVED",
            "data": [
                {
                    "Fiscal Year": 2025,
                    "Fiscal Period": "Q2",
                    "TTM": 0,
                    "Earnings Per Share, Diluted": 0.5,
                }
            ],
        }
    )

    result = valuation.normalize_simfin_income_statement_rows(companies, [])

    assert result["2025-03-31"]["epsDiluted"] == 0.5
    assert result["2025-03-31"]["weightedAverageShsOutDil"] is None


@pytest.mark.parametrize("earnings", [-15, 0, 15])
def test_simfin_derives_eps_from_common_shareholder_income(earnings):
    row = income_row(**{"Net Income Available to Common Shareholders": earnings})
    row["Net Income"] = 999
    result = valuation.normalize_simfin_income_statement_rows(
        company_rows([row]), [share_row()]
    )

    assert result["2025-03-31"]["epsDiluted"] == earnings / 20


@pytest.mark.parametrize("shares", [None, 0, -1, float("nan"), "invalid"])
def test_simfin_rejects_invalid_diluted_shares_without_using_basic_shares(shares):
    result = valuation.normalize_simfin_income_statement_rows(
        company_rows([income_row()]), [share_row(diluted=shares)]
    )

    assert result["2025-03-31"]["weightedAverageShsOutDil"] is None
    assert result["2025-03-31"]["epsDiluted"] is None
    assert result["2025-03-31"]["revenue"] == 100


def test_simfin_missing_shares_or_earnings_preserves_available_fields():
    companies = company_rows(
        [income_row(**{"Net Income Available to Common Shareholders": None})]
    )
    with_shares = valuation.normalize_simfin_income_statement_rows(
        companies, [share_row()]
    )["2025-03-31"]
    without_shares = valuation.normalize_simfin_income_statement_rows(companies, [])[
        "2025-03-31"
    ]

    assert with_shares["weightedAverageShsOutDil"] == 20
    assert with_shares["epsDiluted"] is None
    assert without_shares["weightedAverageShsOutDil"] is None
    assert without_shares["revenue"] == 100


def test_simfin_normalizes_numeric_strings_and_skips_invalid_dates():
    result = valuation.normalize_simfin_income_statement_rows(
        company_rows(
            [
                income_row(**{"Report Date": "invalid"}),
                income_row(
                    Revenue="100",
                    **{"Net Income Available to Common Shareholders": "-15"},
                ),
            ]
        ),
        [share_row(diluted="20")],
    )

    assert list(result) == ["2025-03-31"]
    assert result["2025-03-31"]["revenue"] == 100
    assert result["2025-03-31"]["epsDiluted"] == -0.75


def test_simfin_fetch_uses_restated_quarters_authentication_and_latest_limit():
    async def cached_response(url, params, *, headers):
        if url == valuation.SIMFIN_STATEMENTS_URL:
            return json.dumps(
                company_rows(
                    [
                        income_row(),
                        income_row(
                            **{"Fiscal Period": "Q1", "Report Date": "2024-12-31"}
                        ),
                    ]
                )
            )
        return json.dumps([share_row()])

    with (
        patch.object(valuation, "SIMFIN_API_KEY", "test-key"),
        patch.object(valuation, "quarterly_start_date", return_value="2024-01-01"),
        patch.object(
            valuation.shared, "cached_get", new=AsyncMock(side_effect=cached_response)
        ) as cached_get,
    ):
        rows = asyncio.run(valuation.fetch_simfin_income_statements("AAPL", 1))

    assert list(rows) == ["2025-03-31"]
    assert rows["2025-03-31"]["epsDiluted"] == 0.75
    common_params = {
        "ticker": "AAPL",
        "period": "Q1,Q2,Q3,Q4",
        "start": "2024-01-01",
        "ttm": "false",
    }
    cached_get.assert_any_await(
        valuation.SIMFIN_STATEMENTS_URL,
        {**common_params, "statements": "PL,DERIVED", "asreported": "false"},
        headers={"Authorization": "api-key test-key"},
    )
    cached_get.assert_any_await(
        valuation.SIMFIN_WEIGHTED_SHARES_URL,
        common_params,
        headers={"Authorization": "api-key test-key"},
    )


def test_simfin_failure_is_isolated_from_other_providers():
    sources = tuple(
        source
        for source in valuation.US_INCOME_STATEMENT_SOURCES
        if source.name in {"fmp", "simfin"}
    )
    request = Request("GET", valuation.SIMFIN_STATEMENTS_URL)
    error = HTTPStatusError(
        "Rate limited", request=request, response=Response(429, request=request)
    )
    with (
        patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", sources),
        patch.object(
            valuation, "fetch_fmp_income_statements", new=AsyncMock(return_value={})
        ),
        patch.object(valuation.shared, "cached_get", new=AsyncMock(side_effect=error)),
        patch.object(valuation.logger, "warning") as warning,
    ):
        result = asyncio.run(valuation.fetch_us_income_statement_sources("AAPL", 4))

    assert result == {"fmp": {}, "simfin": {}}
    warning.assert_called_once()


@pytest.mark.parametrize(
    ("key", "flag", "enabled"),
    [
        ("test", None, True),
        ("test", "true", True),
        ("test", "false", False),
        ("", "true", False),
    ],
)
def test_simfin_source_enablement(key: str, flag: str | None, enabled: bool):
    env: dict[str, str] = {
        **os.environ,
        "FMP_KEY": "test",
        "FINMIND_KEY": "test",
        "FINNHUB_API_KEY": "test",
        "ENABLE_MASSIVE_FUNDAMENTALS": "false",
        "ENABLE_EODHD_FUNDAMENTALS": "false",
        "FROM_COINGECKO": "",
        "SIMFIN_API_KEY": key,
    }
    if flag is None:
        env.pop("ENABLE_SIMFIN_FUNDAMENTALS", None)
    else:
        env["ENABLE_SIMFIN_FUNDAMENTALS"] = flag
    code = f"""\
import sys
import types

dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda: False
sys.modules['dotenv'] = dotenv

from backend import shared, valuation

assert shared.ENABLE_SIMFIN_FUNDAMENTALS is {enabled}
assert ('simfin' in valuation.us_income_statement_source_order()) is {enabled}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
