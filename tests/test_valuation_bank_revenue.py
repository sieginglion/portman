import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend import valuation

GROSS_REVENUE = 82_460_000_000
NET_REVENUE = 57_347_000_000
INTEREST_EXPENSE = 25_113_000_000
DILUTED_SHARES = 2_694_200_000
QUARTER_END = "2026-06-30"


def bank_sources():
    return tuple(
        source
        for source in valuation.US_INCOME_STATEMENT_SOURCES
        if source.name in {"fmp", "massive", "finnhub", "tiingo", "simfin"}
    )


@pytest.fixture
def jpm_responses():
    """Relevant fields from JPM Q2 2026 responses captured on 2026-10-27."""
    return {
        "fmp": [
            {
                "date": QUARTER_END,
                "period": "Q2",
                "cik": "0000019617",
                "revenue": GROSS_REVENUE,
                "interestExpense": INTEREST_EXPENSE,
                "weightedAverageShsOutDil": DILUTED_SHARES,
                "epsDiluted": 7.7,
            }
        ],
        "massive": {
            "results": [
                {
                    "period_end": QUARTER_END,
                    "fiscal_quarter": 2,
                    "cik": "0000019617",
                    "revenue": GROSS_REVENUE,
                    "cost_of_revenue": INTEREST_EXPENSE,
                    "gross_profit": NET_REVENUE,
                    "diluted_shares_outstanding": DILUTED_SHARES,
                    "diluted_earnings_per_share": 7.7,
                }
            ]
        },
        "finnhub": {
            "symbol": "JPM",
            "financials": [
                {
                    "period": QUARTER_END,
                    "bankInterestIncome": 50624,
                    "bankNonInterestIncome": 31836,
                    "netInterestIncome": 25511,
                    "interestExpense": 25113,
                    "dilutedAverageSharesOutstanding": 2694.2,
                    "dilutedEPS": 7.7,
                }
            ],
        },
        "tiingo": [
            {
                "date": QUARTER_END,
                "quarter": 2,
                "year": 2026,
                "statementData": {
                    "incomeStatement": [
                        {"dataCode": "revenue", "value": NET_REVENUE},
                        {"dataCode": "intexp", "value": 0},
                        {"dataCode": "shareswaDil", "value": DILUTED_SHARES},
                        {"dataCode": "epsDil", "value": 7.7},
                    ]
                },
            }
        ],
        "simfin": [
            {
                "id": 123,
                "template": "BANKS",
                "statements": [
                    {
                        "statement": "PL",
                        "data": [
                            {
                                "Fiscal Year": 2026,
                                "Fiscal Period": "Q2",
                                "Report Date": QUARTER_END,
                                "TTM": 0,
                                "Net Revenue": NET_REVENUE,
                                "Total Interest Income": 50_624_000_000,
                                "Total Non-Interest Income": 31_836_000_000,
                                "Total Interest Expense": -INTEREST_EXPENSE,
                            }
                        ],
                    },
                    {
                        "statement": "DERIVED",
                        "data": [
                            {
                                "Fiscal Year": 2026,
                                "Fiscal Period": "Q2",
                                "TTM": 0,
                                "Earnings Per Share, Diluted": 7.85205,
                            }
                        ],
                    },
                ],
            }
        ],
        "simfin_shares": [
            {"pid": 123, "fyear": 2026, "period": "Q2", "diluted": DILUTED_SHARES}
        ],
    }


def test_jpm_provider_pipeline_resolves_gross_revenue_before_rps(jpm_responses):
    async def cached_response(url, params, *, headers=None):
        if "weighted-shares-outstanding" in url:
            key = "simfin_shares"
        else:
            key = next(
                name
                for name, domain in {
                    "fmp": "financialmodelingprep.com",
                    "massive": "api.massive.com",
                    "finnhub": "finnhub.io",
                    "tiingo": "api.tiingo.com",
                    "simfin": "prod.simfin.com",
                }.items()
                if domain in url
            )
        return json.dumps(jpm_responses[key])

    with (
        patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", bank_sources()),
        patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
        patch.object(
            valuation.shared, "cached_get", new=AsyncMock(side_effect=cached_response)
        ),
    ):
        source_rows = asyncio.run(valuation.fetch_us_income_statement_sources("JPM", 1))
        resolved = asyncio.run(
            valuation.resolve_us_income_statement_quarters("JPM", source_rows, 1, True)
        )
        aligned = valuation.build_aligned_source_quarters(source_rows)[QUARTER_END]

    assert aligned.fields["revenue"] == {
        "fmp": GROSS_REVENUE,
        "massive": GROSS_REVENUE,
        "finnhub": GROSS_REVENUE,
        "tiingo": None,
        "simfin": GROSS_REVENUE,
    }
    assert aligned.fields["epsDiluted"]["tiingo"] == 7.7
    assert resolved[QUARTER_END]["revenue"] == GROSS_REVENUE
    inputs = valuation.build_quarterly_xps_inputs(resolved, True)
    assert inputs.iloc[0]["rps"] == pytest.approx(GROSS_REVENUE / DILUTED_SHARES)
    # Alignment preserves the original provider period for diagnostics.
    assert source_rows["tiingo"][QUARTER_END]["revenue"] == NET_REVENUE


@pytest.mark.parametrize(
    "expense", [INTEREST_EXPENSE, -INTEREST_EXPENSE, "25113000000"]
)
def test_simfin_bank_adds_funding_cost_when_gross_components_are_missing(
    jpm_responses, expense
):
    row = jpm_responses["simfin"][0]["statements"][0]["data"][0]
    row.pop("Total Interest Income")
    row.pop("Total Non-Interest Income")
    row["Total Interest Expense"] = expense

    result = valuation.normalize_simfin_income_statement_rows(
        jpm_responses["simfin"], jpm_responses["simfin_shares"]
    )[QUARTER_END]

    assert result["revenue"] == GROSS_REVENUE
    assert result["is_bank"] is True


def test_simfin_bank_uses_gross_components_without_adding_expense_twice(jpm_responses):
    row = jpm_responses["simfin"][0]["statements"][0]["data"][0]
    row["Net Revenue"] = 1
    row["Total Interest Expense"] = -999

    result = valuation.normalize_simfin_income_statement_rows(
        jpm_responses["simfin"], jpm_responses["simfin_shares"]
    )[QUARTER_END]

    assert result["revenue"] == GROSS_REVENUE


def test_simfin_bank_does_not_substitute_net_revenue_for_missing_gross(jpm_responses):
    row = jpm_responses["simfin"][0]["statements"][0]["data"][0]
    row.pop("Total Non-Interest Income")
    row.pop("Total Interest Expense")

    result = valuation.normalize_simfin_income_statement_rows(
        jpm_responses["simfin"], jpm_responses["simfin_shares"]
    )[QUARTER_END]

    assert result["revenue"] is None
    assert result["weightedAverageShsOutDil"] == DILUTED_SHARES
    assert result["epsDiluted"] == 7.85205


@pytest.mark.parametrize("bank_source", ["finnhub", "simfin"])
@pytest.mark.parametrize("expense", [INTEREST_EXPENSE, -INTEREST_EXPENSE, 0, None])
def test_tiingo_bank_revenue_requires_a_usable_funding_cost(
    bank_source: str, expense: int | None
):
    source_rows: valuation.IncomeStatementSources = {
        bank_source: {QUARTER_END: {"is_bank": True}},
        "tiingo": {
            QUARTER_END: {
                "revenue": NET_REVENUE,
                "interestExpense": expense,
                "weightedAverageShsOutDil": DILUTED_SHARES,
                "epsDiluted": 7.7,
            }
        },
    }
    with patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", bank_sources()):
        aligned = valuation.build_aligned_source_quarters(source_rows)[QUARTER_END]

    expected = GROSS_REVENUE if expense else None
    assert aligned.fields["revenue"]["tiingo"] == expected
    assert aligned.fields["epsDiluted"]["tiingo"] == 7.7
    assert aligned.fields["weightedAverageShsOutDil"]["tiingo"] == DILUTED_SHARES


def test_nonbank_interest_expense_is_not_added_to_tiingo_revenue():
    source_rows: valuation.IncomeStatementSources = {
        "simfin": {QUARTER_END: {"revenue": 100}},
        "tiingo": {QUARTER_END: {"revenue": 100, "interestExpense": 10}},
    }
    with patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", bank_sources()):
        aligned = valuation.build_aligned_source_quarters(source_rows)[QUARTER_END]

    assert aligned.fields["revenue"]["tiingo"] == 100


def test_bank_classification_applies_to_quarters_without_a_bank_provider_row():
    source_rows: valuation.IncomeStatementSources = {
        "finnhub": {"2026-03-31": {"is_bank": True}},
        "tiingo": {QUARTER_END: {"revenue": NET_REVENUE, "interestExpense": 0}},
    }
    with patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", bank_sources()):
        aligned = valuation.build_aligned_source_quarters(source_rows)[QUARTER_END]

    assert aligned.fields["revenue"]["tiingo"] is None


def test_bank_revenue_conversion_respects_existing_fiscal_date_alignment():
    source_rows: valuation.IncomeStatementSources = {
        "finnhub": {"2026-06-27": {"is_bank": True}},
        "tiingo": {
            QUARTER_END: {
                "revenue": NET_REVENUE,
                "interestExpense": INTEREST_EXPENSE,
            }
        },
    }
    with patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", bank_sources()):
        aligned = valuation.build_aligned_source_quarters(source_rows)

    assert len(aligned) == 1
    assert aligned["2026-06-27"].fields["revenue"]["tiingo"] == GROSS_REVENUE


def test_finnhub_missing_bank_components_remain_unavailable(jpm_responses):
    jpm_responses["finnhub"]["financials"][0].pop("bankNonInterestIncome")
    with patch.object(
        valuation.shared,
        "cached_get",
        new=AsyncMock(return_value=json.dumps(jpm_responses["finnhub"])),
    ):
        result = asyncio.run(valuation.fetch_finnhub_income_statements("JPM", 1))

    assert result[QUARTER_END]["revenue"] is None
    assert result[QUARTER_END]["is_bank"] is True
    assert result[QUARTER_END]["epsDiluted"] == 7.7


def test_zero_finnhub_bank_placeholders_do_not_classify_an_ordinary_company():
    payload = {
        "financials": [
            {
                "period": QUARTER_END,
                "revenue": 100,
                "bankInterestIncome": 0,
                "bankNonInterestIncome": 0,
            }
        ]
    }
    with patch.object(
        valuation.shared, "cached_get", new=AsyncMock(return_value=json.dumps(payload))
    ):
        result = asyncio.run(valuation.fetch_finnhub_income_statements("AAPL", 1))

    assert result[QUARTER_END]["revenue"] == 100_000_000
    assert "is_bank" not in result[QUARTER_END]
