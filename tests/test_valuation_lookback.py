import json
from unittest.mock import AsyncMock, patch

import pytest

from backend import valuation


@pytest.mark.parametrize("market", ["t", "u"])
@pytest.mark.parametrize(
    ("buffer", "expected_start"),
    [(2, "2025-01-22"), (3, "2024-10-22")],
)
async def test_shared_lookback_controls_request_dates_and_preserves_quarter_windows(
    market, buffer, expected_start
):
    # June's quarter is not published yet. March is the newest available quarter.
    published_dates = [
        "2024-12-31",
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
        "2025-12-31",
        "2026-03-31",
    ]
    now = valuation.pd.Timestamp("2026-07-22T12:00:00Z")

    async def cached_response(url, params, *, headers):
        if url == valuation.FINMIND_API_URL:
            date_param = "start_date"
        elif url == valuation.TIINGO_FUNDAMENTALS_STATEMENTS_URL.format("TEST"):
            date_param = "startDate"
        else:
            assert url in (
                valuation.SIMFIN_STATEMENTS_URL,
                valuation.SIMFIN_WEIGHTED_SHARES_URL,
            )
            date_param = "start"
        assert params[date_param] == expected_start
        dates = [date for date in published_dates if date >= params[date_param]]

        if url == valuation.FINMIND_API_URL:
            if params["dataset"] == "TaiwanStockFinancialStatements":
                rows = [
                    {"date": date, "type": field, "value": value}
                    for date in dates
                    for field, value in (("Revenue", 100), ("EPS", 1))
                ]
            else:
                assert params["dataset"] == "TaiwanStockBalanceSheet"
                # The oldest selected quarter needs its predecessor to fill shares.
                rows = [
                    {"date": date, "type": "OrdinaryShare", "value": 100}
                    for date in dates
                    if date != "2025-06-30"
                ]
            return json.dumps({"data": rows})

        if url == valuation.TIINGO_FUNDAMENTALS_STATEMENTS_URL.format("TEST"):
            # The latest quarter cannot reach consensus, requiring the fifth row.
            return json.dumps(
                [
                    {
                        "date": date,
                        "quarter": int(date[5:7]) // 3,
                        "statementData": {
                            "incomeStatement": [
                                {
                                    "dataCode": field,
                                    "value": None if date == "2026-03-31" else value,
                                }
                                for field, value in (
                                    ("revenue", 100),
                                    ("shareswaDil", 10),
                                    ("epsDil", 1),
                                )
                            ]
                        },
                    }
                    for date in dates
                ]
            )

        if url == valuation.SIMFIN_STATEMENTS_URL:
            return json.dumps(
                [
                    {
                        "id": 1,
                        "statements": [
                            {
                                "statement": "PL",
                                "data": [
                                    {
                                        "Fiscal Period": f"Q{int(date[5:7]) // 3}",
                                        "Fiscal Year": int(date[:4]),
                                        "Report Date": date,
                                        "Revenue": 100,
                                        "Net Income Available to Common Shareholders": 10,
                                    }
                                    for date in dates
                                ],
                            }
                        ],
                    }
                ]
            )
        return json.dumps(
            [
                {
                    "pid": 1,
                    "fyear": int(date[:4]),
                    "period": f"Q{int(date[5:7]) // 3}",
                    "diluted": 10,
                }
                for date in dates
            ]
        )

    sources = tuple(
        source
        for source in valuation.US_INCOME_STATEMENT_SOURCES
        if source.name in ("tiingo", "simfin")
    )
    with (
        patch.object(valuation, "QUARTER_LOOKBACK_BUFFER", buffer),
        patch.object(valuation, "ENABLED_US_INCOME_STATEMENT_SOURCES", sources),
        patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
        patch.object(valuation.shared, "ENABLE_XPS_DIAGNOSTICS", False),
        patch.object(
            valuation.pd.Timestamp, "now", side_effect=lambda *, tz: now.tz_convert(tz)
        ),
        patch.object(
            valuation.shared, "cached_get", new=AsyncMock(side_effect=cached_response)
        ) as cached_get,
        patch.object(
            valuation,
            "resolve_us_income_statement_quarters",
            new=AsyncMock(wraps=valuation.resolve_us_income_statement_quarters),
        ) as resolve_quarters,
    ):
        result = await valuation.fetch_resolved_income_statement_quarters(
            market, "TEST", limit=4
        )

    if market == "t":
        assert cached_get.await_count == 2
        resolve_quarters.assert_not_awaited()
        selected_dates = published_dates[-4:]
    else:
        assert cached_get.await_count == 3
        resolve_quarters.assert_awaited_once()
        source_rows = resolve_quarters.call_args.args[1]
        assert set(source_rows) == {"tiingo", "simfin"}
        for rows in source_rows.values():
            assert list(rows) == published_dates[-5:]
        selected_dates = published_dates[-5:-1]

    assert result == {
        "rps": {
            date: {"revenue": 100, "weightedAverageShsOutDil": 10}
            for date in selected_dates
        },
        "eps": {date: {"epsDiluted": 1} for date in selected_dates},
    }
