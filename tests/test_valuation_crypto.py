import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from backend import valuation


def monthly_rows(start: str, periods: int) -> list[dict]:
    rows = []
    for index, timestamp in enumerate(
        valuation.pd.date_range(start, periods=periods, freq="MS")
    ):
        for metric_id, value in (
            ("fees", 100 + index),
            ("earnings", 10 + index),
            ("token_supply_circulating", 10 + index),
        ):
            rows.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%dT00:00:00.000Z"),
                    "metric_id": metric_id,
                    "value": value,
                }
            )
    return rows


class TestTokenTerminalValuation:
    def test_headers_do_not_require_a_cookie(self):
        with patch.dict(os.environ, {"TOKEN_TERMINAL_JWT": "jwt"}, clear=True):
            headers = valuation.token_terminal_headers("uniswap")

        assert headers["x-tt-terminal-jwt"] == "jwt"
        assert "cookie" not in headers

    def test_financial_statement_request_uses_single_tt_procedure(self):
        cached_get = AsyncMock(
            return_value=json.dumps([{"result": {"data": [{"value": 1}]}}])
        )
        with (
            patch.dict(os.environ, {"TOKEN_TERMINAL_JWT": "jwt"}),
            patch.object(valuation.shared, "cached_get", cached_get),
        ):
            expected_headers = valuation.token_terminal_headers("uniswap")
            rows = asyncio.run(
                valuation.fetch_token_terminal_financial_statement("uni")
            )

        assert rows == [{"value": 1}]
        cached_get.assert_awaited_once_with(
            valuation.TOKEN_TERMINAL_FINANCIAL_STATEMENT_URL,
            {
                "batch": "1",
                "input": ('{"0":{"project_slug":"uniswap","granularity":"month"}}'),
            },
            headers=expected_headers,
        )

    def test_builds_raw_dated_rps_and_eps_inputs(self):
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                monthly_rows("2025-01-01", 12), periods=12, include_eps=True
            )

        assert frame.columns.tolist() == ["rps", "eps"]
        assert frame.index.strftime("%Y-%m-%d").tolist() == [
            *valuation.pd.date_range("2025-02-01", periods=12, freq="MS").strftime(
                "%Y-%m-%d"
            )
        ]
        assert frame.notna().all().all()

    def test_raw_inputs_exclude_the_current_month(self):
        rows = monthly_rows("2025-01-01", 13)
        current_month_fees = next(
            row
            for row in rows
            if row["metric_id"] == "fees" and row["timestamp"].startswith("2026-01")
        )
        current_month_fees["value"] = 1_000_000
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows, periods=12, include_eps=True
            )

        assert frame.index.strftime("%Y-%m-%d").tolist()[-1] == "2026-01-01"
        assert frame.loc["2026-01-01", "rps"] == pytest.approx(111 / 21)

    def test_missing_crypto_metric_is_left_for_the_shared_gate(self):
        rows = [
            row for row in monthly_rows("2025-01-01", 12) if row["metric_id"] != "fees"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            inputs = valuation.build_token_terminal_xps_frame(
                rows, periods=12, include_eps=True
            )

        result = valuation.complete_and_roll_xps(
            {str(column): values for column, values in inputs.items()}, 12
        )
        assert list(result) == ["eps"]
        assert result["eps"].tolist() == [12.0]

    def test_missing_crypto_month_is_left_for_the_shared_gate(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if not (
                row["metric_id"] == "fees" and row["timestamp"].startswith("2025-06")
            )
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            inputs = valuation.build_token_terminal_xps_frame(
                rows, periods=12, include_eps=True
            )

        result = valuation.complete_and_roll_xps(
            {str(column): values for column, values in inputs.items()}, 12
        )
        assert valuation.pd.isna(inputs.loc["2025-07-01", "rps"])
        assert list(result) == ["eps"]
        assert result["eps"].tolist() == [12.0]

    def test_missing_supply_fails_at_the_shared_gate(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if row["metric_id"] != "token_supply_circulating"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            inputs = valuation.build_token_terminal_xps_frame(
                rows, periods=12, include_eps=True
            )

        with pytest.raises(ValueError, match="neither a complete"):
            valuation.complete_and_roll_xps(
                {str(column): values for column, values in inputs.items()}, 12
            )

    @pytest.mark.parametrize("invalid_supply", [0, -1, "not-a-number"])
    def test_invalid_supply_fails_at_the_shared_gate(self, invalid_supply):
        rows = monthly_rows("2025-01-01", 12)
        next(
            row
            for row in rows
            if row["metric_id"] == "token_supply_circulating"
            and row["timestamp"].startswith("2025-06")
        )["value"] = invalid_supply
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")
        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            inputs = valuation.build_token_terminal_xps_frame(
                rows, periods=12, include_eps=True
            )

        with pytest.raises(ValueError, match="neither a complete"):
            valuation.complete_and_roll_xps(
                {str(column): values for column, values in inputs.items()}, 12
            )

    def test_fetch_xps_returns_quarterly_crypto_samples(self):
        now = valuation.pd.Timestamp("2026-04-15", tz="UTC")
        with (
            patch.object(
                valuation,
                "fetch_token_terminal_financial_statement",
                new=AsyncMock(return_value=monthly_rows("2025-01-01", 15)),
            ),
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            frame = asyncio.run(valuation.fetch_xps("c", "UNI", q=1))

        assert valuation.pd.DatetimeIndex(frame["rps"].index).strftime(
            "%Y-%m-%d"
        ).tolist() == ["2026-04-01"]

    def test_fetch_xps_returns_every_third_month_for_multiple_quarters(self):
        now = valuation.pd.Timestamp("2026-07-15", tz="UTC")
        with (
            patch.object(
                valuation,
                "fetch_token_terminal_financial_statement",
                new=AsyncMock(return_value=monthly_rows("2025-01-01", 18)),
            ),
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            frame = asyncio.run(valuation.fetch_xps("c", "UNI", q=2))

        assert valuation.pd.DatetimeIndex(frame["rps"].index).strftime(
            "%Y-%m-%d"
        ).tolist() == [
            "2026-04-01",
            "2026-07-01",
        ]

    def test_crypto_calc_px_allows_a_missing_metric(self):
        xps = valuation.pd.DataFrame(
            {"eps": [1.0]}, index=valuation.pd.to_datetime(["2026-01-01"])
        )
        now = valuation.pd.Timestamp("2026-01-02", tz="UTC")
        with (
            patch.object(
                valuation.shared,
                "get_price_series",
                return_value=valuation.pd.Series(
                    [2.0], index=valuation.pd.to_datetime(["2026-01-02"])
                ),
            ),
            patch.object(
                valuation,
                "fetch_token_terminal_monthly_xps",
                return_value={str(column): values for column, values in xps.items()},
            ) as fetch_monthly_xps,
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            result = asyncio.run(valuation.calc_px("c", "UNI", q=1))

        assert valuation.pd.isna(result["ps"].iloc[0])
        assert result["pe"].tolist() == [2.0]
        fetch_monthly_xps.assert_awaited_once_with("UNI", 4, True)
