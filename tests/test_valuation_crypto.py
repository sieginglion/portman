import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

from backend import valuation
import pytest


def monthly_rows(start: str, periods: int) -> list[dict]:
    dates = valuation.pd.date_range(start, periods=periods, freq="MS")
    rows = []
    for index, timestamp in enumerate(dates):
        rows.extend(
            [
                {
                    "timestamp": timestamp.strftime("%Y-%m-%dT00:00:00.000Z"),
                    "metric_id": "fees",
                    "value": 100 + index,
                },
                {
                    "timestamp": timestamp.strftime("%Y-%m-%dT00:00:00.000Z"),
                    "metric_id": "earnings",
                    "value": 10 + index,
                },
                {
                    "timestamp": timestamp.strftime("%Y-%m-%dT00:00:00.000Z"),
                    "metric_id": "token_supply_circulating",
                    "value": 10 + index,
                },
            ]
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
            rows = asyncio.run(
                valuation.fetch_token_terminal_financial_statement("uni")
            )
            cached_get.assert_awaited_once_with(
                valuation.TOKEN_TERMINAL_FINANCIAL_STATEMENT_URL,
                {
                    "batch": "1",
                    "input": ('{"0":{"project_slug":"uniswap","granularity":"month"}}'),
                },
                headers=valuation.token_terminal_headers("uniswap"),
            )

        assert rows == [{"value": 1}]

    def test_xps_rejects_empty_rows(self):
        with pytest.raises(ValueError, match="no usable data"):
            valuation.build_token_terminal_xps_frame([], m=1, include_eps=True)

    def test_xps_uses_monthly_per_token_values_and_omits_current_month(self):
        rows = monthly_rows("2025-01-01", 12)
        rows.extend(monthly_rows("2026-01-01", 1))
        now = valuation.pd.Timestamp("2026-01-15 12:00", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        expected_rps = sum((100 + index) / (10 + index) for index in range(12))
        expected_eps = sum((10 + index) / (10 + index) for index in range(12))
        assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-01-01"]
        assert frame["rps"].iloc[0] == pytest.approx(expected_rps)
        assert frame["eps"].iloc[0] == pytest.approx(expected_eps)

    def test_xps_can_omit_eps(self):
        rows = monthly_rows("2025-01-01", 12)
        rows.extend(monthly_rows("2026-01-01", 1))
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=False,
            )

        assert frame.columns.tolist() == ["rps"]

    def test_xps_skips_missing_fees(self):
        rows = [
            row for row in monthly_rows("2025-01-01", 12) if row["metric_id"] != "fees"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        assert frame.columns.tolist() == ["eps"]

    def test_xps_skips_fees_with_a_missing_month(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if not (
                row["metric_id"] == "fees" and row["timestamp"].startswith("2025-06")
            )
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        assert frame.columns.tolist() == ["eps"]

    def test_xps_skips_missing_earnings(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if row["metric_id"] != "earnings"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        assert frame.columns.tolist() == ["rps"]

    def test_xps_skips_earnings_with_a_missing_month(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if not (
                row["metric_id"] == "earnings"
                and row["timestamp"].startswith("2025-06")
            )
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        assert frame.columns.tolist() == ["rps"]

    def test_xps_rejects_missing_circulating_supply(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if row["metric_id"] != "token_supply_circulating"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="token_supply_circulating"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    def test_xps_rejects_circulating_supply_with_a_missing_month(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if not (
                row["metric_id"] == "token_supply_circulating"
                and row["timestamp"].startswith("2025-06")
            )
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="token_supply_circulating"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    @pytest.mark.parametrize("invalid_supply", [0, -1, "not-a-number"])
    def test_xps_rejects_invalid_circulating_supply(self, invalid_supply):
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        rows = monthly_rows("2025-01-01", 12)
        next(
            row
            for row in rows
            if row["metric_id"] == "token_supply_circulating"
            and row["timestamp"].startswith("2025-06")
        )["value"] = invalid_supply

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="token_supply_circulating"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    def test_xps_rejects_missing_fees_and_earnings(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if row["metric_id"] == "token_supply_circulating"
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="fees and earnings"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    def test_xps_rejects_when_fees_and_earnings_have_separate_gaps(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 12)
            if not (
                (row["metric_id"] == "fees" and row["timestamp"].startswith("2025-06"))
                or (
                    row["metric_id"] == "earnings"
                    and row["timestamp"].startswith("2025-07")
                )
            )
        ]
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="fees and earnings"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    def test_xps_skips_a_metric_for_all_requested_windows_after_a_gap(self):
        rows = [
            row
            for row in monthly_rows("2025-01-01", 13)
            if not (
                row["metric_id"] == "fees" and row["timestamp"].startswith("2025-06")
            )
        ]
        now = valuation.pd.Timestamp("2026-02-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=2,
                include_eps=True,
            )

        assert frame.columns.tolist() == ["eps"]
        assert len(frame) == 2

    def test_xps_ignores_non_numeric_values(self):
        rows = monthly_rows("2025-01-01", 12)
        rows.append(
            {
                "timestamp": "2025-01-01T00:00:00.000Z",
                "metric_id": "fees",
                "value": "not-a-number",
            }
        )
        rows.extend(monthly_rows("2026-01-01", 1))
        now = valuation.pd.Timestamp("2026-01-15", tz="UTC")

        with patch.object(valuation.pd.Timestamp, "now", return_value=now):
            frame = valuation.build_token_terminal_xps_frame(
                rows,
                m=1,
                include_eps=True,
            )

        assert frame["rps"].notna().all()

    def test_xps_rejects_gaps_in_complete_months(self):
        rows = monthly_rows("2025-01-01", 13)
        rows = [row for row in rows if not row["timestamp"].startswith("2025-06")]
        now = valuation.pd.Timestamp("2026-02-15", tz="UTC")

        with (
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
            pytest.raises(ValueError, match="incomplete months"),
        ):
            valuation.build_token_terminal_xps_frame(rows, m=1, include_eps=True)

    def test_fetch_xps_returns_quarterly_crypto_samples(self):
        rows = monthly_rows("2025-01-01", 15)
        now = valuation.pd.Timestamp("2026-04-15", tz="UTC")

        with (
            patch.object(
                valuation,
                "fetch_token_terminal_financial_statement",
                new=AsyncMock(return_value=rows),
            ),
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            frame = asyncio.run(valuation.fetch_xps("c", "UNI", q=1, include_eps=True))

        assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-04-01"]

    def test_fetch_xps_returns_every_third_month_for_multiple_quarters(self):
        rows = monthly_rows("2025-01-01", 18)
        now = valuation.pd.Timestamp("2026-07-15", tz="UTC")

        with (
            patch.object(
                valuation,
                "fetch_token_terminal_financial_statement",
                new=AsyncMock(return_value=rows),
            ),
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            frame = asyncio.run(valuation.fetch_xps("c", "UNI", q=2, include_eps=True))

        assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-04-01", "2026-07-01"]

    def test_crypto_calc_px_uses_monthly_fundamentals_for_price_alignment(self):
        xps = valuation.pd.DataFrame(
            {"rps": [1.0], "eps": [1.0]},
            index=valuation.pd.to_datetime(["2026-01-01"]),
        )
        now = valuation.pd.Timestamp("2026-01-02", tz="UTC")

        with (
            patch.object(valuation.shared, "get_prices", return_value=[2.0]),
            patch.object(
                valuation,
                "fetch_token_terminal_monthly_xps",
                return_value=xps,
            ) as fetch_monthly_xps,
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            result = asyncio.run(valuation.calc_px("c", "UNI", q=1))

        assert result["ps"].tolist() == [2.0]
        fetch_monthly_xps.assert_awaited_once_with("UNI", 4, True)

    def test_crypto_calc_px_allows_a_missing_metric(self):
        xps = valuation.pd.DataFrame(
            {"eps": [1.0]}, index=valuation.pd.to_datetime(["2026-01-01"])
        )
        now = valuation.pd.Timestamp("2026-01-02", tz="UTC")

        with (
            patch.object(valuation.shared, "get_prices", return_value=[2.0]),
            patch.object(
                valuation,
                "fetch_token_terminal_monthly_xps",
                return_value=xps,
            ),
            patch.object(valuation.pd.Timestamp, "now", return_value=now),
        ):
            result = asyncio.run(valuation.calc_px("c", "UNI", q=1))

        assert valuation.pd.isna(result["ps"].iloc[0])
        assert result["pe"].tolist() == [2.0]
