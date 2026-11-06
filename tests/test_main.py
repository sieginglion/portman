import asyncio
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main, valuation


class TestDownsideScore:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1, 2, 4, 8, 16, 3], 0.5),
            ([2, 3, 4, 5, 1], 0.8),
            ([1, 2, 3, 4, 5], 0.0),
            ([1, 2, 2, 3, 2], 0.2),
            ([2, 2, 2], 0.0),
            ([2], 0.0),
        ],
    )
    def test_inverted_empirical_percentile(self, values, expected):
        assert main.calc_downside_score(pd.Series(values)) == expected

    @pytest.mark.parametrize("values", [[], [1, None, 2], [0, 1], [-1, 1]])
    def test_rejects_invalid_history(self, values):
        with pytest.raises(ValueError, match=r"^$"):
            main.calc_downside_score(pd.Series(values, dtype=float))

    async def test_bitcoin_scores_use_the_valuation_proxy(self):
        series = pd.Series([1, 2, 4, 8, 16, 3])
        with patch.object(main, "get_btc_valuation_proxy", return_value=series) as proxy:
            score = await main.calc_btc_score(16)

        assert score == 0.5
        proxy.assert_awaited_once_with(16)


class TestGrowthRoute:
    @pytest.mark.parametrize("market", ["c", "j", "t", "u"])
    @pytest.mark.parametrize("periods", [0, 7])
    def test_unavailable_histories_return_nulls_through_the_real_pipeline(
        self, market, periods
    ):
        stock_rows = {
            date.strftime("%Y-%m-%d"): {"revenue": 100, "epsDiluted": 1}
            for date in pd.date_range("2024-03-31", periods=periods, freq="QE")
        }
        taiwan_rows = [
            {"date": date, "type": field, "value": row[key]}
            for date, row in stock_rows.items()
            for field, key in (("Revenue", "revenue"), ("EPS", "epsDiluted"))
        ]
        crypto_rows = [
            {"timestamp": date.isoformat(), "metric_id": metric, "value": value}
            for date in pd.date_range("2024-01-01", periods=periods, freq="MS")
            for metric, value in (
                ("fees", 100),
                ("earnings", 10),
                ("token_supply_circulating", 10),
            )
        ]
        with (
            patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
            patch.object(valuation.shared, "ENABLE_XPS_DIAGNOSTICS", False),
            patch.object(
                valuation, "fetch_fmp_income_statements", return_value=stock_rows
            ),
            patch.object(
                valuation, "fetch_finmind_taiwan_rows", return_value=taiwan_rows
            ),
            patch.object(
                valuation,
                "fetch_us_income_statement_sources",
                return_value={"fmp": stock_rows, "finnhub": stock_rows},
            ),
            patch.object(
                valuation,
                "fetch_token_terminal_financial_statement",
                return_value=crypto_rows,
            ),
        ):
            response = TestClient(main.app).get(
                "/growths", params={"market": market, "symbol": "A"}
            )

        assert response.status_code == 200
        assert response.json() == [None, None]

    @pytest.mark.parametrize("market", ["j", "t", "u"])
    @pytest.mark.parametrize(
        "missing", [("revenue",), ("epsDiluted",), ("revenue", "epsDiluted")]
    )
    async def test_incomplete_metrics_are_independent_through_the_real_pipeline(
        self, market, missing
    ):
        rows = {
            date.strftime("%Y-%m-%d"): {"revenue": 100 + 10 * i, "epsDiluted": 1 + i}
            for i, date in enumerate(pd.date_range("2024-03-31", periods=8, freq="QE"))
        }
        for field in missing:
            rows["2024-03-31"][field] = None
        with (
            patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
            patch.object(valuation.shared, "ENABLE_XPS_DIAGNOSTICS", False),
            patch.object(valuation, "fetch_fmp_income_statements", return_value=rows),
            patch.object(
                valuation, "fetch_finmind_taiwan_income_statements", return_value=rows
            ),
            patch.object(
                valuation,
                "fetch_us_income_statement_sources",
                return_value={"fmp": rows, "finnhub": rows},
            ),
        ):
            revenue, eps = await main.get_growths(market, "A")

        if "revenue" in missing:
            assert revenue is None
        else:
            assert revenue == pytest.approx(620 / 460 - 1)
        if "epsDiluted" in missing:
            assert eps is None
        else:
            assert eps == pytest.approx(26 / 10 - 1)

    @pytest.mark.parametrize(
        ("market", "provider"),
        [
            ("c", "fetch_token_terminal_financial_statement"),
            ("j", "fetch_fmp_income_statements"),
            ("t", "fetch_finmind_taiwan_rows"),
            ("u", "fetch_us_income_statement_sources"),
        ],
    )
    @pytest.mark.parametrize("error", [RuntimeError, ValueError])
    def test_provider_failures_are_not_confused_with_unavailable_history(
        self, market, provider, error
    ):
        with (
            patch.object(valuation, provider, side_effect=error("provider failure")),
            pytest.raises(error, match="provider failure"),
        ):
            TestClient(main.app).get(
                "/growths", params={"market": market, "symbol": "A"}
            )

    @pytest.mark.parametrize("market", ["j", "t", "u"])
    @pytest.mark.parametrize("missing_shares", [True, False])
    async def test_uses_ttm_total_revenue_and_diluted_eps(self, market, missing_shares):
        rows = {
            date.strftime("%Y-%m-%d"): {
                "revenue": 100 + 10 * i,
                "weightedAverageShsOutDil": None if missing_shares else 10 + i,
                "epsDiluted": 1 + i,
            }
            for i, date in enumerate(pd.date_range("2024-03-31", periods=8, freq="QE"))
        }
        with (
            patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
            patch.object(valuation.shared, "ENABLE_XPS_DIAGNOSTICS", False),
            patch.object(
                valuation, "fetch_finmind_taiwan_income_statements", return_value=rows
            ),
            patch.object(valuation, "fetch_fmp_income_statements", return_value=rows),
            patch.object(
                valuation,
                "fetch_us_income_statement_sources",
                return_value={"fmp": rows, "finnhub": rows},
            ),
        ):
            revenue, eps = await main.get_growths(market, "A")

        assert revenue == pytest.approx(620 / 460 - 1)
        assert eps == pytest.approx(26 / 10 - 1)

    async def test_us_metrics_keep_independent_latest_quarter_fallbacks(self):
        rows = {
            date.strftime("%Y-%m-%d"): {
                "revenue": 100 + 10 * i,
                "epsDiluted": 1 + i if i < 8 else None,
            }
            for i, date in enumerate(pd.date_range("2024-03-31", periods=9, freq="QE"))
        }
        with (
            patch.object(valuation.shared, "ENABLE_SEC_REFERENCE", False),
            patch.object(valuation.shared, "ENABLE_XPS_DIAGNOSTICS", False),
            patch.object(
                valuation,
                "fetch_us_income_statement_sources",
                return_value={"fmp": rows, "finnhub": rows},
            ),
        ):
            revenue, eps = await main.get_growths("u", "A")

        assert revenue == pytest.approx(660 / 500 - 1)
        assert eps == pytest.approx(26 / 10 - 1)

    @pytest.mark.parametrize(
        ("revenue", "eps", "expected"),
        [
            ([100, 110, 120, 130, 150], [2, 3, 4, 5, 6], [0.5, 2.0]),
            ([100, 100, 100, 100, 80], [2, 2, 2, 2, -1], [-0.2, -1.5]),
            ([100] * 5, [-2, 1, 2, 3, 4], [0.0, None]),
            ([100] * 5, [0, 1, 2, 3, 4], [0.0, None]),
            ([0, 1, 2, 3, 4], [2] * 5, [None, 0.0]),
            (None, [2] * 5, [None, 0.0]),
            ([100] * 5, None, [0.0, None]),
            ([100] * 5, [2] * 4, [0.0, None]),
            ([100] * 5, [2, 2, 2, 2, float("inf")], [0.0, None]),
            ([float("nan"), 1, 2, 3, 4], [2] * 5, [None, 0.0]),
            (None, None, [None, None]),
        ],
    )
    def test_returns_independent_revenue_and_eps_growth(self, revenue, eps, expected):
        series = {
            key: pd.Series(values)
            for key, values in (("revenue", revenue), ("eps", eps))
            if values is not None
        }
        with patch.object(valuation, "fetch_growth_series", return_value=series):
            response = TestClient(main.app).get(
                "/growths", params={"market": "u", "symbol": "A"}
            )

        assert response.status_code == 200
        for actual, value in zip(response.json(), expected, strict=True):
            if value is None:
                assert actual is None
            else:
                assert actual == pytest.approx(value)

    def test_btc_keeps_sma_growth_proxy(self):
        with patch.object(main, "calc_btc_growth", return_value=0.25):
            response = TestClient(main.app).get(
                "/growths", params={"market": "c", "symbol": "BTC"}
            )
        assert response.json() == [0.25, None]

    def test_other_crypto_uses_total_revenue_and_per_token_earnings(self):
        series = {"revenue": pd.Series([2, 3, 4, 5, 6]), "eps": pd.Series([1] * 5)}
        with patch.object(
            valuation, "fetch_growth_series", return_value=series
        ) as fetch_growth_series:
            response = TestClient(main.app).get(
                "/growths", params={"market": "c", "symbol": "UNI"}
            )
        assert response.json() == [2.0, 0.0]
        fetch_growth_series.assert_awaited_once_with("c", "UNI")

    def test_provider_errors_are_not_suppressed(self):
        with (
            patch.object(
                valuation, "fetch_growth_series", side_effect=RuntimeError("provider")
            ),
            pytest.raises(RuntimeError, match="provider"),
        ):
            TestClient(main.app).get("/growths", params={"market": "u", "symbol": "A"})


class TestBtcValuationProxy:
    async def test_uses_a_fixed_sixteen_quarter_moving_average(self):
        with patch.object(
            main, "get_btc_price_to_sma_series", return_value=pd.Series()
        ) as get_btc_price_to_sma_series:
            await main.get_btc_valuation_proxy(4)

        get_btc_price_to_sma_series.assert_awaited_once_with(91 * 4, 91 * 16)


class TestXpsDateLine:
    def test_adds_each_xps_date_within_the_series_range(self):
        ax = Mock()
        series = pd.Series([1, 2], index=pd.to_datetime(["2025-01-15", "2025-10-10"]))
        xps_dates = pd.DatetimeIndex(
            pd.to_datetime(["2025-01-01", "2025-03-31", "2025-06-15", "2025-12-01"])
        )

        main.add_xps_date_lines(ax, series, xps_dates)

        assert [call.args[0] for call in ax.axvline.call_args_list] == list(
            pd.to_datetime(["2025-03-31", "2025-06-15"])
        )


class TestPlotValuationSeries:
    def test_draws_a_non_empty_series_and_applies_axis_labels(self):
        ax = MagicMock()
        series = pd.Series(
            [1.0, 2.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"])
        )
        xps_dates = pd.DatetimeIndex(pd.to_datetime(["2025-01-02"]))

        with (
            patch.object(main, "add_percentile_bands") as add_percentile_bands,
            patch.object(main, "add_xps_date_lines") as add_xps_date_lines,
        ):
            main.plot_valuation_series(
                ax,
                series,
                "AAPL",
                "P/S",
                16,
                "#2563eb",
                xps_dates=xps_dates,
            )

        add_percentile_bands.assert_called_once_with(ax, series)
        add_xps_date_lines.assert_called_once_with(ax, series, xps_dates)
        ax.plot.assert_called_once_with(
            series.index, series, color="#2563eb", linewidth=1.5, zorder=2
        )
        ax.scatter.assert_called_once()
        scatter_args = ax.scatter.call_args
        assert scatter_args.args[0].equals(series.index[-1:])
        assert scatter_args.args[1].equals(series.iloc[-1:])
        assert scatter_args.kwargs == {
            "s": 28,
            "color": "#2563eb",
            "linewidths": 0,
            "zorder": 3,
        }
        ax.set_title.assert_called_once_with("AAPL historical P/S, last 16 quarters")
        ax.set_ylabel.assert_called_once_with("P/S")
        ax.spines[["top", "right"]].set_visible.assert_called_once_with(False)

    def test_skips_drawing_an_empty_series_but_applies_axis_labels(self):
        ax = MagicMock()
        series = pd.Series(dtype=float)

        with (
            patch.object(main, "add_percentile_bands") as add_percentile_bands,
            patch.object(main, "add_xps_date_lines") as add_xps_date_lines,
        ):
            main.plot_valuation_series(ax, series, "AAPL", "P/E", 16, "#dc2626")

        add_percentile_bands.assert_not_called()
        add_xps_date_lines.assert_not_called()
        ax.plot.assert_not_called()
        ax.scatter.assert_not_called()
        ax.set_title.assert_called_once_with("AAPL historical P/E, last 16 quarters")
        ax.set_ylabel.assert_called_once_with("P/E")
        ax.spines[["top", "right"]].set_visible.assert_called_once_with(False)


class TestPxRoute:
    def test_crypto_px_uses_the_token_terminal_valuation_path(self):
        px = pd.DataFrame(
            {"ps": [12.0], "pe": [34.0]}, index=pd.to_datetime(["2026-01-01"])
        )

        def close_and_return(fig, plt):
            plt.close(fig)
            return "response"

        with (
            patch.object(valuation, "calc_px", return_value=px) as calc_px,
            patch.object(main, "figure_to_png_response", side_effect=close_and_return),
        ):
            response = asyncio.run(main.get_px("c", "UNI", 16))

        assert response == "response"
        calc_px.assert_awaited_once_with("c", "UNI", 16)

    def test_stock_px_passes_an_unavailable_ps_series_as_empty_to_the_plotter(self):
        px = pd.DataFrame(
            {
                "ps": [None, None],
                "pe": [10.0, 11.0],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )
        ps_dates = pd.DatetimeIndex([])
        pe_dates = pd.DatetimeIndex(pd.to_datetime(["2025-12-31", "2026-01-02"]))
        px.attrs["ps_dates"] = ps_dates
        px.attrs["pe_dates"] = pe_dates

        def close_and_return(fig, plt):
            plt.close(fig)
            return "response"

        with (
            patch.object(valuation, "calc_px", return_value=px),
            patch.object(main, "plot_valuation_series") as plot,
            patch.object(main, "figure_to_png_response", side_effect=close_and_return),
        ):
            response = asyncio.run(main.get_px("u", "A", 16))

        assert response == "response"
        ps, pe = (call.args[1] for call in plot.call_args_list)
        assert ps.empty
        assert pe.equals(px["pe"])
        assert plot.call_args_list[0].kwargs["xps_dates"].equals(ps_dates)
        assert plot.call_args_list[1].kwargs["xps_dates"].equals(pe_dates)


class TestFigureToPngResponse:
    def test_returns_a_png_response_and_closes_the_figure(self):
        fig = Mock()
        plt = Mock()

        def savefig(buf, format):
            assert format == "png"
            buf.write(b"png")

        fig.savefig.side_effect = savefig

        response = main.figure_to_png_response(fig, plt)

        assert response.media_type == "image/png"
        assert response.body == b"png"
        fig.autofmt_xdate.assert_called_once_with()
        fig.tight_layout.assert_called_once_with()
        plt.close.assert_called_once_with(fig)

    def test_closes_the_figure_when_png_rendering_fails(self):
        fig = Mock()
        plt = Mock()
        fig.savefig.side_effect = RuntimeError("render failed")

        with pytest.raises(RuntimeError, match="render failed"):
            main.figure_to_png_response(fig, plt)

        plt.close.assert_called_once_with(fig)


class TestScoresRoute:
    def test_stock_scores_use_inverted_empirical_percentiles(self):
        prices = pd.DataFrame(
            {
                "ps": [1, 2, 4, 8, 16, 3],
                "pe": [2, 3, 4, 5, 6, 1],
            }
        )
        with patch.object(valuation, "calc_px", return_value=prices) as calc_px:
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "u", "symbol": "A", "q": 16},
            )

        assert response.status_code == 200
        assert response.json() == [0.5, 5 / 6]
        calc_px.assert_awaited_once_with("u", "A", 16)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ({"ps": [None, 1], "pe": [1, 1]}, [None, 0.0]),
            ({"ps": [1, 1], "pe": [None, 1]}, [0.0, None]),
            ({"ps": [1, 1, 1], "pe": [1, None, 1]}, [0.0, None]),
        ],
    )
    def test_stock_scores_allow_either_multiple_to_be_unavailable(
        self, values, expected
    ):
        with patch.object(valuation, "calc_px", return_value=pd.DataFrame(values)):
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "u", "symbol": "A", "q": 16},
            )

        assert response.status_code == 200
        assert response.json() == expected

    def test_bitcoin_scores_forward_the_requested_quarters(self):
        with patch.object(main, "calc_btc_score", return_value=0.5) as calc_btc_score:
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "c", "symbol": "BTC", "q": 16},
            )

        assert response.status_code == 200
        assert response.json() == [0.5, None]
        calc_btc_score.assert_awaited_once_with(16)

    def test_scores_schema_only_requires_market_symbol_and_quarters(self):
        response = TestClient(main.app).get("/openapi.json")

        assert response.status_code == 200
        parameters = response.json()["paths"]["/scores"]["get"]["parameters"]
        assert {parameter["name"] for parameter in parameters} == {"market", "symbol", "q"}
        assert all(parameter["required"] for parameter in parameters)


class TestDiagnosticsRoute:
    def test_diagnostics_returns_the_payload(self):
        payload = {
            "total_quarters": 7,
            "coverage": {"revenue": {"fmp": 7}},
            "agreement": {},
        }
        with patch.object(valuation, "get_xps_diagnostics", return_value=payload):
            client = TestClient(main.app)
            response = client.get("/diagnostics")

        assert response.status_code == 200
        assert response.json() == payload
        assert client.get("/metrics").status_code == 404


class TestCallRecorderLifecycle:
    def test_enabled_recorder_starts_and_writes_on_shutdown(self):
        recorder = Mock()

        async def run_lifespan():
            async with main.lifespan(main.app):
                pass

        with (
            patch.object(main, "CALL_RECORDER_ENABLED", True),
            patch.object(main, "ValuationCallRecorder", return_value=recorder),
        ):
            asyncio.run(run_lifespan())

        recorder.enable.assert_called_once_with()
        recorder.write.assert_called_once_with(
            Path.cwd() / "portman-valuation-call-edges.json"
        )
        recorder.disable.assert_called_once_with()
