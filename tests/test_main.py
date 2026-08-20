import asyncio
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main, valuation


class TestDownsideScore:
    def test_lower_quantile_is_half_the_requested_upper_quantile(self):
        series = pd.Series([1, 2, 4, 8, 16, 3])

        score = main.calc_downside_score(series, valuation_score_upper=0.8)

        assert score == 1


class TestBtcValuationProxy:
    async def test_uses_a_fixed_sixteen_quarter_moving_average(self):
        with patch.object(
            main, "get_btc_price_to_sma_series", return_value=pd.Series()
        ) as get_btc_price_to_sma_series:
            await main.get_btc_valuation_proxy(4)

        get_btc_price_to_sma_series.assert_awaited_once_with(91 * 4, 91 * 16)


class TestQuarterEndLine:
    def test_adds_a_line_for_each_quarter_end_within_the_series_range(self):
        ax = Mock()
        series = pd.Series([1, 2], index=pd.to_datetime(["2025-01-15", "2025-10-10"]))

        main.add_quarter_end_lines(ax, series)

        assert [call.args[0] for call in ax.axvline.call_args_list] == list(
            pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"])
        )


class TestPlotValuationSeries:
    def test_draws_a_non_empty_series_and_applies_axis_labels(self):
        ax = MagicMock()
        series = pd.Series(
            [1.0, 2.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"])
        )

        with (
            patch.object(main, "add_percentile_bands") as add_percentile_bands,
            patch.object(main, "add_quarter_end_lines") as add_quarter_end_lines,
        ):
            main.plot_valuation_series(ax, series, "AAPL", "P/S", 16, "#2563eb")

        add_percentile_bands.assert_called_once_with(ax, series)
        add_quarter_end_lines.assert_called_once_with(ax, series)
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
            patch.object(main, "add_quarter_end_lines") as add_quarter_end_lines,
        ):
            main.plot_valuation_series(ax, series, "AAPL", "P/E", 16, "#dc2626")

        add_percentile_bands.assert_not_called()
        add_quarter_end_lines.assert_not_called()
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
    def test_stock_scores_use_the_requested_upper_quantile(self):
        prices = pd.DataFrame(
            {
                "ps": [1, 2, 4, 8, 16, 3],
                "pe": [1, 2, 4, 8, 16, 3],
            }
        )
        with patch.object(valuation, "calc_px", return_value=prices) as calc_px:
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "u", "symbol": "A", "q": 16, "u": 0.8},
            )

        assert response.status_code == 200
        assert response.json() == [1.0, 1.0]
        calc_px.assert_awaited_once_with("u", "A", 16)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ({"ps": [None, 1], "pe": [1, 1]}, [None, 0.0]),
            ({"ps": [1, 1], "pe": [None, 1]}, [0.0, None]),
        ],
    )
    def test_stock_scores_allow_either_multiple_to_be_unavailable(
        self, values, expected
    ):
        with patch.object(valuation, "calc_px", return_value=pd.DataFrame(values)):
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "u", "symbol": "A", "q": 16, "u": 0.8},
            )

        assert response.status_code == 200
        assert response.json() == expected

    def test_bitcoin_scores_forward_the_requested_upper_quantile(self):
        with patch.object(main, "calc_btc_score", return_value=1.25) as calc_btc_score:
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "c", "symbol": "BTC", "q": 16, "u": 0.8},
            )

        assert response.status_code == 200
        assert response.json() == [1.25, None]
        calc_btc_score.assert_awaited_once_with(16, 0.8)

    def test_scores_require_the_upper_quantile(self):
        response = TestClient(main.app).get(
            "/scores",
            params={"market": "c", "symbol": "BTC", "q": 16},
        )

        assert response.status_code == 422


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
