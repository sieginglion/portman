import asyncio
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from backend import main, valuation
from fastapi.testclient import TestClient


class DownsideScoreTests(unittest.TestCase):
    def test_lower_quantile_is_half_the_requested_upper_quantile(self):
        series = pd.Series([1, 2, 4, 8, 16, 3])

        score = main.calc_downside_score(series, valuation_score_upper=0.8)

        self.assertEqual(score, 1)


class ScoresRouteTests(unittest.TestCase):
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

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [1.0, 1.0])
        calc_px.assert_awaited_once_with("u", "A", 16)

    def test_bitcoin_scores_forward_the_requested_upper_quantile(self):
        with patch.object(main, "calc_btc_score", return_value=1.25) as calc_btc_score:
            response = TestClient(main.app).get(
                "/scores",
                params={"market": "c", "symbol": "BTC", "q": 16, "u": 0.8},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [1.25, None])
        calc_btc_score.assert_awaited_once_with(16, 0.8)

    def test_scores_require_the_upper_quantile(self):
        response = TestClient(main.app).get(
            "/scores",
            params={"market": "c", "symbol": "BTC", "q": 16},
        )

        self.assertEqual(response.status_code, 422)


class DiagnosticsRouteTests(unittest.TestCase):
    def test_diagnostics_returns_the_payload(self):
        payload = {"total_quarters": 7, "consensus_pairs": {}}
        with patch.object(valuation, "get_xps_diagnostics", return_value=payload):
            client = TestClient(main.app)
            response = client.get("/diagnostics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(client.get("/metrics").status_code, 404)


class CallRecorderLifecycleTests(unittest.TestCase):
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
