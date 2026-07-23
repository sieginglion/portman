import asyncio
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from backend import main, valuation
from fastapi.testclient import TestClient


class DownsideScoreTests(unittest.TestCase):
    def test_lower_quantile_is_half_the_configured_upper_quantile(self):
        series = pd.Series([1, 2, 4, 8, 16, 3])

        with patch.object(main, "DOWNSIDE_SCORE_UPPER", 0.8):
            score = main.calc_downside_score(series)

        self.assertEqual(score, 1)


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
