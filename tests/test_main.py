import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend import main, valuation
from fastapi.testclient import TestClient


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
