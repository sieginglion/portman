import unittest
from unittest.mock import patch

from backend import main, valuation
from fastapi.testclient import TestClient


class DiagnosticsRouteTests(unittest.TestCase):
    def test_diagnostics_returns_the_payload(self):
        payload = {'total_quarters': 7, 'consensus_pairs': {}}
        with patch.object(valuation, 'get_xps_diagnostics', return_value=payload):
            client = TestClient(main.app)
            response = client.get('/diagnostics')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(client.get('/metrics').status_code, 404)
