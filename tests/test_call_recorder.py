import json
import tempfile
import unittest
from pathlib import Path

from backend import valuation
from backend.call_recorder import ValuationCallRecorder


class ValuationCallRecorderTests(unittest.TestCase):
    def test_records_each_internal_call_edge_once(self):
        recorder = ValuationCallRecorder(valuation)
        recorder.enable()
        try:
            self.assertTrue(valuation.field_has_consensus("revenue", 1, 1))
            self.assertTrue(valuation.field_has_consensus("revenue", 2, 2))
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {("field_has_consensus", "source_log_diff")},
        )

    def test_writes_only_the_unique_edges(self):
        recorder = ValuationCallRecorder(valuation)
        recorder.edges.add(("caller", "callee"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "edges.json"
            recorder.write(output)

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                [["caller", "callee"]],
            )
