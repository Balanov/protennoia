"""Profile structure, the vector-not-score commitment, and the CLI."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from protennoia import protocol as P


EXPECTED_METRICS = {"ma", "aog", "cgs", "cce", "ablation", "pct", "eib"}


class TestProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = P.evaluate_system("compositional_model", seed=0)

    def test_all_metrics_present(self):
        self.assertEqual(set(self.profile["metrics"].keys()), EXPECTED_METRICS)

    def test_no_aggregate_scalar(self):
        """The protocol must not reduce its vector to a score."""
        text = json.dumps(self.profile, default=float)
        for forbidden in ('"score"', '"total_score"', '"aggregate"', '"consciousness_score"'):
            self.assertNotIn(forbidden, text)

    def test_versioning_is_recorded(self):
        self.assertEqual(self.profile["protocol_version"], P.PROTOCOL_VERSION)
        self.assertIn("implementation_version", self.profile)

    def test_seed_is_recorded(self):
        self.assertEqual(self.profile["seed"], 0)

    def test_uncertainty_is_reported(self):
        pooled = P.pool([P.evaluate_system("compositional_model", seed=s) for s in (0, 1, 2)])
        for key in ("ma", "cgs", "cce", "pct"):
            entry = pooled["metrics"][key]
            primary = "auc" if key == "ma" else key
            self.assertIn("median", entry[primary])
            self.assertIn("ci", entry[primary])


class TestCLI(unittest.TestCase):
    def test_run_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "p.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P.main(["run", "--system", "lookup_table", "--seed", "0", "--out", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["system"], "lookup_table")

    def test_selftest_returns_zero_only_when_gate_passes(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = P.main(["selftest"])
        self.assertEqual(rc, 0)
        self.assertIn("GATE: PASSED", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
