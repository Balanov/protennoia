"""The fail-closed gate.

This is the test that matters most in the protocol: if any metric fails to
fall on its designated negative control, the measurement instrument is broken
and the protocol must not issue a profile. The test fails in that case.
"""
import unittest

from protennoia import protocol as P


class TestControlGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = P.run_matrix(seeds=(0, 1), reps=1)

    def test_gate_passes(self):
        self.assertTrue(
            self.matrix["gate_passed"],
            msg="control gate failed: "
                + str([r for r in self.matrix["controls"]["rows"] if not r["passed"]]),
        )

    def test_every_negative_control_falls(self):
        for row in self.matrix["controls"]["rows"]:
            if row.get("kind") == "demonstration":
                continue
            self.assertTrue(row["passed"], msg=f"{row['metric']} did not fall at {row['control_system']}")

    def test_demonstration_binding_holds(self):
        rows = [r for r in self.matrix["controls"]["rows"] if r.get("kind") == "demonstration"]
        self.assertTrue(rows, msg="no demonstration binding was evaluated")
        for row in rows:
            self.assertTrue(row["passed"], msg=f"demonstration failed: {row['expectation']}")

    def test_registry_is_populated(self):
        self.assertGreaterEqual(len(P.CONTROL_BINDINGS), 6)
        self.assertGreaterEqual(len(P.DEMONSTRATION_BINDINGS), 1)

    def test_every_binding_names_a_real_system(self):
        from protennoia.systems import system_names
        names = set(system_names())
        for b in P.CONTROL_BINDINGS:
            self.assertIn(b["control"], names)
        for b in P.DEMONSTRATION_BINDINGS:
            self.assertIn(b["system"], names)

    def test_gate_failure_blocks_profile_issuance(self):
        """A broken control must block the protocol, not merely be reported."""
        original = P.CONTROL_BINDINGS[0]["check"]
        try:
            P.CONTROL_BINDINGS[0]["check"] = lambda v: False
            broken = P.run_matrix(systems=["random_policy", "lookup_table",
                                           "compositional_model"], seeds=(0,), reps=1)
            self.assertFalse(broken["gate_passed"])
            # the subject profile must be withheld, not merely flagged
            self.assertIn("compositional_model", broken["withheld_systems"])
            self.assertNotIn("compositional_model", broken["systems"])
            # control diagnostics are retained so the defect can be diagnosed
            self.assertIn("random_policy", broken["systems"])
        finally:
            P.CONTROL_BINDINGS[0]["check"] = original

    def test_gate_passing_issues_profiles(self):
        ok = P.run_matrix(systems=["random_policy", "compositional_model"],
                           seeds=(0,), reps=1)
        self.assertTrue(ok["gate_passed"])
        self.assertEqual(ok["withheld_systems"], [])
        self.assertIn("compositional_model", ok["systems"])

    def test_run_command_refuses_to_issue_a_subject_profile(self):
        import io as _io
        import os as _os
        import tempfile as _tf
        from contextlib import redirect_stderr
        original = P.CONTROL_BINDINGS[0]["check"]
        try:
            P.CONTROL_BINDINGS[0]["check"] = lambda v: False
            with _tf.TemporaryDirectory() as tmp:
                out = _os.path.join(tmp, "p.json")
                err = _io.StringIO()
                with redirect_stderr(err):
                    rc = P.main(["run", "--system", "compositional_model",
                                 "--seed", "0", "--out", out])
                self.assertEqual(rc, 2)
                with open(out, encoding="utf-8") as fh:
                    data = __import__("json").load(fh)
                self.assertFalse(data["profile_issued"])
                self.assertNotIn("metrics", data)
        finally:
            P.CONTROL_BINDINGS[0]["check"] = original


if __name__ == "__main__":
    unittest.main()
