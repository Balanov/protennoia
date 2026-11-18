"""Reference-system behaviour and determinism."""
import unittest

import numpy as np

from protennoia import protocol as P
from protennoia.systems import build_system, make_instances, system_names


class TestSystems(unittest.TestCase):
    def test_all_systems_evaluate(self):
        for name in system_names():
            prof = P.evaluate_system(name, seed=0)
            self.assertIn("metrics", prof)
            self.assertEqual(prof["system"], name)

    def test_determinism_same_seed(self):
        a = P.evaluate_system("compositional_model", seed=3)
        b = P.evaluate_system("compositional_model", seed=3)
        self.assertEqual(a["metrics"]["ma"]["auc"], b["metrics"]["ma"]["auc"])
        self.assertEqual(a["metrics"]["cgs"]["acc_base"], b["metrics"]["cgs"]["acc_base"])
        self.assertEqual(a["metrics"]["cgs"]["acc_novel"], b["metrics"]["cgs"]["acc_novel"])
        self.assertEqual(a["metrics"]["pct"]["pct"], b["metrics"]["pct"]["pct"])

    def test_different_seeds_differ(self):
        a = build_system("random_policy", seed=0)
        b = build_system("random_policy", seed=1)
        item = {"verb": "jump", "mod": "red", "rel": "above",
                "distractor": "d0", "label": 0, "canonical_label": 0}
        self.assertNotEqual(a.answer(item).tolist(), b.answer(item).tolist())

    def test_task_split_is_disjoint(self):
        rng = np.random.default_rng(0)
        base, novel = make_instances(rng)
        base_keys = {(i["verb"], i["mod"], i["rel"]) for i in base}
        novel_keys = {(i["verb"], i["mod"], i["rel"]) for i in novel}
        self.assertEqual(base_keys & novel_keys, set())
        self.assertEqual(len(base_keys), 18)
        self.assertEqual(len(novel_keys), 9)

    def test_memoriser_keeps_base_and_loses_novel(self):
        prof = P.evaluate_system("lookup_table", seed=0)
        self.assertAlmostEqual(prof["metrics"]["cgs"]["acc_base"], 1.0, places=6)
        self.assertLess(prof["metrics"]["cgs"]["acc_novel"], 0.3)

    def test_mock_is_confident_without_self_model(self):
        prof = P.evaluate_system("mock_calibrated_confidence", seed=0)
        self.assertGreater(prof["metrics"]["ma"]["auc"], 0.85)
        self.assertLess(prof["metrics"]["aog"]["aog"], 0.05)
        self.assertLess(prof["metrics"]["cce"]["cce"], 0.15)

    def test_only_the_agent_can_learn_from_a_signal(self):
        for name in system_names():
            s = build_system(name, seed=0)
            if name == "recurrent_agent":
                self.assertTrue(s.supports_correction)
            else:
                self.assertFalse(s.supports_correction)

    def test_agent_learns_from_the_error_signal(self):
        prof = P.evaluate_system("recurrent_agent", seed=0)
        self.assertGreater(prof["metrics"]["aog"]["aog"], 0.0)

    def test_fixed_response_has_no_perturbational_complexity(self):
        prof = P.evaluate_system("fixed_response", seed=0)
        self.assertAlmostEqual(prof["metrics"]["pct"]["pct"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
