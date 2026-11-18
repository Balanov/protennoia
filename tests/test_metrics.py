"""Unit tests for the metric implementations."""
import unittest

import numpy as np

from protennoia import metrics as M


class TestAUC(unittest.TestCase):
    def test_hand_computed(self):
        # scores [0.1, 0.4, 0.35, 0.8], labels [0, 0, 1, 1]
        # pairs: pos=2,3 neg=0,1
        # (3>0)=1, (3>1)=1, (2>0)=1, (2>1)=0  -> 3/4
        auc = M.auc_roc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
        self.assertAlmostEqual(auc, 0.75, places=10)

    def test_perfect_and_inverted(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(M.auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0, places=10)
        self.assertAlmostEqual(M.auc_roc(y, np.array([0.8, 0.9, 0.1, 0.2])), 0.0, places=10)

    def test_ties_give_half(self):
        auc = M.auc_roc(np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertAlmostEqual(auc, 0.5, places=10)

    def test_single_class_is_nan(self):
        self.assertTrue(np.isnan(M.auc_roc(np.array([1, 1]), np.array([0.2, 0.3]))))


class TestCalibration(unittest.TestCase):
    def test_perfect_calibration_is_zero(self):
        conf = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        corr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(M.expected_calibration_error(conf, corr), 0.0, places=6)

    def test_confident_but_not_perfect_is_penalised(self):
        conf = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        corr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(M.expected_calibration_error(conf, corr), 0.1, places=6)

    def test_overconfidence_is_penalised(self):
        conf = np.full(10, 0.9)
        corr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        ece = M.expected_calibration_error(conf, corr)
        self.assertAlmostEqual(ece, 0.4, places=6)

    def test_brier(self):
        conf = np.array([1.0, 0.0])
        corr = np.array([1.0, 1.0])
        self.assertAlmostEqual(M.brier_score(conf, corr), 0.5, places=10)


class TestCGS(unittest.TestCase):
    def test_ratio_and_gap(self):
        out = M.compute_cgs(0.8, 0.4)
        self.assertAlmostEqual(out["cgs"], 0.5, places=10)
        self.assertAlmostEqual(out["sg_gap"], 0.4, places=10)

    def test_not_applicable_at_chance(self):
        out = M.compute_cgs(0.12, 0.13)
        self.assertFalse(out["applicable"])
        self.assertIsNone(out["cgs"])

    def test_memorisation_is_flagged(self):
        out = M.compute_cgs(0.9, 0.5, nll_base=1.0, nll_novel=0.5)
        self.assertTrue(out["void"])


class TestCCE(unittest.TestCase):
    def test_noise_is_zero(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(400, 8))
        y = rng.integers(0, 4, size=400)
        out = M.compute_cce(z, y)
        self.assertTrue(out["applicable"])
        self.assertLess(out["cce"], 0.15)

    def test_label_determined_is_high(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 4, size=400)
        z = np.zeros((400, 8))
        z[np.arange(400), y] = 5.0
        z += rng.normal(scale=0.05, size=(400, 8))
        out = M.compute_cce(z, y)
        self.assertGreater(out["cce"], 0.8)


class TestAblationProfile(unittest.TestCase):
    def test_graceful_curve(self):
        f = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        a = np.array([1.0, 0.98, 0.96, 0.90, 0.30, 0.10])
        out = M.compute_ablation_profile(f, a, chance=0.125, delta=0.05)
        self.assertTrue(out["applicable"])
        # budget absorbed before accuracy falls more than delta below baseline
        self.assertAlmostEqual(out["R"], 0.2, places=10)
        # crossing into chance-level performance
        self.assertAlmostEqual(out["delta_c"], 0.5, places=10)
        self.assertGreater(out["kappa"], 0.0)

    def test_baseline_at_chance_is_not_applicable(self):
        f = np.array([0.0, 0.1, 0.2])
        a = np.array([0.125, 0.125, 0.125])
        out = M.compute_ablation_profile(f, a, chance=0.125)
        self.assertFalse(out["applicable"])
        self.assertIsNone(out["R"])


class TestPCT(unittest.TestCase):
    def test_identical_responses_are_floor(self):
        r = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        out = M.compute_pct([r, r, r, r])
        self.assertAlmostEqual(out["pct"], 0.0, places=10)

    def test_differentiated_responses_are_above_floor(self):
        rs = [np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
              np.array([0.05, 0.8, 0.05, 0.05, 0.05]),
              np.array([0.05, 0.05, 0.8, 0.05, 0.05])]
        out = M.compute_pct(rs)
        self.assertGreater(out["pct"], 0.5)


class TestEIB(unittest.TestCase):
    def test_formula(self):
        out = M.compute_eib(1e-6, 1000.0, 8.0)
        self.assertAlmostEqual(out["eib"], 1e-6 / (1000.0 * 8.0), places=15)

    def test_reference_is_relative(self):
        base = M.compute_eib(1e-6, 1000.0, 8.0)
        out = M.compute_eib(2e-6, 1000.0, 8.0, reference=base["eib"])
        self.assertAlmostEqual(out["eib_relative"], 2.0, places=10)

    def test_degenerate_is_not_applicable(self):
        self.assertFalse(M.compute_eib(1e-6, 0.0, 8.0)["applicable"])


class TestBootstrap(unittest.TestCase):
    def test_deterministic(self):
        vals = np.random.default_rng(0).normal(size=200)
        a = M.bootstrap_ci(vals, np.median, n_resample=200, seed=7)
        b = M.bootstrap_ci(vals, np.median, n_resample=200, seed=7)
        self.assertEqual(a, b)

    def test_interval_brackets_median(self):
        vals = np.random.default_rng(1).normal(size=300)
        lo, hi = M.bootstrap_ci(vals, np.median, n_resample=300, seed=3)
        self.assertLessEqual(lo, np.median(vals))
        self.assertGreaterEqual(hi, np.median(vals))


if __name__ == "__main__":
    unittest.main()
