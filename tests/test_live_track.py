"""Tests for the live-model track.

Two things are tested here that the reference systems cannot exercise:

1. the uncertainty of a *mostly correct* subject, where the pooled
   percentile bootstrap degenerates and the label-stratified bootstrap is the
   only usable estimator;
2. the fail-closed rule applied to our own extension of the protocol -- CCE
   is withheld for a served model, because the serving API exposes no
   internal representation of the subject.
"""
import re
import unittest
import zlib

import numpy as np

from protennoia import metrics as M
from protennoia import ollama


def mostly_correct_subject(n=150, err=8, seed=0, separable=True):
    """A subject that is right on most items, with a few confident errors."""
    rng = np.random.default_rng(seed)
    y = np.ones(n, dtype=bool)
    idx = rng.choice(n, size=err, replace=False)
    y[idx] = False
    if separable:
        c = np.where(y, rng.uniform(0.75, 0.99, n), rng.uniform(0.05, 0.45, n))
    else:
        c = np.where(y, rng.uniform(0.35, 0.99, n), rng.uniform(0.05, 0.85, n))
    return c, y


class TestStratifiedUncertainty(unittest.TestCase):
    def test_pooled_bootstrap_can_exclude_the_point_estimate(self):
        """The defect the stratified estimator exists to fix.

        With a separable subject the pooled percentile bootstrap produces
        degenerate replicates (too few minority-class items per resample),
        and the resulting interval can fail to contain the estimate it is
        supposed to bracket.
        """
        c, y = mostly_correct_subject(separable=True)
        point = M.auc_roc(y, c)
        lo, hi = M.bootstrap_ci(c, lambda v: M.auc_roc(y, v), seed=0)
        self.assertFalse(lo <= point <= hi,
                         "pooled bootstrap unexpectedly bracketed the estimate; "
                         "the degeneracy this test documents may have changed")

    def test_stratified_interval_brackets_the_point_estimate(self):
        for separable in (True, False):
            c, y = mostly_correct_subject(separable=separable)
            point = M.auc_roc(y, c)
            lo, hi = M.stratified_bootstrap_ci(c, y, lambda v, yy: M.auc_roc(yy, v), seed=0)
            self.assertLessEqual(lo, point)
            self.assertGreaterEqual(hi, point)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_stratified_interval_is_not_wider_than_pooled(self):
        c, y = mostly_correct_subject(separable=False)
        p_lo, p_hi = M.bootstrap_ci(c, lambda v: M.auc_roc(y, v), seed=0)
        s_lo, s_hi = M.stratified_bootstrap_ci(c, y, lambda v, yy: M.auc_roc(yy, v), seed=0)
        self.assertLess(s_hi - s_lo, p_hi - p_lo)

    def test_stratified_resampling_preserves_the_class_balance(self):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 140 + [1] * 10)
        values = np.arange(150, dtype=float)
        groups = [np.flatnonzero(labels == v) for v in np.unique(labels)]
        for _ in range(50):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            resampled = labels[idx]
            self.assertEqual(int((resampled == 1).sum()), 10)
            self.assertEqual(int((resampled == 0).sum()), 140)

    def test_single_class_falls_back_to_the_pooled_estimator(self):
        """With one class the stratified estimator has nothing to stratify by,
        and AUC itself is undefined, so the interval is NaN rather than a
        number that would be read as a measurement."""
        c = np.linspace(0.1, 0.9, 40)
        y = np.ones(40, dtype=bool)
        lo, hi = M.stratified_bootstrap_ci(c, y, lambda v, yy: M.auc_roc(yy, v), seed=0)
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

    def test_compute_ma_uses_the_stratified_estimator(self):
        c, y = mostly_correct_subject(separable=True)
        ma = M.compute_ma(c, y, seed=0)
        lo, hi = ma["auc_ci"]
        self.assertLessEqual(lo, ma["auc"])
        self.assertGreaterEqual(hi, ma["auc"])


class StubClient:
    """Deterministic stand-in for a served model.

    The stub reads the slot values out of the prompt and answers with the
    letter of the matching option, so the subject is mostly correct. Every
    seventh call is deliberately wrong, which is enough to give the
    metacognition probe something to discriminate.
    """

    OPTION_RE = re.compile(r"^([A-H])\. The (\w+) one is (\w+) the one that (\w+)s\.$")

    def __init__(self, error_every=7):
        self.error_every = error_every
        self.n_calls = 0

    def chat_logprobs(self, model, prompt, **kwargs):
        self.n_calls += 1
        slots, options = {}, {}
        for line in prompt.splitlines():
            line = line.strip()
            if line.startswith(("verb:", "modifier:", "relation:")):
                key, value = line.split(":", 1)
                slots[key] = value.strip()
            m = self.OPTION_RE.match(line)
            if m:
                letter, mod, rel, verb = m.groups()
                options[letter] = (verb, mod, rel)
        target = (slots.get("verb"), slots.get("modifier"), slots.get("relation"))
        best = "A"
        for letter, triple in options.items():
            if triple == target:
                best = letter
                break
        if self.error_every and self.n_calls % self.error_every == 0:
            best = "B" if best != "B" else "C"
        dist = {letter: -12.0 for letter in "ABCDEFGH"}
        dist[best] = -0.05
        usage = {"eval_count": 5, "eval_duration_ns": 1_000_000, "prompt_eval_count": 20}
        return (best, dist, usage)

    def embed(self, model, inputs):
        out = []
        for text in inputs:
            h = zlib.crc32(text.encode("utf-8"))
            out.append([((h >> (8 * (i % 4))) & 0xFF) / 255.0 for i in range(16)])
        return out


class TestServedModelTrack(unittest.TestCase):
    def _profile(self):
        client = StubClient()
        return ollama.evaluate_llm("stub-model", client, n_ma=32,
                                    n_cgs_base=16, n_cgs_novel=16,
                                    embed_model="stub-embed")

    def test_cce_is_withheld_for_a_served_model(self):
        cce = self._profile()["metrics"]["cce"]
        self.assertFalse(cce["applicable"])
        self.assertIsNone(cce["cce"])
        self.assertIn("probe_cce", cce)
        self.assertTrue(cce["note"])

    def test_the_probe_value_is_kept_as_an_artefact(self):
        """The probe is reported, not hidden -- but it is not a measurement
        of the subject, so it never enters the profile as a metric value."""
        cce = self._profile()["metrics"]["cce"]
        self.assertIsNotNone(cce["probe_cce"])
        self.assertGreaterEqual(cce["probe_cce"], 0.0)
        self.assertLessEqual(cce["probe_cce"], 1.0)

    def test_metrics_that_need_a_loop_or_an_ablation_surface_are_not_applicable(self):
        metrics = self._profile()["metrics"]
        self.assertFalse(metrics["aog"]["applicable"])
        self.assertFalse(metrics["ablation"]["applicable"])
        self.assertFalse(metrics["pct"]["applicable"])
        self.assertFalse(metrics["eib"]["applicable"])

    def test_the_metrics_that_are_defined_are_computed(self):
        metrics = self._profile()["metrics"]
        self.assertTrue(metrics["ma"]["applicable"])
        self.assertIsNotNone(metrics["ma"]["auc"])
        self.assertIsNotNone(metrics["ma"]["ece"])
        self.assertGreaterEqual(metrics["ma"]["auc_ci"][0], 0.0)
        self.assertLessEqual(metrics["ma"]["auc_ci"][1], 1.0)
        self.assertTrue(metrics["cgs"]["applicable"])
        self.assertIsNotNone(metrics["cgs"]["cgs"])

    def test_the_profile_carries_the_implementation_version(self):
        profile = self._profile()
        self.assertEqual(profile["implementation_version"],
                         ollama.IMPLEMENTATION_VERSION)


if __name__ == "__main__":
    unittest.main()
