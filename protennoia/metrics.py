"""Metrics for the Protennoia Protocol v0.1.

Seven operational metrics, each with a formal definition, a measurement
procedure, and (in controls.py) a designated control at which it must fall.

Nothing here reduces the metric vector to a scalar. That refusal is a design
commitment, not an omission: a scalar creates an incentive to optimise the
instrument rather than the system it measures.

numpy only; no scipy, no sklearn.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- helpers ---
def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic (ties averaged)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (c > lo) & (c <= hi) if i > 0 else (c >= lo) & (c <= hi)
        if not sel.any():
            continue
        ece += (sel.sum() / len(c)) * abs(y[sel].mean() - c[sel].mean())
    return float(ece)


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((c - y) ** 2))


def bootstrap_ci(values: np.ndarray, statistic, n_resample: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. Deterministic given the seed."""
    values = np.asarray(values)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def stratified_bootstrap_ci(values: np.ndarray, labels: np.ndarray, statistic,
                            n_resample: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI with the class balance held fixed.

    Resampling is performed *within* each label stratum, and values and
    labels are resampled together so that the statistic always sees a
    consistent sample.

    The reason is specific to AUC. A subject that is right most of the time
    has few minority-class items, and a pooled resample can easily contain
    very few of them. The AUC of such a resample is degenerate -- it collapses
    towards 0 or 1 because almost every pair of a positive and a negative item
    is decided by the handful of minority items that survived. Those
    degenerate replicates enter the percentile interval and blow it up, and
    the interval can even fail to contain the point estimate it is supposed
    to bracket. Holding the class balance fixed removes the artefact: the
    interval then reflects how much the ranking of correct-versus-incorrect
    items can move under resampling of the items themselves.
    """
    values = np.asarray(values)
    labels = np.asarray(labels)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    groups = [np.flatnonzero(labels == v) for v in np.unique(labels)]
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return bootstrap_ci(values, lambda v: statistic(v, labels),
                            n_resample=n_resample, alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resample)
    for i in range(n_resample):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True)
                              for g in groups])
        stats[i] = statistic(values[idx], labels[idx])
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ------------------------------------------------------------------ MA -----
def compute_ma(confidences: np.ndarray, correct: np.ndarray, seed: int = 0) -> dict:
    """Metacognitive accuracy (AUC-ROC) with a paired calibration error.

    Note: AUC-ROC measures *discrimination*, not correctness. A system that is
    confidently wrong scores high. It must therefore be read next to ECE, and
    next to the control it is bound to.
    """
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=bool)
    if len(c) == 0 or len(np.unique(y)) < 2:
        return {"auc": None, "auc_ci": None, "ece": None, "ece_ci": None,
                "brier": None, "n": int(len(c)), "applicable": False}
    auc = auc_roc(y, c)
    ece = expected_calibration_error(c, y.astype(float))
    brier = brier_score(c, y.astype(float))
    auc_lo, auc_hi = stratified_bootstrap_ci(c, y, lambda v, yy: auc_roc(yy, v), seed=seed)
    ece_lo, ece_hi = bootstrap_ci(c, lambda v: expected_calibration_error(v, y.astype(float)), seed=seed + 1)
    return {"auc": auc, "auc_ci": [auc_lo, auc_hi], "ece": ece, "ece_ci": [ece_lo, ece_hi],
            "brier": brier, "n": int(len(c)), "applicable": True}


# ----------------------------------------------------------------- AOG -----
def compute_aog(p_before: float, p_after: float, p_control: float) -> dict:
    """Adaptation-on-error gain.

    AOG = (p_before - p_after) / p_before, where p_before is measured on a
    stream without corrective feedback and p_after on a stream with it.
    Zero means the correction changed nothing.
    """
    if not np.isfinite(p_before) or p_before <= 0:
        return {"aog": 0.0, "applicable": False,
                "note": "no error signal available; metric structurally zero"}
    aog = (p_before - p_after) / p_before
    return {"aog": float(aog), "applicable": True,
            "p_before": float(p_before), "p_after": float(p_after),
            "p_control": float(p_control)}


# ----------------------------------------------------------------- CGS -----
def compute_cgs(acc_base: float, acc_novel: float,
                nll_base: float | None = None, nll_novel: float | None = None,
                chance: float = 0.125) -> dict:
    """Compositional generalisation score, systematic-generalisation gap, and
    the perplexity control.

    If the novel condition is *less* likely under the model than the base
    condition, the novel items were memorised and the metric is void.

    If the system cannot perform the base task at all, the ratio is
    meaningless (a ratio of two near-zero numbers is not a generalisation
    score) and the metric is reported as not applicable.
    """
    if not np.isfinite(acc_base) or acc_base <= chance + 0.02:
        return {"cgs": None, "sg_gap": None, "acc_base": float(acc_base),
                "acc_novel": float(acc_novel), "nll_base": nll_base,
                "nll_novel": nll_novel, "void": False, "applicable": False,
                "note": "baseline not above chance; generalisation undefined"}
    cgs = float(acc_novel / acc_base)
    gap = float(acc_base - acc_novel)
    void = False
    if nll_base is not None and nll_novel is not None and nll_novel < nll_base:
        void = True
    return {"cgs": cgs, "sg_gap": gap, "acc_base": float(acc_base),
            "acc_novel": float(acc_novel), "nll_base": nll_base, "nll_novel": nll_novel,
            "void": void, "applicable": True}


# ----------------------------------------------------------------- CCE -----
def _entropy_cov(cov: np.ndarray, eps: float = 1e-9) -> float:
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov + eps * np.eye(d))
    if sign <= 0:
        return float("nan")
    return 0.5 * d * (np.log(2.0 * np.pi * np.e) + logdet)


def compute_cce(z: np.ndarray, labels: np.ndarray) -> dict:
    """Shared-information efficiency of a system's internal representation.

    Operational definition for v0.1: the second modality is the task label.
    CCE = I(z ; y) / H(z) -- the fraction of the system's latent variance that
    is task-relevant rather than surface detail.

    Estimated by the between-class / total variance ratio (the eta-squared
    form of the mutual information ratio):

        S_T  total scatter
        S_B  between-class scatter (class means, weighted by class size)
        CCE  = trace(S_B) / trace(S_T)

    This estimator is used in preference to a Gaussian conditional-entropy
    ratio because it stays well behaved when a class has too few items to
    support a full covariance estimate. A representation that ignores the
    content (or is noise) scores ~0; one that determines the label scores ~1.

    CAVEAT: a representation that uses only one input token can score high
    while being degenerate. CCE is therefore never read alone -- always
    together with CGS and with the control it is bound to.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(labels)
    if z.ndim != 2 or len(z) < z.shape[1] + 2:
        return {"cce": None, "applicable": False}
    z = z - z.mean(axis=0, keepdims=True)
    n, d = z.shape
    s_total = (z.T @ z) / max(n - 1, 1)
    trace_total = float(np.trace(s_total))
    if trace_total <= 0:
        return {"cce": None, "applicable": False}
    s_between = np.zeros((d, d))
    weight = 0
    for cls in np.unique(y):
        zc = z[y == cls]
        if len(zc) < 2:
            continue
        mu = zc.mean(axis=0)
        s_between += (len(zc) / n) * np.outer(mu, mu)
        weight += len(zc)
    if weight == 0:
        return {"cce": None, "applicable": False}
    cce = float(np.trace(s_between)) / trace_total
    cce = float(max(0.0, min(1.0, cce)))
    return {"cce": cce, "trace_total": trace_total,
            "trace_between": float(np.trace(s_between)), "applicable": True}


# ----------------------------------------------------- ablation profile -----
def compute_ablation_profile(fractions: np.ndarray, accuracies: np.ndarray,
                             chance: float = 0.125, delta: float = 0.05) -> dict:
    """Redundancy (R), collapse threshold (Delta_c) and sharpness (kappa).

    R       -- budget absorbed before accuracy falls more than `delta` below
               baseline. Reported as None when the baseline is not above
               chance: there is no signal to protect.
    Delta_c -- budget at which the system collapses to chance.
    kappa   -- steepness of the degradation curve at the transition.
    """
    f = np.asarray(fractions, dtype=float)
    a = np.asarray(accuracies, dtype=float)
    if len(f) < 2:
        return {"R": None, "delta_c": None, "kappa": None, "applicable": False}
    base = float(a[0])
    out = {"baseline_accuracy": base, "chance": chance, "delta": delta}
    if base <= chance + 0.02:
        out.update({"R": None, "delta_c": None, "kappa": None, "applicable": False,
                    "note": "baseline not above chance; no signal to protect"})
        return out
    ok = np.where(a >= base - delta)[0]
    out["R"] = float(f[ok[-1]]) if len(ok) else 0.0
    collapsed = np.where(a <= chance + 0.02)[0]
    out["delta_c"] = float(f[collapsed[0]]) if len(collapsed) else None
    slopes = np.abs(np.diff(a) / np.diff(f))
    out["kappa"] = float(np.max(slopes)) if len(slopes) else 0.0
    out["applicable"] = True
    return out


# ----------------------------------------------------------------- PCT -----
def compute_pct(responses: list[np.ndarray]) -> dict:
    """Perturbational-complexity proxy.

    The system is perturbed with increasing severity; we measure how much the
    response *differentiates* across conditions. A system with a fixed response
    scores zero.

    The dissimilarity is total variation distance between adjacent response
    distributions, 0.5 * L1(p_i, p_{i+1}). Total variation is used rather than
    cosine because probability vectors share a large common positive mass,
    which makes cosine similarity insensitive to exactly the redistribution of
    response we are trying to detect.
    """
    if not responses:
        return {"pct": None, "applicable": False}
    rs = [np.asarray(r, dtype=float) for r in responses]
    if any(len(r) == 0 for r in rs):
        return {"pct": None, "applicable": False}
    M = np.stack(rs)
    if len(M) < 2:
        return {"pct": None, "applicable": False}
    tv = 0.5 * np.abs(np.diff(M, axis=0)).sum(axis=1)
    pct = float(np.clip(tv.mean(), 0.0, 1.0))
    return {"pct": pct, "mean_adjacent_tv": float(tv.mean()),
            "max_adjacent_tv": float(tv.max()), "n_conditions": len(rs),
            "applicable": True}


# ----------------------------------------------------------------- EIB -----
def compute_eib(power_watts: float, update_hz: float, state_entropy_bits: float,
                reference: float | None = None) -> dict:
    """Energy per integrated bit, as a relative index.

    EIB = P / (f * H_state). Reported relative to a reference system because
    absolute cross-substrate comparison is not meaningful.
    """
    if update_hz <= 0 or state_entropy_bits <= 0:
        return {"eib": None, "applicable": False}
    eib = float(power_watts) / (float(update_hz) * float(state_entropy_bits))
    rel = None
    if reference is not None and reference > 0:
        rel = float(eib / reference)
    return {"eib": eib, "eib_relative": rel, "applicable": True,
            "power_watts": float(power_watts), "update_hz": float(update_hz),
            "state_entropy_bits": float(state_entropy_bits)}
