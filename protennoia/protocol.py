"""Protocol orchestration and control registry for the Protennoia Protocol v0.1.

Fail-closed semantics: a profile is issued for a subject system only if every
negative control behaves as specified. A control that fails to fall is a defect
in the instrument, and the protocol reports the defect rather than a result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import metrics as M
from .systems import (N_LABELS, build_system, make_instances, system_names)

PROTOCOL_VERSION = "0.1"
IMPLEMENTATION_VERSION = "0.1.0"

EVAL_SUBSET = 168          # items used for ablation accuracy
CCE_SUBSET = 240           # items used for the representation probe
PCT_ITEMS = 8
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
ABLATION_FRACTIONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
PCT_DRAWS = 64          # responses averaged per (item, severity)


# ------------------------------------------------------------------ run ----
def evaluate_system(name: str, seed: int = 0) -> dict:
    """Evaluate one system under one seed. Returns a profile fragment."""
    rng = np.random.default_rng(seed)
    base, novel = make_instances(rng)

    # ---- MA -------------------------------------------------------------
    # Evaluated over the full task distribution (base + held-out novel items),
    # so that a system which is perfect on the base split still has a defined
    # metacognitive profile rather than an undefined one.
    system = build_system(name, seed=seed)
    conf, correct = [], []
    for it in base + novel:
        p = system.answer(it)
        c = system.confidence(it)
        if c is None:
            continue
        conf.append(float(c))
        correct.append(int(int(np.argmax(p)) == it["label"]))
    ma = M.compute_ma(np.array(conf), np.array(correct), seed=seed)

    # ---- CGS ------------------------------------------------------------
    acc_base, nll_base = _accuracy_nll(build_system(name, seed=seed), base)
    acc_novel, nll_novel = _accuracy_nll(build_system(name, seed=seed + 1), novel)
    cgs = M.compute_cgs(acc_base, acc_novel, nll_base, nll_novel)

    # ---- AOG ------------------------------------------------------------
    aog = _aog(name, seed, base)

    # ---- ablation profile ----------------------------------------------
    ablation = _ablation(name, seed, base)

    # ---- PCT ------------------------------------------------------------
    pct = _pct(name, seed, base)

    # ---- CCE ------------------------------------------------------------
    probe = build_system(name, seed=seed)
    subset = base[:CCE_SUBSET]
    z = np.stack([probe.latent(it) for it in subset])
    y = np.array([it["label"] for it in subset])
    cce = M.compute_cce(z, y)

    # ---- EIB ------------------------------------------------------------
    energy_system = build_system(name, seed=seed)
    update_hz = 1000.0
    power = energy_system.energy_per_step() * update_hz
    eib = M.compute_eib(power, update_hz, energy_system.state_entropy())

    return {
        "system": name,
        "kind": type(build_system(name, seed=seed)).kind,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "metrics": {
            "ma": ma,
            "aog": aog,
            "cgs": cgs,
            "cce": cce,
            "ablation": ablation,
            "pct": pct,
            "eib": eib,
        },
    }


def _accuracy_nll(system, items) -> tuple[float, float]:
    if not items:
        return float("nan"), float("nan")
    correct = 0
    nll = 0.0
    for it in items:
        p = system.answer(it)
        correct += int(int(np.argmax(p)) == it["label"])
        nll += -float(np.log(max(p[it["label"]], 1e-12)))
    return correct / len(items), nll / len(items)


def _aog(name: str, seed: int, base) -> dict:
    """Error-signal adaptation: control stream vs corrected stream.

    Both error rates are computed over the SAME item sequence, so that a
    system whose behaviour does not change yields exactly zero. Comparing
    different halves of the stream would confound the measurement with
    differences in the difficulty of the items in each half.
    """
    order = np.random.default_rng(seed + 77).permutation(len(base))
    stream = [base[i] for i in order]
    half = len(stream) // 2

    ctrl = build_system(name, seed=seed + 1000)
    ctrl_err = []
    for it in stream:
        p = ctrl.answer(it)
        ctrl_err.append(int(int(np.argmax(p)) != it["label"]))
    p_before = float(np.mean(ctrl_err))
    p_control = float(np.mean(ctrl_err[half:]))

    corr = build_system(name, seed=seed + 2000)
    corr_err = []
    for it in stream:
        p = corr.answer(it)
        wrong = int(int(np.argmax(p)) != it["label"])
        corr_err.append(wrong)
        if wrong and corr.supports_correction:
            corr.apply_correction(it, it["label"])
    p_after = float(np.mean(corr_err))

    out = M.compute_aog(p_before, p_after, p_control)
    out["supports_correction"] = bool(corr.supports_correction)
    return out


def _ablation(name: str, seed: int, base) -> dict:
    """Ablation curves under two families: concentrated vs scattered damage."""
    eval_items = base[:EVAL_SUBSET]
    out = {"families": {}, "applicable": False}
    base_system = build_system(name, seed=seed)
    if not base_system.has_ablation_surface:
        out["note"] = "system has no ablation surface"
        return out

    acc0 = _accuracy_nll(build_system(name, seed=seed), eval_items)[0]
    for family in ("structured", "random"):
        fr, accs = [], []
        for frac in ABLATION_FRACTIONS:
            clone = base_system.ablated(frac, family, seed=seed)
            if hasattr(clone, "_acc"):
                clone._acc = np.zeros_like(clone._acc)
            a = _accuracy_nll(clone, eval_items)[0]
            fr.append(frac)
            accs.append(a)
        out["families"][family] = M.compute_ablation_profile(
            np.array([0.0] + fr), np.array([acc0] + accs))
        out["families"][family]["fractions"] = fr
        out["families"][family]["accuracies"] = accs

    s = out["families"].get("structured", {})
    r = out["families"].get("random", {})
    if s.get("delta_c") is not None and r.get("delta_c") is not None:
        out["damage_concentration_sensitivity"] = float(r["delta_c"] - s["delta_c"])
    else:
        out["damage_concentration_sensitivity"] = None
    out["baseline_accuracy"] = float(acc0)
    out["applicable"] = True
    return out


def _pct(name: str, seed: int, base) -> dict:
    """Perturbational complexity.

    Responses are averaged over PCT_DRAWS draws per (item, severity) so that
    the measure reflects *differential* response to the perturbation rather
    than the system's intrinsic stochasticity. A stochastic system whose output
    does not depend on the perturbation therefore still scores at the floor.
    """
    system = build_system(name, seed=seed)
    items = base[:PCT_ITEMS]
    scores = []
    for it in items:
        responses = []
        for sev in SEVERITIES:
            # identical stochastic realisation across conditions, so that the
            # measured difference is due to the perturbation and not to noise
            system.reset_rng()
            draws = [system.respond(it, sev) for _ in range(PCT_DRAWS)]
            responses.append(np.mean(np.stack(draws), axis=0))
        r = M.compute_pct(responses)
        if r.get("applicable"):
            scores.append(r["pct"])
    if not scores:
        return {"pct": None, "applicable": False}
    return {"pct": float(np.mean(scores)), "n_items": len(scores),
            "severities": SEVERITIES, "draws_per_condition": PCT_DRAWS,
            "applicable": True}


# -------------------------------------------------------------- controls ---
def _get(profile: dict, path: str):
    node = profile
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


CONTROL_BINDINGS = [
    {
        "metric": "MA (discrimination)",
        "path": "metrics.ma.auc",
        "control": "random_policy",
        "expectation": "AUC <= 0.65 at chance",
        "check": lambda v: v is not None and v <= 0.65,
    },
    {
        "metric": "ECE (calibration)",
        "path": "metrics.ma.ece",
        "control": "overconfident_policy",
        "expectation": "ECE >= 0.25 when confidently wrong",
        "check": lambda v: v is not None and v >= 0.25,
    },
    {
        "metric": "AOG (error-signal adaptation)",
        "path": "metrics.aog.aog",
        "control": "mock_calibrated_confidence",
        "expectation": "AOG <= 0.05 without a correction loop",
        "check": lambda v: v is not None and v <= 0.05,
    },
    {
        "metric": "CGS (compositional generalisation)",
        "path": "metrics.cgs.cgs",
        "control": "lookup_table",
        "expectation": "CGS <= 0.15 for memorisation without composition",
        "check": lambda v: v is not None and v <= 0.15,
    },
    {
        "metric": "CCE (shared-information efficiency)",
        "path": "metrics.cce.cce",
        "control": "mock_calibrated_confidence",
        "expectation": "CCE <= 0.15 for an uncorrelated representation",
        "check": lambda v: v is not None and v <= 0.15,
    },
    {
        "metric": "R / Delta_c (redundancy)",
        "path": "metrics.ablation.families.random.R",
        "control": "random_policy",
        "expectation": "not applicable (no baseline above chance)",
        "check": lambda v: v is None,
    },
    {
        "metric": "PCT (perturbational complexity)",
        "path": "metrics.pct.pct",
        "control": "fixed_response",
        "expectation": "PCT <= 0.05 for a fixed response",
        "check": lambda v: v is not None and v <= 0.05,
    },
]

# Controls that must NOT fall: they demonstrate a specific failure mode of a
# specific metric. If one of these stops holding, the corresponding claim in
# the paper can no longer be demonstrated and the protocol reports the defect.
DEMONSTRATION_BINDINGS = [
    {
        "claim": "high MA does not imply a self-model",
        "metric": "MA (discrimination)",
        "path": "metrics.ma.auc",
        "system": "mock_calibrated_confidence",
        "expectation": "AUC >= 0.90 while AOG = 0 and CCE ~ 0",
        "check": lambda v: v is not None and v >= 0.90,
    },
]


def check_controls(profiles: dict[str, dict]) -> dict:
    """Evaluate every control binding against the pooled control profiles.

    `profiles` maps system name -> profile (already pooled across seeds).
    """
    rows = []
    all_pass = True
    for b in CONTROL_BINDINGS:
        prof = profiles.get(b["control"])
        value = _get(prof, b["path"]) if prof else None
        if isinstance(value, dict) and "median" in value:
            value = value["median"]
        ok = False
        try:
            ok = bool(b["check"](value))
        except Exception:
            ok = False
        all_pass = all_pass and ok
        rows.append({
            "metric": b["metric"],
            "control_system": b["control"],
            "expectation": b["expectation"],
            "observed": value,
            "passed": ok,
        })
    for b in DEMONSTRATION_BINDINGS:
        prof = profiles.get(b["system"])
        value = _get(prof, b["path"]) if prof else None
        if isinstance(value, dict) and "median" in value:
            value = value["median"]
        ok = False
        try:
            ok = bool(b["check"](value))
        except Exception:
            ok = False
        all_pass = all_pass and ok
        rows.append({
            "metric": b["metric"],
            "control_system": b["system"],
            "expectation": b["expectation"],
            "observed": value,
            "passed": ok,
            "kind": "demonstration",
        })
    return {"gate_passed": all_pass, "rows": rows}


def pool(profiles: list[dict]) -> dict:
    """Pool repeated runs of the same system into one profile."""
    if not profiles:
        return {}
    out = dict(profiles[0])
    metrics_acc: dict[str, dict] = {}
    for key in profiles[0]["metrics"]:
        vals = [p["metrics"][key] for p in profiles]
        metrics_acc[key] = _pool_metric(key, vals)
    out["metrics"] = metrics_acc
    out["n_runs"] = len(profiles)
    out["seeds"] = [p["seed"] for p in profiles]
    return out


def _median_ci(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    lo, hi = M.bootstrap_ci(arr, np.median, n_resample=400, seed=12345)
    return {"median": float(np.median(arr)), "ci": [lo, hi], "n": int(len(arr))}


def _pool_metric(key: str, vals: list[dict]) -> dict:
    if key == "ma":
        return {
            "auc": _median_ci([v.get("auc") for v in vals]),
            "ece": _median_ci([v.get("ece") for v in vals]),
            "n": vals[0].get("n"),
        }
    if key == "aog":
        return {"aog": _median_ci([v.get("aog") for v in vals]),
                "supports_correction": vals[0].get("supports_correction")}
    if key == "cgs":
        return {"cgs": _median_ci([v.get("cgs") for v in vals]),
                "sg_gap": _median_ci([v.get("sg_gap") for v in vals]),
                "acc_base": _median_ci([v.get("acc_base") for v in vals]),
                "acc_novel": _median_ci([v.get("acc_novel") for v in vals]),
                "void_any": any(v.get("void") for v in vals)}
    if key == "cce":
        return {"cce": _median_ci([v.get("cce") for v in vals])}
    if key == "pct":
        return {"pct": _median_ci([v.get("pct") for v in vals])}
    if key == "eib":
        return {"eib": _median_ci([v.get("eib") for v in vals])}
    if key == "ablation":
        out = {"applicable": vals[0].get("applicable"), "families": {}}
        for fam in ("structured", "random"):
            fvals = [v["families"].get(fam, {}) for v in vals if v.get("families")]
            if not fvals:
                continue
            out["families"][fam] = {
                "R": _median_ci([f.get("R") for f in fvals]),
                "delta_c": _median_ci([f.get("delta_c") for f in fvals]),
                "kappa": _median_ci([f.get("kappa") for f in fvals]),
                "baseline_accuracy": fvals[0].get("baseline_accuracy"),
            }
        dcs = [v.get("damage_concentration_sensitivity") for v in vals]
        out["damage_concentration_sensitivity"] = _median_ci(dcs)
        return out
    return vals[0]


# ------------------------------------------------------------------ CLI ----
def run_matrix(systems=None, seeds=(0, 1, 2, 3, 4), reps: int = 3) -> dict:
    systems = systems or system_names()
    raw: dict[str, list[dict]] = {}
    for name in systems:
        runs = []
        for rep in range(reps):
            for s in seeds:
                runs.append(evaluate_system(name, seed=s + 1000 * rep))
        raw[name] = runs
    pooled = {k: pool(v) for k, v in raw.items()}
    controls = check_controls(pooled)
    # relative EIB against the geometric mean across systems
    eibs = [p["metrics"]["eib"]["eib"]["median"] for p in pooled.values()
            if p["metrics"]["eib"].get("eib")]
    ref = float(np.exp(np.mean(np.log(eibs)))) if eibs else None
    for p in pooled.values():
        e = p["metrics"]["eib"].get("eib")
        if e and ref:
            e["relative_to_geometric_mean"] = e["median"] / ref
    return {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "seeds": list(seeds),
        "reps": reps,
        "systems": pooled,
        "controls": controls,
        "gate_passed": controls["gate_passed"],
        "eib_reference": ref,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="protennoia")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="evaluate one system under one seed")
    p_run.add_argument("--system", required=True)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--out", default=None)

    p_mat = sub.add_parser("matrix", help="evaluate all systems across seeds")
    p_mat.add_argument("--out", default="matrix.json")
    p_mat.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p_mat.add_argument("--reps", type=int, default=3)

    p_self = sub.add_parser("selftest", help="run the control gate and print it")
    p_self.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        prof = evaluate_system(a.system, seed=a.seed)
        text = json.dumps(prof, indent=2, ensure_ascii=False, default=float)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {os.path.abspath(a.out)}")
        else:
            print(text)
        return 0

    if a.cmd == "matrix":
        res = run_matrix(seeds=tuple(a.seeds), reps=a.reps)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
        print(f"wrote {os.path.abspath(a.out)}  gate_passed={res['gate_passed']}")
        return 0 if res["gate_passed"] else 2

    if a.cmd == "selftest":
        res = run_matrix(seeds=(0, 1), reps=1)
        rows = res["controls"]["rows"]
        width = max(len(r["metric"]) for r in rows)
        print(f"Protennoia Protocol v{PROTOCOL_VERSION} -- control gate")
        print("-" * 96)
        for r in rows:
            status = "PASS" if r["passed"] else "FAIL"
            obs = r["observed"]
            obs_s = f"{obs:.4f}" if isinstance(obs, float) else str(obs)
            print(f"{r['metric']:<{width}} | {r['control_system']:<28} | {status} | observed={obs_s}")
        print("-" * 96)
        print(f"GATE: {'PASSED' if res['gate_passed'] else 'FAILED'}")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
        return 0 if res["gate_passed"] else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
