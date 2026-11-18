# Protennoia Protocol — v0.1

**An operational, substrate-agnostic measurement protocol for AI cognitive candidacy.**
Seven metrics, mandatory negative controls, a fail-closed evaluation gate, a reproducible
reference implementation, and raw machine-readable output.

> **This is not a detector of consciousness, and it will never issue a certificate.**
> The protocol produces a *profile*: a vector of measurements with per-metric uncertainty,
> produced under a declared protocol version. See [Why no certificate](#why-no-certificate).

---

## Why this exists

The literature on AI and consciousness has converged on a set of **indicator frameworks**:
principled lists of properties that a system is thought to require in order to be conscious
(Butlin et al. 2023; Chandaria et al. 2026; Shiller et al. 2026; Niazi 2026). Those frameworks
specify *what to look for*. They do not specify *how to look* — which tasks, which baselines,
how many items, what confidence interval, or what observation would count as evidence that
an indicator is absent.

This repository supplies the missing layer: an executable protocol.

```
indicator framework  →  what a system would need      (theorems about the target)
Protennoia Protocol  →  how you would measure it      (the instrument)
```

## Install and run

No installation is required beyond numpy.

```bash
git clone https://github.com/Balanov/protennoia
cd protennoia
pip install numpy

python -m protennoia selftest          # run the control gate
python -m protennoia matrix --out results/matrix.json
python -m protennoia run --system compositional_model --seed 0
```

Requirements: Python ≥ 3.10, numpy. No accelerator, no network, no model weights, no API keys.
All randomness is seeded; two runs at the same seed produce identical output.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## The seven metrics

| Metric | What it measures | Control it must fall at |
|---|---|---|
| **MA** (+ ECE) | metacognitive discrimination, paired with calibration | random policy (AUC ≈ 0.5) |
| **AOG** | adaptation-on-error gain | system without a correction loop (= 0) |
| **CGS** (+ SG-gap) | compositional generalisation, perplexity-controlled | lookup table (≈ 0.13) |
| **CCE** | shared-information efficiency of the representation | uncorrelated representation (≈ 0) |
| **R, Δ_c, κ** | redundancy, collapse threshold, transition sharpness | baseline at chance (n/a) |
| **PCT** | perturbational complexity (AI port of the clinical PCI) | fixed response (floor) |
| **EIB** | energy per integrated bit (relative index) | sanity-bounded only |

Every metric has a designated negative control at which it **must** fall. The gate is
fail-closed: if a control misbehaves, no profile is issued.

```bash
$ python -m protennoia selftest
Protennoia Protocol v0.1 -- control gate
------------------------------------------------------------------------
MA (discrimination)                 | random_policy                | PASS | observed=0.5000
ECE (calibration)                   | overconfident_policy         | PASS | observed=0.5083
AOG (error-signal adaptation)       | mock_calibrated_confidence   | PASS | observed=0.0000
CGS (compositional generalisation)  | lookup_table                 | PASS | observed=0.1264
CCE (shared-information efficiency) | mock_calibrated_confidence   | PASS | observed=0.0198
R / Delta_c (redundancy)            | random_policy                | PASS | observed=None
PCT (perturbational complexity)     | fixed_response               | PASS | observed=0.0000
MA (discrimination)                 | mock_calibrated_confidence   | PASS | observed=0.9800
------------------------------------------------------------------------
GATE: PASSED
```

The last row is a **demonstration binding**, not a negative control: it asserts that a system
can be highly discriminating about its own errors while possessing no self-model at all.

## Three design commitments

**Fail-closed.** Every metric is bound to a control at which it must fall, and the gate blocks
profile issuance when it does not. This is enforced by `tests/test_controls.py`, so the property
is enforced by construction rather than promised in a discussion section.

**Vector, not score.** The protocol deliberately refuses to reduce its metrics to a scalar.
A scalar creates a gradient: once a summary number exists, it becomes an optimisation target,
and its meaning drifts toward whatever was optimised for. The profile schema rejects an
aggregate field. Comparison is done on vectors.

**No certification.** The protocol issues a profile, never a permit. A "consciousness
compliance" certificate would be an instrument whose principal use is documentary — an operator
seeking permission to deploy would want to display one, and the research literature would
become a source of regulatory credential. Instruments with that affordance are captured by the
interests they are meant to constrain. Profiles are published beside the system they
characterise, expire as the system changes, and are useful to people who want to *know*.

## Reference systems

Ten systems spanning the relevant behavioural space. Eight are controls; two are subjects.

| System | Kind | Role |
|---|---|---|
| `thermostat_pid` | control | integration-like criteria satisfied trivially; no discrimination |
| `random_policy` | control | chance behaviour, calibrated at base rate |
| `overconfident_policy` | control | 40% correct, 90% confident → calibration floor |
| `lookup_table` | control | perfect retrieval, no composition |
| `ngram_model` | control | back-off n-gram, partial transfer |
| `mlp_fixed` | control | additive representation, no interaction, no recurrence |
| `mock_calibrated_confidence` | control | perfectly discriminative about its errors, **no self-model** |
| `fixed_response` | control | identical output regardless of input |
| `compositional_model` | subject | content-structured representation |
| `recurrent_agent` | subject | carries state, can learn from an error signal |

The reference systems are analytic surrogates, not competitive models. The purpose of v0.1 is to
validate the *measurement layer*.

## Measuring real language models

`protennoia.ollama` applies the same metric code to a language model served locally by
[Ollama](https://ollama.com). It exists because a protocol validated only on surrogates validates
the instrument, not the question.

```bash
python -m protennoia.ollama \
  --models qwen3:8b gemma3-12b:latest qwen3:14b \
  --n-ma 150 --n-cgs-base 100 --n-cgs-novel 100 \
  --embed-model bge-m3:latest \
  --out results/ollama_matrix.json
```

Confidence is the model's own token log-probability over the option letters, taken from a single
forward pass with the answer constrained to one option letter and thinking mode disabled. It
cannot see the ground truth and it is not a post-hoc self-report.

Two things are worth knowing before you read such a profile. First, **most of the metrics are not
defined for a served model**: the API exposes no ablation surface, no correction loop and no
internal state, so AOG, the ablation family, PCT, EIB and CCE are reported as not applicable
rather than as zero. Second, **the CCE probe is withheld**: the only latent a serving API can
offer is an embedding of the query text produced by a *separate* host embedding model, which
characterises that embedding model rather than the subject — it saturates near 1.0 for every
model measured, including ones that answer incorrectly. The probe value is kept as an artefact
of the run and the metric is withheld from the profile: the fail-closed rule applied to our own
extension of the protocol.

The first results, on six models from 8B to 27B, are in §6.9 of the accompanying paper.

## Uncertainty on the discrimination metric

The AUC confidence interval is a **label-stratified** bootstrap: resampling is performed within
each class, and values and labels are resampled together. The pooled percentile bootstrap is
degenerate for subjects with few errors — a resample can contain too few minority-class items,
the AUC of such a resample collapses towards 0 or 1, and the interval can even fail to contain
the point estimate it is supposed to bracket. For one of the models in the live track the pooled
estimator reported `[0.08, 0.90]` around an AUC of `0.855`.

## Why the ablation index was replaced

Conventional ablation-robustness indices take the form `1 − |ΔP|/ΔA`. This is not invariant: the
value depends on the arbitrary width of the chosen ablation window, and it leaves the unit
interval as soon as degradation exceeds a few percentage points. More importantly, the
qualitative conclusion it supports can be an artefact of the *ablation family* — structured
removal of coherent units and random masking can disagree about the same model.

The protocol therefore reports three quantities under **both** families: redundancy **R** (budget
absorbed before degradation exceeds δ), collapse threshold **Δ_c**, and sharpness **κ**. The
disagreement between families is reported as a finding rather than resolved silently.

## Reproducibility

- Every profile records the protocol version, implementation version, seed, and the outcome of
  every control run in that same profile.
- Raw outputs are written as JSON; nothing in the pipeline is lossy.
- The test suite includes a gate that fails if any metric fails to fall on its control.
- Two runs at the same seed produce byte-identical profiles.

## Cite this

```bibtex
@software{balanov2026protennoia,
  author  = {Balanov, Anton N.},
  title   = {Protennoia Protocol: an operational measurement protocol for AI cognitive candidacy},
  year    = {2026},
  version = {0.1.3},
  url     = {https://github.com/Balanov/protennoia/releases/tag/v0.1.3},
  url     = {https://github.com/Balanov/protennoia}
}
```

Please cite the **versioned protocol identifier**, not the repository HEAD: measurements taken
under different protocol versions must remain distinguishable.

## Licence

MIT. See `LICENSE`.

## Citation of the underlying protocol paper

> Balanov, A. N. (2026). *Measuring AI cognitive candidacy: the Protennoia Protocol — an
> operational, substrate-agnostic measurement protocol with human baselines, mandatory negative
> controls, and a reproducible implementation.* [journal, preprint, DOI to be assigned].

Archived software: **10.5281/zenodo.24327764** (version 0.1.3, MIT). The preprint PDF and the
manuscript sources are in `paper/` of this repository.

## Protocol specification

`docs/PROTOCOL.md` in this repository is the versioned, citable specification of the protocol.
The full specification, including the human reference values and the statistical procedure, is
in `docs/PROTOCOL.md` §3–§5.
