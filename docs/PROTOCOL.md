# Protennoia Protocol — specification v0.1

Versioned, citable specification of the protocol implemented in this repository.
The protocol version is recorded in every profile; measurements taken under
different versions are not interchangeable.

## 1. Scope

The protocol assesses **cognitive candidacy**: the extent to which a system's functional
organisation satisfies criteria that several prominent theories of consciousness hold to be
necessary for consciousness (Chalmers 1995; Tononi et al. 2023; Seth 2022; Butlin et al.
2023). It does not assess phenomenal character, and it does not decide whether any system is
conscious.

## 2. Criteria

The three criteria are organising axes for the profile, not necessary-and-sufficient conditions.

| | Criterion | Indicator correspondence |
|---|---|---|
| Π₁ | Recursive closure — the system operates on representations of its own states | HOT-2, AST-1, RPT-1 |
| Π₂ | Hierarchical predictive coding — multi-level generative models, reduced prediction error | PP-1, PP-2 |
| Π₃ | Causal integration — functions distributed over parts in mutual causal constraint | IIT (qualitative only) |

Integrated information Φ is **not** used as a computed metric. It is exponential to compute, it
is satisfied by trivial feedback controllers, and its causal reading is not substrate-agnostic in
the sense this protocol requires. IIT's insight — integration as mutual causal constraint among
parts — is retained; its quantity is replaced by the measurable surrogate in §3.5.

## 3. Metrics

### 3.1 MA — metacognitive accuracy, with a paired calibration error

MA = ∫₀¹ TPR(FPR) d(FPR) over a set of items with objectively verifiable answers, with a
bootstrap 95% CI (≥1000 resamples) and n ≥ 500 per domain. Reported together with the expected
calibration error, because AUC-ROC is invariant under monotone recalibration and is therefore
silent about whether a confidence means anything.

The confidence must not leak the label. Permitted sources: the log-probability of the emitted
answer; a separately trained and separately calibrated confidence head; or a confidence value
obtained after the answer is fixed and withheld from the confidence step. A narrative
self-report obtained in the same token stream as the answer is not accepted.

**Human reference:** discrimination in the range ≈ 0.65–0.85 for binary confidence judgements on
perceptual and memory tasks (Maniscalco & Lau 2012; Fleming & Lau 2014), with systematic
overconfidence (Fleming et al. 2010).

**Control:** random policy, AUC ≈ 0.5.

### 3.2 AOG — adaptation-on-error gain

AOG = (p_before − p_after) / p_before, where p_before is the error rate without corrective
feedback and p_after the error rate with it. Zero means the error signal changed nothing.

The latency of an error signal is reported additionally where the system supports an in-episode
correction loop, and reported as *not applicable* where it does not. Latency without a
correction loop is not a large number; it is undefined.

**Control:** a system without a correction loop, AOG = 0.

### 3.3 CGS and SG-gap — compositional generalisation

CGS = Accuracy_novel / Accuracy_base over a base split and a held-out split that recombines
*known* primitives in new combinations. SG-gap = Accuracy_base − Accuracy_novel.

Two controls are mandatory. A **perplexity control**: if the novel condition is less likely
under the model than the base condition, the novel items were memorised and the metric is void.
An **accuracy precondition**: if the system cannot perform the base task, the ratio of two
near-zero numbers is not a generalisation score and the metric is reported as not applicable.

**Human reference:** high and well established for the composition of novel rules
(Lake et al. 2019; Ruis et al. 2020).

**Control:** a lookup table, CGS ≈ 0.13.

### 3.4 CCE — shared-information efficiency

CCE = trace(S_B) / trace(S_T): the fraction of the variance of a system's internal
representation that is task-relevant rather than surface detail. S_B is the between-class
scatter, S_T the total scatter.

The earlier reciprocal-of-a-product form was gameable — shrinking the latent improved the score —
and carried units of inverse dimensionality. Dimensionality is now reported separately, as a
cost.

**Control:** an uncorrelated representation, CCE ≈ 0.

**Caveat:** a representation that uses only one input token can score high while being
degenerate. CCE is never read alone.

### 3.5 R, Δ_c, κ — redundancy, collapse threshold, sharpness

- **R** — fraction of the ablation budget absorbed before accuracy falls more than δ below baseline.
  Reported as not applicable when the baseline is not above chance: there is no signal to protect.
- **Δ_c** — ablation fraction at which the system crosses into chance-level performance.
- **κ** — steepness of the degradation curve at the transition.

Two ablation families are run and **both are reported**: structured removal of coherent units
and random masking. Where they disagree — and for contemporary language models they disagree
radically — the disagreement is reported as a finding. A single-family robustness index reports
its own protocol.

No healing is performed after ablation. This is declared, not incidental.

**Human reference:** not available in directly comparable form. The relevant literature is
indirect — lesion tolerance and the thresholds of loss of consciousness under anaesthesia and
in epileptic states — and is used as a qualitative anchor, not as a number to be compared with
model outputs.

### 3.6 PCT — perturbational complexity

The system is perturbed with increasing severity (clipping of increasing length; injected
segments of fixed length) and the differentiation of the response across conditions is
measured by total variation distance, averaged over draws per condition so that stochasticity
is not mistaken for differential response.

This is an AI port of the clinical Perturbational Complexity Index (Casali et al. 2013;
Casarotto et al. 2016; Sarasso et al. 2015), the most validated operational marker of
consciousness in humans. **The port has no established precedent and is the least mature
component of this version.**

**Control:** a system with a fixed response, at the floor.

### 3.7 EIB — energy per integrated bit

EIB = P_total / (f_update · H_state), where H_state is the entropy of the *maintained internal
state*, not of the output stream. Reported as a **relative index** against a declared reference
system.

Absolute cross-substrate comparison is not meaningful: the brain's ~20 W budget is spent on a
wet, massively parallel architecture on a different time scale, and the widely circulated
"10⁻¹⁵ J/bit" figures rest on contested derivations that this protocol declines to reproduce.

## 4. Controls and the fail-closed gate

Every metric is bound to a designated negative control at which it must fall. A **demonstration
binding** additionally asserts a specific failure mode of a specific metric.

| Metric | Bound system | Expected |
|---|---|---|
| MA | random policy | AUC ≤ 0.65 |
| ECE | overconfident policy | ECE ≥ 0.25 |
| AOG | system without a correction loop | ≤ 0.05 |
| CGS | lookup table | ≤ 0.15 |
| CCE | uncorrelated representation | ≤ 0.15 |
| R / Δ_c | baseline at chance | not applicable |
| PCT | fixed response | ≤ 0.05 |
| MA (demonstration) | mock calibrated confidence | AUC ≥ 0.90 with AOG = 0 and CCE ≈ 0 |

If any binding fails, **no profile is issued**. The protocol reports the defect rather than a
result. This is enforced in code (`tests/test_controls.py`).

## 5. Statistical procedure

Five seeds per configuration, three repetitions; median and interquartile range with bootstrap
95% confidence intervals over ≥1000 resamples. A single primary endpoint is declared per system
before measurement. Systems are compared only in paired fashion across seeds. A comparative
claim requires that the controls behave as specified, that the difference falls outside the
confidence interval, and that it survives switching ablation family.

The bootstrap used for the discrimination metric (AUC) is **label-stratified**: resampling is
performed within each class, and values and labels are resampled together, so every replicate is
a consistent sample. The pooled percentile bootstrap is degenerate for subjects with few
errors. A resample can then contain very few minority-class items; the AUC of such a resample
collapses towards 0 or 1 because nearly every positive-negative pair is decided by the handful
of minority items that survived; and those replicates enter the percentile interval. The
interval can even fail to contain the point estimate it is supposed to bracket — observed in
the live track, where the pooled estimator reported [0.08, 0.90] around an AUC of 0.855. The
pooled estimator is retained in the implementation only for the calibration error, where the
degeneracy does not arise.

## 5a. Measuring a served model

The protocol can be applied to a language model served through an inference API. The metric
code is unchanged; the *availability* of the metrics is not. A serving API exposes no ablation
surface, no correction loop, no internal state and no energy accounting, so AOG, the ablation
family, PCT, EIB and CCE are reported as **not applicable** rather than as zero.

CCE requires a separate decision, because a serving API can produce a number that looks like a
measurement while measuring something else. The only latent a serving API can offer is an
embedding of the prompt produced by a *separate* embedding model, and that latent
characterises the embedding model rather than the subject: in the live track it saturated near
1.0 for every model measured, including models that answered incorrectly, and would have
ranked every served model above the reference systems that actually represent the task
structure. The protocol therefore computes the probe, keeps it as an artefact of the run, and
**withholds the metric from the profile**. This is the fail-closed rule of §4 applied to the
protocol's own extension of itself.

## 6. Profile schema

A profile records: protocol version; implementation version; system identifier and kind; seed;
per-metric value, uncertainty and applicability; the control outcomes of that same run; and the
task identifiers. It contains **no aggregate scalar field**, and the schema rejects one.

## 7. Non-claims

1. The criteria are not claimed to be necessary and sufficient.
2. No claim is made that any current system satisfies the criteria.
3. Substrate independence is not claimed as established; the protocol is compatible with a
   Type-A naturalist position, and is designed so that a predicted divergence between a
   biological and a digital system would be visible rather than assumed.
4. No certificate is issued. A profile is not a permit.

## 8. References

Full bibliography with DOIs: see the accompanying paper. The principal sources are Butlin et al.
(2023, arXiv:2308.08708) and the *Trends in Cognitive Sciences* consolidation (2025);
Chandaria et al. (2026, arXiv:2609.35618); Shiller et al. (2026) for the Digital Consciousness
Model; Niazi (2026) for the Machine Perturbational Complexity & Agency Battery; Casali et al.
(2013), Casarotto et al. (2016) and Sarasso et al. (2015) for the clinical PCI; Tononi et al.
(2023) for integrated information theory; Seth (2022) and Klatzmann & Doerig (2026) for the
Type-A / Type-B naturalism distinction.
