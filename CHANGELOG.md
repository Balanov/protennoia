# Changelog

All notable changes to the protocol are recorded here. The protocol version is
part of every profile; measurements taken under different versions are not
interchangeable.

## [0.1.0] - 2026-11-18

Initial release.

### Added
- Seven metrics: MA (+ECE), AOG, CGS (+SG-gap), CCE, R/Delta_c/kappa, PCT, EIB.
- Control registry: seven negative controls plus one demonstration binding.
- Fail-closed evaluation gate; no profile is issued when a control misbehaves.
- Ten reference systems: eight controls, two subjects.
- Profile schema with protocol version, implementation version, seed, per-metric
  uncertainty, and the control outcomes of the same run. No aggregate field.
- Deterministic reference implementation (numpy + stdlib only).

### Design decisions taken in this version
- **ESL replaced by AOG.** Error-signal latency is undefined for systems with no
  in-episode correction loop, which is most static inference-time systems. AOG
  measures the effect of the error signal and is defined for every system.
- **ARI replaced by (R, Delta_c, kappa).** The conventional scalar index is not
  invariant to the width of the chosen ablation window and leaves the unit
  interval after a few points of degradation. Both ablation families are run and
  reported; their disagreement is a finding, not a defect.
- **CCE redefined** as the between-class / total variance ratio of the
  representation, estimated against the task label. The earlier reciprocal-of-a-
  product form was gameable: shrinking the latent improved the score.
- **PCT measured with total variation**, not cosine, and averaged over draws per
  condition, so that stochasticity is not mistaken for differential response.
- **No certification.** The protocol issues a profile and declines to be usable
  as a permit.

### Known limitations
- The reference systems are analytic surrogates, not competitive models. The
  protocol validates the measurement layer; it does not rank models.
- The perturbational-complexity port has no established precedent and is the
  least mature component.
- Human reference values are incomplete for R and Delta_c, and are reported as
  ranges with task qualifiers rather than as points.
- Energy comparison is relative to a reference system; absolute cross-substrate
  comparison is not meaningful.

## [0.1.3] - 2026-11-18

Adds the live-model track (measured language models under the same protocol as
the reference systems) and fixes two defects that the live track exposed.

### Added
- `protennoia.ollama`: an adapter that measures a served LLM under the same
  metric code as the reference systems. Confidence is the model's own token
  log-probability over the option letters; the answer is constrained to a
  single option letter and thinking mode is disabled.
- Profile of six locally served models (8B-27B, instruction-tuned and base),
  with per-metric uncertainty and throughput.
- 11 new tests (57 in total), including an offline stub of a served model so
  that the live track is covered without a running inference server.

### Fixed
- **AUC uncertainty for mostly correct subjects.** The pooled percentile
  bootstrap degenerates when a subject has few minority-class items: a
  resample can contain too few of them, the AUC of such a resample collapses
  towards 0 or 1, and those replicates enter the percentile interval. The
  interval can even fail to contain the point estimate it is meant to bracket.
  The estimator is now a label-stratified bootstrap that resamples within each
  class and resamples values and labels together. For a perfectly separable
  subject the old estimator reported [0.28, 0.70] around an AUC of 1.00; the
  new one reports [1.00, 1.00]. This changes published AUC intervals and
  invalidates the intervals in earlier versions of the results file.
- **CCE was not withheld for served models.** The serving API exposes no
  internal representation of the subject, so the probe that stood in for it
  measured a separate host embedding model: it saturated near 1.0 for every
  model, including ones that answered incorrectly, and would have ranked a
  trivial answerer above a structured one. CCE is now computed as a probe,
  reported as an artefact, and withheld from the profile -- the fail-closed
  rule applied to our own extension of the protocol. Two of the seven metrics
  (MA with ECE, and CGS) remain well defined for a served model.
- Version strings in the Ollama adapter were hard-coded and disagreed with the
  package version; they now come from the package constants.

### Known limitations
- The live track measures local models on one host with one embedding model;
  results are not a model ranking and do not transfer to other hardware.
- CGS is at ceiling for the models measured (accuracy 0.93-0.98 on the base
  split), so the task suite does not discriminate among them. A harder
  compositional split is required before the metric can separate strong
  systems.