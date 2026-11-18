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