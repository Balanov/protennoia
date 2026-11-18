# Measuring AI cognitive candidacy: the Protennoia Protocol

**An operational, substrate-agnostic measurement protocol with human baselines, mandatory negative controls, and a reproducible implementation**

Anton N. Balanov
ORCID 0000-0003-1891-1147 · Peoples' Friendship University of Russia (RUDN), Moscow, Russian Federation
Corresponding author: office@balanov.pro

**Repository:** `github.com/Balanov/protennoia` · **Protocol spec (versioned, citable):** 10.5281/zenodo.24327764 · **Implementation:** `protennoia` package (numpy, stdlib-only, deterministic; v0.1.3)

---

## Abstract

Research on AI and consciousness has produced indicator frameworks: principled lists of properties a system must exhibit to count as conscious. They specify what to look for but not how to look — not the tasks, baselines, sample sizes or failure criteria — so none supports reproducible comparison. We close that gap. The Protennoia Protocol organises indicators around three criteria (recursive closure, hierarchical predictive coding, causal integration) and replaces each with an operational metric: metacognitive accuracy with a paired calibration error, adaptation-on-error gain, compositional generalisation with a perplexity control, shared-information efficiency of the internal representation, a redundancy/collapse-threshold/sharpness triple under two ablation families, a perturbational-complexity proxy ported from clinical PCI, and a relative energy index. Three commitments distinguish it. Every metric is bound to a designated negative control at which it must fall, and evaluation is fail-closed: if a control misbehaves, no profile is issued. Metrics are published as a vector, never a scalar score. The protocol issues a profile, never a certificate: a certificate of consciousness compliance would license deployment, not inform. We release a deterministic, dependency-free implementation, ten reference systems, and a test suite that fails if a metric misses its control. On the reference systems the gate passes and metacognitive discrimination inverts the intuitive ordering: the system with no self-model attains the highest discrimination of its errors. Measured on six served language models, only two of the seven metrics are defined; the protocol withholds the rest rather than filling gaps.

**Highlights**

- First executable protocol for measuring AI cognitive candidacy end to end.
- Seven metrics, each bound to a negative control at which it must fall.
- Fail-closed gate: no profile is issued if any control misbehaves.
- Vector output, never a scalar score and never a certificate.
- Deterministic numpy-only implementation with a 57-test suite.

**Keywords:** AI consciousness; measurement protocol; negative controls; metacognition; compositional generalisation; perturbational complexity; reproducibility

---

## 1. Introduction

Two claims have become commonplace in public discussion about artificial intelligence: that current systems are or are not conscious, and that the question will be settled by scaling. Neither claim has been earned. The scientific literature that bears on the first — a small, fast-growing body of work at the intersection of consciousness science and machine learning — is careful in a way that public discourse is not, and that carefulness is precisely what makes the question tractable. It is also, so far, unfinished.

The state of the art can be stated briefly. Working from the major neuroscientific theories of consciousness, Butlin et al. (Butlin et al., 2023) derived a set of *indicator properties* — computational criteria intended to capture what each theory considers necessary for consciousness — and proposed that systems satisfying more of them are more likely to be conscious. The framework was deliberately probabilistic rather than decisive: it neither certifies nor excludes. It was followed by a peer-reviewed consolidation of the same indicators (Butlin et al., 2025). In parallel, a 14-author effort extended the framework to a five-level hierarchy of functional description grounded in Marr's levels, treating indicators as evidence with likelihood ratios and reporting, for illustrative profiles, credences for a contemporary language model ranging from 0.005 to 0.793 depending on where credence is placed among the theories (Chandaria et al., 2026). The Digital Consciousness Model assembles 13 theoretical stances and more than 200 indicators, yielding a median posterior of roughly 0.08 for 2024-era language models (Shiller et al., 2026). The Machine Perturbational Complexity & Agency Battery proposes to port clinical neurophysiological assessment to artificial systems along four axes (Niazi, 2026).

These contributions are substantial, and the field should be careful about how it characterises them. They are not lists of unanswered wishes; they are the product of a difficult attempt to apply the machinery of consciousness science to computational systems without pretending that the theory has settled. What they share is a characteristic that is easy to overlook because it is a property of the *genre* rather than of any individual work: **they specify what to look for, but not how to look.** An indicator such as "metacognitive monitoring distinguishing reliable perceptual representations from noise" (Butlin et al., 2023) is a hypothesis about a mechanism, not a measurement procedure. It does not determine which tasks elicit a confidence estimate, how that estimate must be obtained so that it does not leak the label, what the human reference value is for the same task, how many items are needed, what the confidence interval is, or — most importantly — what observation would count as evidence that the indicator is *absent*. A framework of indicators without instruments leaves the central empirical question of the field exactly where it was: whether any current system satisfies the conditions that its authors believe are necessary.

This gap has a second, less visible cost. Because indicators are assessed by expert judgement rather than by protocol, results are not comparable across systems, across labs, or across time. There is no baseline table, so "system X exhibits metacognitive monitoring" and "system X exhibits metacognitive monitoring" may refer to different measurements. There are no negative controls, so a framework cannot demonstrate that its instruments are capable of failing. And there is no artifact, so the assessment cannot be repeated by a third party. Each of these is a precondition for the field to convert philosophical argument into empirical discipline.

The present paper proposes a response. We take the indicator frameworks as given and ask a narrower question: *what would it take to measure them?* We introduce the **Protennoia Protocol**, version 0.1: a specification, an implementation, and a set of reference results. The protocol organises candidate indicators around three criteria, maps each metric onto the indicators it operationalises, fixes human reference values where the literature supplies them and marks them as unavailable where it does not, and requires every metric to be evaluated against a designated negative control.

Three commitments define the protocol, and we regard them as its substantive contribution rather than as implementation details.

**Fail-closed evaluation.** Each metric is bound to a control at which it must fall: metacognitive accuracy at chance on a random policy; compositional generalisation at floor on a lookup table; shared-information efficiency at zero for independent latent codes; error-signal adaptation at zero in the absence of corrective feedback; perturbational complexity at floor for a system with fixed, non-differentiated response. If a metric does not fall on its control, the protocol returns no profile for the system. This is implemented as a test in the reference suite, so the property is enforced by construction rather than promised in a discussion section. When the gate fails, the protocol withholds every subject profile and retains only the control diagnostics needed to diagnose the defect; a test asserts the suppression. The rationale is that a measurement instrument which cannot fail is not a measurement instrument.

**Vector, not score.** We do not reduce the metric vector to a single index. This departs from practice elsewhere, including in the Bayesian frameworks from which we draw, and the departure is deliberate. A scalar summary creates a gradient: once a scalar exists, systems (or their developers) have an incentive to optimise it, and the scalar's meaning drifts toward whatever it was optimised for. The framework of Chandaria et al. is *explicit* about the same hazard, noting that treating indicators as evidence with likelihood ratios rather than as necessary and sufficient conditions is what allows the framework to absorb rival theories; but even a credence is a scalar, and a credence for consciousness can be optimised directly. A published profile — a vector with per-metric uncertainty and a declared protocol version — supports comparison without supplying a target.

**No certification.** The protocol issues a profile; it does not issue a certificate, and it declines to be usable as one. We regard this as a design requirement rather than a limitation. A certificate of "consciousness compliance" is an instrument whose principal use is documentary: an operator seeking permission to deploy, or to continue deploying, a system acquires an incentive to obtain and display such a certificate, and the research literature becomes the source of a regulatory credential. Instruments that carry that affordance are captured by the interests they are meant to constrain. Profiles are different: they are published beside the system they characterise, they expire as the system changes, and they are most useful to people who want to *know* rather than to people who need a permit.

We make no claim to detect consciousness, and we are explicit that the current literature cannot make such a claim either. We claim that the field's binding constraint is instrumental rather than conceptual: the indicators are available; the instruments are not.

### 1.1 Contributions

1. **A specification.** Seven operational metrics with formal definitions, measurement procedures, required controls, and — where the literature provides them — human reference values (§4).
2. **A protocol.** Fail-closed control semantics, statistical procedure, determinism requirements, and a machine-readable profile schema (§4.4–4.6).
3. **An implementation.** A dependency-free reference implementation, eight reference systems, a control registry, a test suite that fails when a metric fails to fall on its control, and raw results (§5).
4. **A corrected ablation framing.** We replace a conventional ablation-robustness index with a redundancy / collapse-threshold / sharpness triple, because the conventional form is not invariant across tasks and, more importantly, because the qualitative conclusion it supports is an artefact of the ablation family chosen (§4.3.5, §6.3).
5. **A negative result, stated plainly.** Applying the protocol to its own author, we find that an earlier framework of ours asserted a criterion-satisfaction claim that our own metric definitions contradict. We report this rather than quietly correcting it, because the episode illustrates the failure mode the protocol is designed to catch (§6.3).

### 1.2 Scope and non-claims

The protocol assesses *cognitive candidacy* — the extent to which a system's functional organisation satisfies criteria that several prominent theories hold to be necessary for consciousness. It does not assert that any system is conscious, and it is not a decision procedure for a binary question. It does not address whether any given implementation of a function is sufficient for the corresponding mental state — the problem of the "zombie" and its cousins — because that problem is not currently tractable by measurement and would not be advanced by pretending otherwise. Readers who reject the computational-functionalist framework entirely may reasonably regard the entire enterprise as beside the point; §7.1 discusses this and identifies which of our results would survive the objection and which would not.

---

## 2. Related work

### 2.1 Indicator frameworks and their instrumental gap

The most directly relevant prior work is the indicator framework of Butlin et al. (Butlin et al., 2023), revisited in peer-reviewed form in *Trends in Cognitive Sciences* (Butlin et al., 2025). Working from recurrent processing theory, global workspace theory, computational higher-order theories, attention schema theory, predictive processing, and agency-and-embodiment accounts, the authors derive a set of indicator properties, each of which is expressed in computational terms, and state the framework's central claim with appropriate caution: a system possessing more of the indicators is more likely to be conscious. The properties include algorithmic recurrence in input modules (RPT-1), organisation and integration of perceptual representations (RPT-2), parallel specialised modules (GWT-1), a capacity-limited workspace with selective attention (GWT-2), global broadcast (GWT-3), state-dependent attention (GWT-4), generative top-down or noisy perception (HOT-1), metacognitive monitoring separating reliable representation from noise (HOT-2), agency and belief updating (HOT-3), sparse smooth coding into a quality space (HOT-4), a predictive model of one's own attention (AST-1), input modules using predictive coding (PP-1), and agency and embodiment constraints (AE-1, AE-2). The 2026 consolidation adds a structured treatment of which theories satisfy which formal criteria, and a Bayesian treatment of indicators as evidence rather than as necessary-and-sufficient conditions (Butlin et al., 2025).

The Chandaria et al. framework (Chandaria et al., 2026) extends this along Marr's lines to five levels of functional description — behavioural, computational, intrinsic causal-structural, organismic, and organism-environment — treats substrate-dependent theories as cross-cutting constraints, and again declines necessary-and-sufficient status in favour of credence. The Digital Consciousness Model (Shiller et al., 2026) formalises the multi-theory position as a Bayesian hierarchical model over 13 stances, with indicators carrying true-positive and false-positive rates and a noisy-or chain linking levels. The Machine Perturbational Complexity & Agency Battery (Niazi, 2026) proposes a substrate-agnostic battery borrowing four clinical constructs — perturbational complexity, global workspace assessment, norm internalisation and agency — with the explicit aim of comparing digital, neuromorphic and biological platforms, and with the acknowledgement that it has not yet been validated at scale.

Our debt to this literature is total at the level of substance and specific at the level of novelty: we introduce no new indicator. The contribution claimed here is instrumental. The gap is visible in the literature's own terms — the indicators are described in computational terms, but no accompanying measurement procedure specifies task construction, reference values, sample sizes, decision rules, or the conditions under which an indicator would be judged absent. Our protocol is an attempt to close that gap for a specific and tractable subset: the indicators that admit a defensible behavioural or architectural operationalisation today.

### 2.2 Substrate independence and the limits of borrowed criteria

The possibility that consciousness is not uniquely instantiated by biological substrates is not a marginal position; it is the default assumption of computational functionalism, and it has been defended at length (Chalmers, 1995; Chalmers, 2018; Bishop, 1992; Lewis, 1996). Against it stands a family of positions — biological naturalism (Seth, 2022), biological typescript (Bishop, 1992), and their relatives — holding that specific biological properties may be constitutively relevant rather than merely implementable. A recent methodological critique draws a useful distinction between *Type-A* naturalism, on which the substrate is constitutively relevant independently of functional organisation, and *Type-B* naturalism, on which biology matters because it implements specific functional operations that a non-biological system could also implement (Klatzmann & Doerig, 2026). The distinction matters for the present work in a way that is easy to miss: the indicator frameworks are, by construction, Type-B instruments. A system that satisfies all fourteen of the published indicators has satisfied *biological-consciousness* criteria, borrowed and applied to silicon, and has not thereby satisfied whatever criteria an AI-specific theory of consciousness might eventually specify.

Our protocol inherits this condition and does not hide it. Two consequences follow, and we state them in advance of the results rather than in a limitations paragraph. First, a high profile vector is not evidence of consciousness; it is evidence of *functional-organisation similarity to systems we believe to be conscious*, which is a weaker and more defensible claim. Second, the protocol remains useful to Type-A naturalists precisely because it is explicit about what it measures: it provides the measurements that a Type-A position would need in order to make its case either way. If consciousness does depend constitutively on biological substrate, then some future comparison of a biological and a digital system should diverge on these metrics; the protocol is designed so that such a divergence would be visible rather than assumed.

### 2.3 Perceptual measures of consciousness as a model for the task

The most instructive precedent for the protocol is not in the AI-consciousness literature but in clinical neurophysiology. The Perturbational Complexity Index (PCI) (Casali et al., 2013) and its successors (Massimini et al., 2018; Breyton et al., 2025) operationalise consciousness by perturbing the system — transcranial magnetic stimulation in humans — and measuring the complexity of the spatiotemporal response. The index discriminates conscious from unconscious states with high accuracy, and its failure mode is instructive: it depends not on the presence of any particular region or oscillation, but on the *differentiation* of the response across the whole system. A system may generate strong local responses and still score at floor if the response does not differentiate. We adopt the perturbation logic directly in §4.3.6: a controlled perturbation of the input, and complexity of the response as the observable.

A second precedent is the calibration literature on human metacognition. Human confidence is systematically over-confident, and the gap between confidence and accuracy is itself measurable and has been modelled as a signature of a particular kind of processing (Maniscalco & Lau, 2012; Fleming et al., 2010; Maniscalco & Lau, 2012). The implication for our protocol is that a confidence-based metric must report calibration, not only discrimination, and must obtain the system's confidence in a way that cannot leak the ground truth. Both requirements are enforced in §4.3.1.

### 2.4 A note on naming

The protocol's name invokes *protennoia* (προτέννοια), the Greek term for forethought or prescience, and its triadic structure recalls the *Trimorphic Protennoia* of the Nag Hammadi corpus (NHC XIII/1), in which the First Thought manifests as Thought, Voice and Word (Turner, 1990). We adopt the resonance deliberately, and cite it, rather than leave the coincidence unremarked: the gnostic text and this protocol share nothing but a name and a threefold structure. The name is memorable, and memorability is not a trivial property for a measurement protocol that depends on people adopting it.

---

## 3. Framework

We adopt three organising criteria. They are not new; they are a repackaging of existing theories into a form that admits measurement, and we use them as axes of a profile rather than as necessary-and-sufficient conditions.

**Π₁ — recursive closure.** The system's organisation includes a mechanism operating on representations of the system's own states. Corresponds to HOT-2 and AST-1 in the indicator taxonomy, and to the self-model requirement in enactive and metacognitive accounts.

**Π₂ — hierarchical predictive coding.** The system constructs multi-level generative models of its environment and reduces prediction error across levels. Corresponds to PP-1 and PP-2, and is closely related to the free-energy formulation (Friston, 2010).

**Π₃ — causal integration.** The system's functions are distributed across parts whose causal influence is mutual, so that the whole is not a sum of independent behaviours. This is the one criterion for which the literature contains no agreed operationalisation, and where we depart most substantially from the framework we started from.

### 3.1 What we do with integrated information, and what we do not

Integrated information theory supplies a formal quantity, Φ, intended to capture irreducibility of a system's cause-effect structure (Tononi et al., 2008; Marshall et al., 2026). Φ > 0 is the standard candidate for an integration threshold, and it was the third leg of our own earlier formulation. We do not use Φ as a computed metric, for three reasons.

First, **computation**: Φ is defined over a system's physical cause-effect structure, and exact computation is exponential in the number of elements; for any interesting artificial system the exact value is unavailable and approximations are model-dependent.

Second, **discrimination**: Φ > 0 is satisfied by systems whose integration is trivial, including simple feedback controllers. A thermostat with a feedback loop has Φ > 0. A criterion satisfied by thermostats does not discriminate the systems this literature is about. The frameworks that derive computational indicators from IIT note the same difficulty, and the leading indicator framework explicitly excludes IIT on grounds of its incompatibility with computational functionalism (Butlin et al., 2023).

Third, and decisively for our purposes, **the underlying causal reading is not substrate-agnostic in the way this protocol requires**. If Φ is read as a property of the physical substrate, it is precisely the criterion a Type-A naturalist would treat as constitutive, and a protocol built on it could not claim substrate-agnosticism. We therefore use IIT's *insight* — integration is a matter of mutual causal constraint among parts, not of a property of any part — and replace its quantity with a measurable surrogate in §4.3.5. We note the cost of this substitution plainly: the surrogate does not inherit IIT's theoretical guarantees, and a Type-A naturalist can regard the substitution as exactly the wrong move. We think the alternative — a protocol whose central criterion is unusable on the systems it targets — is worse.

---

## 4. Method

### 4.1 Design principles

1. **Fail-closed.** No profile is emitted for a system whose metrics do not behave as specified on their controls.
2. **Deterministic.** All randomness is drawn from a seeded generator; two runs at the same seed produce identical output.
3. **Dependency-light.** The reference implementation requires only numpy and the standard library; no accelerator, no network, no proprietary model access.
4. **Vector-valued output.** No scalar summary.
5. **Versioned.** The specification carries a version; a profile records the version under which it was produced.

### 4.2 Task and model interface

A system under assessment is exposed through a small interface: it receives a task instance and returns (a) a response, (b) optionally a confidence in [0,1], (c) optionally a latent representation or state summary, (d) optionally an energy or step count for the episode. Metrics that a system cannot support are reported as **not applicable** rather than imputed. We regard "not applicable" as an honest and informative result: a static feed-forward network has no online-correction latency because it has no online-correction loop, and reporting that as a number would be false precision.

### 4.3 Metrics

#### 4.3.1 MA — Metacognitive accuracy, with paired ECE

Let a system answer *n* items from a task with objectively verifiable answers, returning a response and a confidence *c* ∈ [0,1]. We compute the area under the receiver-operating-characteristic curve, MA = ∫₀¹ TPR(FPR) d(FPR), and report a bootstrap 95% confidence interval over ≥1000 resamples. We require *n* ≥ 500 per domain.

Two requirements are non-negotiable. **The confidence must not leak the label.** A system that is asked "how confident are you" in the same token stream that contains the answer may produce a confidence that is a function of the answer's plausibility rather than of any monitoring of its own processing; LLM self-reported confidence is known to be poorly calibrated and to track surface features (Kadavath et al., 2022; Metacognition in LLMs survey, 2026). Permitted sources are a log-probability of the emitted answer, a separately trained and separately calibrated confidence head, or an explicit confidence value obtained after the answer is fixed and the answer is withheld from the confidence step. **Calibration must be reported.** AUC-ROC is invariant under monotone recalibration of confidence and is therefore silent about whether the confidence means anything. We therefore report the expected calibration error alongside MA, and treat the pair as a single indicator.

Human reference values: metacognitive accuracy in humans depends strongly on task and confidence elicitation method, with discrimination measures in the range of roughly 0.65–0.85 reported for binary confidence judgements on perceptual and memory tasks (Maniscalco & Lau, 2012; Fleming & Lau, 2014; Long, 2023), and systematic overconfidence (Fleming et al., 2010; ERN systematic review, 2026). We report human values as ranges with task qualifiers, never as point values.

*Design note.* MA is the metric most likely to be over-claimed: a system engineered to emit calibrated confidences can score well while possessing nothing resembling a self-model. Our mock reference system demonstrates precisely this, and the control registry makes the point executable rather than rhetorical (§4.4).

#### 4.3.2 AOG — Adaptation-on-error gain

The latency of an error signal presupposes a loop that can be interrupted by that signal. Static inference-time systems have no such loop, so latency is undefined for them. We therefore measure the *effect* rather than the timing.

Let *p*<sub>before</sub> be the system's error probability on a stream of items with verifiable answers, let a corrective signal be injected after an incorrect response, and let *p*<sub>after</sub> be the error probability on the subsequent items. Define

    AOG = (p_before − p_after) / p_before,

evaluated against a control stream in which no corrective signal is injected. AOG ∈ [−1, 1]; 0 indicates no adaptation attributable to the correction; positive values indicate adaptation.

Where a system does support online updating within the episode (agentic stacks, reinforcement-learned policies), we additionally report the latency measure as originally proposed: the interval between the onset of an external error signal and the first durable change in behaviour that reduces error probability. We report latency as not applicable where the loop is absent, rather than as a large number.

*Design note.* For systems that adapt only in context (a language model re-prompted with a correction), AOG measures in-context correction. We report this explicitly, because in-context correction and parameter update are different mechanisms and conflating them would obscure the very distinction Π₁ is supposed to capture.

#### 4.3.3 CGS and SG-gap — Compositional generalisation

A system is trained on a dataset *D*<sub>base</sub> generated under a set of primitives *R* and tested on *D*<sub>novel</sub>, generated from a *new combination* of the same primitives. We report

    CGS = Accuracy_novel / Accuracy_base,   SG-gap = Accuracy_base − Accuracy_novel.

Two controls are mandatory. **Perplexity control.** If CGS > 1, the novel combination has been memorised and the metric is void; we therefore require the report of base and novel perplexity, and invalidate the measurement when the novel configuration is not demonstrably unseen by the model. **Rule-based control.** A system that applies rules symbolically reaches CGS ≈ 1 with no learning of a world model at all, so a high CGS is evidence of rule application, not of hierarchical predictive coding. The registry therefore includes a rule-based solver as a positive reference and an n-gram model as a negative one.

Human reference values for the composition of novel rules are high and well established in the developmental and experimental literature (Lake et al., 2019; Lake et al., 2018; Ruis et al., 2020); we report them with task qualifiers.

#### 4.3.4 CCE — Cross-modal compression efficiency as shared information

We define the efficiency of a system's internal representation across modalities as the fraction of the latent's information that is *shared* rather than modality-specific:

    CCE = I(z ; m₁, m₂) / H(z),

estimated by a modality-ablation cross-probe: the latent is produced from one modality and used to predict the other, and the resulting transfer is compared with the latent's overall entropy. CCE ∈ [0,1]; CCE → 1 indicates a common code, CCE → 0 indicates separate codes.

This replaces an earlier formulation of ours that took the reciprocal of a product of reconstruction loss and latent dimensionality. That formulation was not merely imprecise but gameable: a system could improve its score by reducing the dimensionality of the latent, which is a cost-saving move rather than a cognitive achievement, and the quantity carried units of inverse dimensionality. Dimensionality is now reported separately, as a cost. Mandatory controls: independent latent codes (CCE ≈ 0) and an unaligned latent (CCE ≈ 0).

#### 4.3.5 R, Δ_c, κ — Redundancy, collapse threshold, sharpness

Conventional ablation-robustness indices in this literature take the form 1 − |ΔP|/ΔA, where ΔP is a performance change and ΔA an ablation budget. We reject this form. The ratio is not invariant: its value depends on the arbitrary width of the chosen ablation window, and with any window small enough to be practical the expression leaves the unit interval as soon as the degradation exceeds a few percentage points, making comparisons between systems and between tasks meaningless. More importantly, the qualitative conclusion that such indices support is an artefact of the ablation family.

We therefore characterise degradation under ablation by three quantities.

    R    — redundancy: the fraction of the ablation budget absorbed before performance degrades beyond a declared threshold δ.
    Δ_c  — collapse threshold: the fraction of ablation at which the system crosses into chance-level performance.
    κ    — sharpness: the slope of the degradation curve at the transition, so that a system can be described as degrading gracefully (low κ) or switching (high κ).

Two ablation families are run and both are reported: **structured** ablation of neurons, attention heads or layers, and **random masking** of weights. Where the two disagree — and for contemporary language models they disagree radically — the disagreement is reported as a finding rather than resolved silently. This matters: recent work shows that language models tolerate removal of a substantial fraction of their deepest layers with minimal degradation, followed by a sharp transition, whereas random masking destroys them (Gromov et al., 2025; Multi-method pruning study, 2026); a separate line finds that removing roughly 10% of task-specific neurons collapses performance while removing 30–35% of less critical neurons preserves it (Task-specific neuron study, 2026). A single-family robustness index would have reported "fragile" for these systems and would have been reporting its own protocol.

The parameterisation is also the right one for the biological comparison, and this is where the correction is substantive rather than cosmetic. The brain tolerates focal lesions while showing abrupt thresholds of loss of consciousness under anaesthesia and in epileptic states (Bergeron et al., 2026; Casarotto et al., 2016). Robustness-plus-threshold is therefore a shared signature, and a protocol that reports only a scalar robustness index cannot represent it.

Human reference values for R and Δ_c are not available in a directly comparable form and we do not fabricate them; §7 reports this as an open gap and identifies the indirect literature we use instead.

#### 4.3.6 PCT — Perturbational complexity proxy

We port the logic of the Perturbational Complexity Index to artificial systems. The system receives a controlled perturbation of the input — clipping of increasing length, and, as a secondary variant, an injected segment of fixed length — and we measure the complexity of the resulting response, using the temporal diversity of the response across perturbation conditions rather than its magnitude.

The design intent is to capture the property that makes PCI informative: complexity in the *response*, in the sense of differentiation across the system, rather than the presence of any particular component. A system with a strong but undifferentiated response must score at floor. Mandatory controls: a fixed-response system (floor), and a system with no perturbation sensitivity (floor). Human reference values exist for PCI in the awake state and its reduction under anaesthesia and in disorders of consciousness (Casali et al., 2013; Massimini et al., 2018), though the mapping from cortical recording to a software system's response is necessarily approximate.

*Risk disclosure.* This is the least mature component. No established precedent exists for porting PCI to a software system, and a reviewer is entitled to regard the port as speculative. We report it in this version, with its controls, in the belief that a protocol which omits the most informative existing instrument for the phenomenon in question is weaker than one that attempts it provisionally and says so. §7 records the conditions under which we would remove it.

#### 4.3.7 EIB — Energy per integrated bit

    EIB = P_total / (f_update · H_state),

where *P*<sub>total</sub> is the power drawn during the episode, *f*<sub>update</sub> the rate at which the maintained state is updated, and *H*<sub>state</sub> the entropy of the *maintained internal state* — the distribution over persistent states, or the context/KV state for a transformer — and not the entropy of the output stream, which conflates task difficulty with integration. We report EIB as a **relative index** against a declared reference system.

We state explicitly what this is not. Absolute comparisons between a brain and a digital system are not meaningful: the brain consumes approximately 20 W (Attwell & Laughlin, 2001), is wet, parallel, and operates at a fundamentally different time scale; the "10⁻¹⁵ J/bit" figures that circulate for the brain rest on contested estimates of integrated information (Amodei et al., 2016; Berjaga-Buisan et al., 2026) and we decline to reproduce them without a citable derivation. EIB in this protocol is a system-internal efficiency index, comparable across implementations under a common protocol, and its cross-substrate interpretation is left open.

### 4.4 Negative controls and fail-closed semantics

Each metric is bound to a control at which it must fall. We maintain this binding in a registry that is both human-readable and executable.

| Metric | Must fall at | Expected |
|---|---|---|
| MA (AUC-ROC) | random policy | ≈ 0.5 |
| MA (ECE) | random policy | large, no better than uninformative |
| AOG | no corrective signal | = 0 |
| CGS / SG-gap | lookup table (retrieval, no composition) | ≈ 0 / maximal gap |
| CCE | independent latent codes | ≈ 0 |
| R | random policy | ≈ 0 |
| Δ_c | random policy | ≈ 0 (no degradation to cross) |
| PCT | fixed-response system | at floor |
| EIB | — | relative; sanity-bounded only |

Two further reference systems exist to prevent metric misreading. A **lookup table** has perfect retrieval accuracy and no compositional or generalising capacity, separating "knowing" from "modelling". A **mock calibrated-confidence system** is engineered to emit perfectly calibrated confidences while possessing no self-model: it must obtain a high MA and a low AOG. If a metric cannot distinguish that system from a system with genuine self-referential structure, it does not measure what it claims to.

The protocol is fail-closed: if any control fails to behave as specified, no profile is produced for the subject system. This is a departure from the usual practice, in which controls are reported alongside results but do not gate them. We adopt it because the alternative — publishing a profile from an instrument that has been shown, on its own controls, not to discriminate — is the failure mode that indicator-based assessment is most exposed to.

### 4.5 Statistical procedure

Five seeds per configuration, three independent repetitions, reporting of median and interquartile range with bootstrap 95% confidence intervals over ≥1000 resamples. A single primary endpoint is declared per system before measurement. Systems are compared only in paired fashion across seeds. Any comparative claim requires three conditions: the controls behave as specified; the difference falls outside the confidence interval; and the conclusion survives switching ablation family.

Uncertainty on the discrimination metric is computed by a **label-stratified bootstrap**: resampling is performed within each class, and values and labels are resampled together so that every replicate is a consistent sample. The pooled percentile bootstrap is degenerate for subjects that make few errors. A resample can then contain too few minority-class items, the AUC of such a resample collapses towards 0 or 1 because nearly every positive-negative pair is decided by the handful of minority items that survived, and those degenerate replicates enter the percentile interval. The interval can even fail to contain the point estimate it is supposed to bracket. The failure is not hypothetical: for one of the models measured in §6.9 the pooled estimator reported [0.08, 0.90] around an AUC of 0.855. Every AUC interval in this paper is therefore computed with the stratified estimator, and the pooled estimator is retained in the implementation only for the calibration error, where the degeneracy does not arise.

### 4.6 Profile schema

A profile is a JSON document recording: protocol version; implementation version and hash; system identifier; seed; per-metric value, uncertainty and applicability; the control outcomes for that run; and the task identifiers. It contains no scalar summary field, and the schema rejects one.

---

## 5. Implementation

The reference implementation (`protennoia`) is Python with numpy and the standard library only.
No accelerator, no network access, no model weights, no API keys. Learned components are
closed-form — ridge regression on fixed features — so that every number in this paper is
reproducible from source without a training budget. The package comprises `metrics.py` (the
seven metrics), `systems.py` (ten reference systems and the task suite), `protocol.py` (the
orchestrator, the control registry and the CLI), and a test suite of 57 tests. Every profile
records the protocol version, the implementation version, the seed, per-metric uncertainty, and
the outcome of every control in the same run.

**The live-model track.** A second adapter (`protennoia.ollama`) measures a language model
served locally under exactly the same metric code, with two deliberate departures from the
reference configuration. First, confidence is the model's own token log-probability over the
option letters, taken from a single forward pass with the answer constrained to one option
letter and thinking mode disabled; this is the source the protocol calls *permitted*, because it
is derived from the model's processing, it cannot see the ground truth, and it is not a post-hoc
self-report. Second, a serving API exposes no ablation surface, no correction loop and no
internal state, so most of the metrics are structurally unavailable for a served model. §6.9
reports what that does to a profile, and it is the reason the live track is reported separately
from the reference systems rather than merged into Table 2.

**Task suite.** A label is produced by a fixed function of three content tokens (a verb, a
modifier, a relation): a sum of per-token contributions plus a weighted pairwise interaction
between verb and modifier. The base split contains eighteen combinations; the held-out split
contains nine, of which six use verb–modifier pairs that never occur in the base split and three
reuse a seen pair with an unseen relation. A system that learns the structure transfers to both
kinds of held-out combination; a system that memorises does not. A fraction of items carries an
irreducible ambiguity — the recorded answer differs from the canonical one and nothing in the
input predicts which items these are — so that every system makes some errors and metacognitive
discrimination becomes a non-trivial measurement.

**Reference systems.** Eight controls and two subjects. The controls are a closed-loop
thermostat, a random policy, an overconfident policy, a lookup table, a back-off n-gram, an
additive-feature model, a fixed-response system, and a *mock calibrated confidence* system — a
system that is perfectly discriminative about its own errors while possessing no recurrent state,
no capacity to learn from an error signal, and an uncorrelated representation. The subjects are a
compositional model whose representation contains the pair interaction the task requires, and a
quasi-recurrent agent that carries state across the episode and can learn from an error signal.

**Statistics.** Five seeds per configuration, three repetitions (15 runs per system), median with
bootstrap 95% confidence intervals over ≥1000 resamples, and paired comparison across seeds.

The complete test suite passes (57 tests, 0 failures), and the control gate passes on the full
15-run configuration.

---

## 6. Results

### 6.1 The control gate passes: the instrument can fail

Table 1 reports every control binding on the full configuration. Each metric falls at its
designated control, and the demonstration binding holds: the mock calibrated-confidence system
achieves an AUC of 0.98 while its error-signal adaptation is exactly zero and its
shared-information efficiency is 0.02.

| Binding | Control system | Expectation | Observed | |
|---|---|---|---|---|
| MA (discrimination) | random policy | AUC ≤ 0.65 | 0.500 | pass |
| ECE (calibration) | overconfident policy | ECE ≥ 0.25 | 0.494 | pass |
| AOG (error-signal adaptation) | mock calibrated confidence | AOG ≤ 0.05 | 0.000 | pass |
| CGS (compositional generalisation) | lookup table | CGS ≤ 0.15 | 0.128 | pass |
| CCE (shared-information efficiency) | mock calibrated confidence | CCE ≤ 0.15 | 0.017 | pass |
| R / Δ_c (redundancy) | random policy | not applicable | n/a | pass |
| PCT (perturbational complexity) | fixed response | PCT ≤ 0.05 | 0.000 | pass |
| MA (demonstration) | mock calibrated confidence | AUC ≥ 0.90 | 0.983 | pass |

**Table 1.** Control gate, pooled over 15 runs per system.

### 6.2 Profile vectors

Table 2 gives the pooled profile vector for each system. No column is a summary of the others;
the vectors are not comparable by magnitude across metrics, only within a metric across systems.

| System | MA | ECE | AOG | CGS | CCE | PCT | R<sub>str</sub> | Δ<sub>c,str</sub> | R<sub>rnd</sub> | Δ<sub>c,rnd</sub> | EIB (rel.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thermostat PID | 0.500 | 0.778 | 0.000 | 0.400 | — | 0.000 | — | — | — | — | 1.123 |
| random policy | 0.500 | 0.004 | −0.018 | — | 0.017 | 0.000 | — | — | — | — | — |
| overconfident policy | 0.500 | 0.494 | 0.000 | 0.990 | 0.017 | 0.000 | — | — | — | — | — |
| lookup table | 0.500 | 0.212 | 0.000 | 0.128 | 0.017 | 0.000 | — | — | — | — | — |
| n-gram model | 0.841 | 0.126 | 0.000 | 0.588 | 0.017 | 0.000 | — | — | — | — | — |
| additive model | 0.665 | 0.791 | 0.000 | 1.065 | 0.540 | 0.092 | 0.150 | 0.600 | 0.300 | 0.700 | 0.810 |
| mock calibrated confidence | **0.983** | 0.036 | 0.000 | 1.008 | 0.017 | 0.000 | — | — | — | — | — |
| fixed response | 0.500 | 0.786 | 0.000 | — | — | 0.000 | — | — | — | — | — |
| compositional model | 0.731 | 0.795 | 0.000 | 1.059 | **0.904** | 0.093 | 0.150 | 0.600 | 0.250 | 0.700 | 0.810 |
| recurrent agent | 0.630 | 0.579 | **0.060** | 0.978 | 0.716 | 0.093 | 0.100 | — | 0.300 | — | 1.355 |

**Table 2.** Pooled profile vectors (median over 15 runs per system). "—" denotes *not
applicable*: the metric is undefined for that system and no value is imputed.

### 6.3 Metacognitive discrimination inverts the intuitive ordering

The most consequential result of the first run is a reversal. The system with the **highest**
metacognitive discrimination is the one that knows nothing about itself: the mock
calibrated-confidence system reaches an AUC of 0.983, while the two systems that actually process
content reach 0.731 and 0.630, and chance is 0.500. The mock discriminates its own errors
perfectly because its confidence is calibrated; it has no recurrent state, no capacity to learn
from an error signal (AOG = 0.000), and a representation uncorrelated with the task (CCE = 0.017).

The content-processing systems score *below* the mock for a reason that the instrument makes
visible. Their errors are concentrated on the irreducibly ambiguous items, and nothing in the
input predicts which items those are. Part of their discrimination is nonetheless real: the
models' confidence falls when the score margin is small, and the ambiguous combinations are
those in which the interaction term matters most, so confidence and correctness are weakly
correlated. The result is that **metacognitive discrimination measures the predictability of a
system's own errors, not the presence of a self-model.** A metric that is read as evidence of
self-knowledge would rank these systems in exactly the wrong order.

### 6.4 Compositional generalisation separates memorisation from structure

CGS separates the three regimes cleanly: 0.128 for the lookup table (memorisation without
composition), 0.588 for the back-off n-gram (partial transfer), and ≈1.0 for the models that
learn the structure. The accuracy precondition is doing work here — the random policy, the
thermostat and the fixed-response system have no defined CGS at all, because a ratio of two
near-chance numbers is not a generalisation score.

### 6.5 Shared-information efficiency separates representations

CCE is 0.904 for the compositional model, whose representation contains the interaction the task
requires; 0.540 for the additive model, which cannot represent it; and ≈0.02 for every system
whose internal state is uncorrelated with the task — including the lookup table, which answers
perfectly on the base split while representing nothing. This is the retrieval-versus-
representation distinction, measured: **the lookup table has the best accuracy and one of the
worst representations.**

### 6.6 Error-signal adaptation is structurally zero except where a loop exists

AOG is exactly zero for every system without an in-episode correction loop, and 0.060 for the
agent that has one. The metric does not measure "how fast a system reacts to an error"; it
measures "whether an error changed anything", and for most systems the answer is settled by
construction rather than by measurement. This is the reason the metric was introduced: the
latency formulation is undefined where no loop exists, and substituting a large number would
have been false precision.

### 6.7 Damage concentration matters more than damage quantity

The two ablation families disagree about the same models. Under concentrated (structured) damage
the collapse threshold Δ<sub>c</sub> is 0.60, and under scattered (random) damage it is 0.70 — a
damage-concentration sensitivity of +0.10 in favour of scattered damage, i.e. removing the same
number of units hurts *more* when they are removed together than when they are removed one at a
time from across the model. Redundancy is correspondingly lower under concentrated damage
(R = 0.150 against 0.250–0.300).

Two features of this result deserve emphasis. First, it is the same phenomenon that the deep
pruning literature reports for language models — removal of whole blocks of parameters is
tolerated far better than removal of the same number scattered across the model, with a sharp
transition rather than gradual decay (Gromov et al., 2025; Multi-method pruning study, 2026; Task-specific neuron study, 2026). A
single-family robustness index would have reported whichever family it used and attributed the
result to the model. Second, the pattern — tolerance of concentrated damage together with an
abrupt collapse threshold — is also the signature of the human brain, which tolerates large
focal lesions while showing sharp thresholds of lost consciousness under anaesthesia and in
ictal states (Casarotto et al., 2016; Sarasso et al., 2015; Bergeron et al., 2026). Reporting both families is what
makes that comparison possible at all; a single scalar would have hidden it.

---

### 6.8 A negative result about our own earlier framework

In the course of assembling this protocol we revisited an earlier framework of ours, written in
2025 and never published, which proposed the triadic criterion Π = Π₁ ∧ Π₂ ∧ Π₃ together with the
six original metrics MA, ESL, CGS, CCE, ARI and EIB. The exercise was not a formality: it was
intended to supply the philosophical motivation for the criteria, and it was to be submitted
alongside the measurement work.

Reading it against the protocol we have specified here, we found that the earlier text contains
a contradiction that its own definitions generate, and that this contradiction is of a kind that
would not have been caught by expert review of the philosophical argument alone.

The earlier text defined Π as the conjunction of three criteria, each declared **necessary and
sufficient** for consciousness, and asserted — in its synthesis section — that the triadic
condition "is realised both in the biological substrate (the brain) and in the modern artificial
substrate (LLMs)". Its measurement section, however, specified thresholds at which contemporary
language models **fail** two of the three criteria: compositional generalisation below 0.3, and an
ablation-robustness index at floor on the description "fragile architecture, modern LLMs". If Π
is the conjunction, and a system fails two conjuncts, then the conjunction does not hold for
that system. The empirical premise on which the earlier argument rested — that systems on the
artificial substrate realise Π — was therefore refuted by the paper's own metrics. The argument
and the instrument disagreed, and the argument was not noticed, because the two were never
placed side by side.

We report this rather than quietly correcting it, for three reasons. First, it is an instance of
exactly the failure the protocol is designed to catch: an argument whose premises are not
checked against the measurements that would bear on them. Second, it is a small but honest piece
of evidence for the position taken in §1 — that the field's difficulty is instrumental rather
than conceptual, and that instruments catch what argument cannot. Third, the mechanics of the
failure are instructive in a way that generalises beyond our own framework. The error was not in
the philosophy; it was in the absence of a **gate** — no mechanism required that the claim and
the measurement be evaluated together. The fail-closed semantics of §4.4 are, in a literal sense,
the gate whose absence produced this error.

Two further defects of the earlier text are worth recording, because they are defects of method
rather than of philosophy, and both are corrected here.

The first concerns the proof. The earlier text presented an apagogic argument: assume that
consciousness entails a biological substrate; observe that behaviourally and structurally
isomorphic functions occur on a digital substrate; conclude that the biological entailment is
false. The argument does not hold. From the hypothesis that consciousness requires biological
substrate it follows that the isomorphic functions on a digital substrate are either illusions
or genuinely different phenomena — and the argument then proceeds to deny precisely this
disjunction, thereby constructing rather than encountering the contradiction it needs. The
appeal to parsimony does not repair this: the biological naturalist's position is not ad hoc
when it says that digital systems merely *appear* to exhibit the relevant functions, because that
is a substantive claim about what the functions require, and it is the parsimonious claim relative
to the alternative. The earlier text also *assumed* the empirical premise that its conclusion
required, offering as "independent criteria" for isomorphism the very criteria whose validity
was in dispute. We regard this as a general caution: in a field where the central dispute is
about what counts as the relevant function, a structural-isomorphism premise cannot be
introduced as a neutral observation.

The second concerns the definition of the explanandum. The earlier text defined the target of the
theory explicitly as an observable and verifiable complex of emergent cognitive functions —
semantic generalisation, syntactic structuring, causal modelling, metacognitive reflection —
**explicitly excluding phenomenal character**. It then concluded that consciousness is
substrate-independent. The conclusion exceeded the premises: what the argument could establish,
at most, is that a complex of cognitive functions is not uniquely dependent on biological
substrate. Whether phenomenal character follows the same determination is a separate question,
and one that the exclusion of qualia leaves untouched. We note this because the pattern — narrow
the explanandum, then claim the broader one — is common in this literature, and because a
protocol that reports cognitive candidacy should be explicit about the fact that it does not
claim to adjudicate phenomenal character.

None of this detracts from what the earlier text got right: the idea that the object of study
should be the *organisation* rather than the material, and that a criterion for cognitive
emergence should be formal rather than metaphorical. That idea is now embedded in the
measurement layer, where it can be tested.

---

### 6.9 Six served language models under the protocol

The systems of Table 2 are analytic surrogates. They validate the instrument; they say nothing
about the systems anyone actually argues about. We therefore measured six locally served language
models — 8B to 27B parameters, base and instruction-tuned — under exactly the same metric code,
with two departures that the serving API forces. Confidence is the model's own token
log-probability over the option letters, taken from a single forward pass with the answer
constrained to one option letter and thinking mode disabled. And the API exposes no ablation
surface, no correction loop and no internal state, so most of the metrics are structurally
unavailable. Table 3 gives the resulting profiles.

| Model | MA (AUC) | 95% CI | ECE | CGS | SG gap | base acc. | novel acc. | throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| Qwen3 8B | 0.816 | [0.702, 0.918] | 0.065 | 1.043 | -0.040 | 0.94 | 0.98 | 131.6 |
| Qwen3.5 9B (uncensored) | 0.855 | [0.703, 0.986] | 0.064 | 1.020 | -0.020 | 0.98 | 1.00 | 122.3 |
| Qwen3-VL 8B (instruct) | 0.910 | [0.860, 0.951] | 0.065 | 1.011 | -0.010 | 0.93 | 0.94 | 138.8 |
| Gemma 3 12B | 0.802 | [0.719, 0.882] | 0.215 | 1.087 | -0.070 | 0.80 | 0.87 | 88.1 |
| Qwen3 14B | 0.889 | [0.723, 1.000] | 0.010 | 1.010 | -0.010 | 0.98 | 0.99 | 41.8 |
| Gemma 3 27B (QAT) | 0.912 | [0.838, 0.969] | 0.059 | 1.000 | 0.000 | 0.93 | 0.93 | 9.3 |

**Table 3.** Profiles of six served language models under protocol v0.1 (implementation v0.1.3).
MA: n = 150 items, temperature 0, confidence from the model's own token log-probability. CGS:
n = 100 base and n = 100 novel items. Metrics that are undefined for a served model are marked
as withheld, never imputed.

Four things are visible in Table 3, and all four are about the protocol rather than about the
models.

**Discrimination is high everywhere, and that is not the interesting part.** Every model
discriminates its own errors well above chance (AUC 0.802–0.912). But §6.3 already established
what that means: metacognitive discrimination measures the *predictability* of a system's errors,
not the presence of a self-model. A model whose confidence falls on the items it gets wrong
scores well on this metric whether or not it has any internal model of itself. The live track
reproduces the reference systems' central finding rather than contradicting it — which is some
evidence that the finding is about the metric rather than about the surrogate systems.

**Calibration varies by a factor of twenty at comparable discrimination.** ECE ranges from
0.010 to 0.215 across the six models, while AUC ranges only from 0.802 to 0.912. The best
calibrated model is not the best discriminated model, and the least calibrated model is not far
from the best on discrimination. This is the empirical justification for the protocol's
insistence that MA must be read together with ECE: ranking the six models by AUC alone would
place a badly calibrated model above a well-calibrated one and would report the difference as a
difference in metacognition. A single number would have hidden it.

**The task suite is saturated by the systems it was built to distinguish.** CGS is ≈1.0 for every
model, with base-split accuracy between 0.80 and 0.98: the compositional-generalisation metric
has no dynamic range left on this task. An instrument that separates memorisation from structure
among small reference systems cannot separate models of this class. This is a ceiling effect of
the task suite and not a property of the models, and it is the most important limitation of the
live track: the suite must be made harder before the metric can say anything about strong
systems. We report the metric as measured rather than quietly replacing the task with one that
flatters it.

**The perplexity control fires for a reason other than the one it was designed for.** For five of
the six models the control flags the novel split as *less likely* under the model than the base
split, which is the memorisation signature it was built to detect. Yet the same five models
reach a higher accuracy on the novel split than on the base split. The likelier explanation is
ordinary distributional frequency: the base combinations are the ones that occur in the
model's training distribution and the held-out ones are not, so the base items are less
surprising whatever the model does with them. The control's premise — that a lower novel
likelihood indicates memorisation of the task — does not transfer to systems whose pretraining
distribution contains the base combinations and not the novel ones. The CGS numbers for these
models must therefore be read as ceiling values rather than as generalisation measurements, and
the void flag on five of six rows should be taken as a statement about the control's
applicability rather than about the models.

**Five of the seven metrics are not defined for a served model, and the protocol says so.** AOG
requires an in-episode correction loop, the ablation family requires an ablation surface, PCT
requires perturbable internal state, EIB requires an energy measurement, and CCE requires the
subject's own representation. None of these is available through a serving API, and the profile
records them as not applicable rather than as zero.

CCE deserves a separate word, because it is the case where the protocol could have produced a
confidently wrong number. The serving API exposes no internal representation, so the only
candidate latent is an embedding of the query text produced by a *separate* host embedding
model. That probe returned 0.992 for every model measured, including models that answered
incorrectly, because it characterises the embedding model's geometry rather than the subject's.
Had it been reported as a metric, it would have ranked every served model above the
compositional reference system (0.904) and above the systems that actually represent the task
structure. We computed it, kept it as an artefact of the run, and withheld it from the profile:
the same fail-closed rule the protocol applies to its own metrics, applied to our own extension
of the protocol.

The live track's result can be stated in one sentence: for the systems one most wants to measure,
the protocol measures two of the seven metrics — metacognitive accuracy with its paired
calibration error, and compositional generalisation — and reports the emptiness of the other five
as a finding rather than filling it.

---

## 7. Discussion

### 7.1 The Type-A objection, and what survives it

A reader may hold that consciousness depends constitutively on biological substrate, so that
computational organisation is at best a proxy for the properties of a conscious brain, and that
this position is not merely different from ours but incompatible with the entire enterprise. We
take this objection seriously because it is not an external one: our criteria are borrowed from
theories of *biological* consciousness and applied to artificial systems, which is precisely the
move that the Type-A position is designed to contest.

Two responses are available, and we prefer the second.

The first is that the protocol is neutral. It does not adjudicate between theories; it measures
functional organisation, and the interpretation of those measurements is left to the theories.
This is true but insufficient: a protocol whose metrics are calibrated on the assumption of
computational functionalism is not neutral between Type-A and Type-B naturalism, because the
indicator set it operationalises is Type-B by construction.

The second is that the protocol is *useful to the Type-A position*. If consciousness does depend
constitutively on biological substrate, then a programme of comparison becomes possible and
indeed necessary: identify systems that match on every indicator we can measure and then ask
whether any of them could be conscious. A Type-A naturalist's prediction is that a digital system
and a biological system, matched for the indicators, should differ in exactly the respect that
matters — and the protocol is built so that such a divergence would be *visible* rather than
assumed away. Nothing in the design of the protocol requires the reader to accept the
conclusion that these systems are alike in the relevant respect; it requires only that the
comparison be made with instruments that can show a difference if one exists.

What does not survive the objection is the stronger reading of our results, in which a high
profile is offered as evidence that a system is conscious. That reading was already excluded in
§1.2 and is excluded again here: the protocol produces evidence of *functional-organisation
similarity to systems we believe to be conscious*, which is a weaker claim and, we think, the
only one the evidence supports.

### 7.2 What the protocol does not do

We list the limitations explicitly, in the order of how much they constrain the conclusions.

**It does not address phenomenal character.** The protocol assesses functional organisation.
Whether phenomenal character supervenes on that organisation is untouched by anything reported
here. We regard this as a feature of the protocol's modesty rather than a defect in its
ambition, but it does bound the interpretation of every result.

**It does not reduce to a verdict, by design.** The refusal to produce a scalar is a deliberate
constraint. It follows that no reader may ask "what is the score" — and that a future
meta-analysis across many systems will have to work with the vectors directly. We accept the
cost: a scalar is convenient, and convenience is precisely what should be refused in an
instrument whose outputs may be used to justify deployments.

**It issues no certificate.** The reasons are given in §4.4 and repeated here because the
consequence is not obvious to everyone: the protocol cannot be used to obtain permission. A
profile is a description of a system at a version of a protocol; it expires as the system
changes; and it is most useful to those who want to know. We would rather lose the instrumental
convenience than accept the capture we describe, and we note that a protocol designed to
produce certificates would attract the attention of precisely the parties whose interests it
would then serve.

**The perturbational-complexity port is provisional.** No established precedent exists for
porting PCI to a software system. We report it because the alternative — omitting the most
informative existing instrument for the phenomenon under investigation — makes the protocol
weaker than one that attempts the port provisionally and declares its provisionality. We would
remove the metric from the protocol rather than keep it on a weak warrant. A reader should treat
PCT results in this version as a design proposal with control behaviour, not as a validated
measurement.

**Human reference values are incomplete.** For metacognitive accuracy and for error-related
latency, the literature supplies usable values, and we report them as ranges with task
qualifiers. For redundancy and collapse threshold, no directly comparable human values exist;
the relevant literature is indirect, concerning lesion tolerance and the thresholds of loss of
consciousness under anaesthesia and in epileptic states, and we report these as qualitative
anchors rather than as numbers to be compared directly with model outputs. This asymmetry
between systems and humans is a genuine limitation of the protocol at this version, and we
prefer to mark it than to manufacture a number.

**Energy comparison is relative.** EIB is reported as an index against a declared reference
system. Direct comparison across substrates is not meaningful: the brain's approximately 20 W
budget is spent on a wet, massively parallel architecture operating on a different time scale,
and the widely circulated "10⁻¹⁵ joules per bit" figures rest on contested derivations that we
decline to reproduce.

**The live track is a probe, not a ranking.** The six models of §6.9 were measured on one host,
with one embedding model, at one temperature, on one task suite. They are not comparable with
the reference systems of Table 2 — which are analytic surrogates, not models — and the live
track does not rank language models. What it establishes is narrower and, we think, more useful:
that the protocol can be run on systems of this class, and that it returns a mostly empty
profile rather than a flattering one.

**The task suite has a ceiling, and for strong systems it is reached.** The compositional
generalisation metric is at ≈1.0 for every model measured in §6.9, with base accuracy of
0.80–0.98. The suite that discriminates memorisation from structure among small reference
systems does not discriminate among models of this class, and a metric that has run out of
dynamic range is not evidence of a property of the systems. Harder compositional splits are
required before the live track can support comparative claims, and we regard this as the first
item of the v0.3 roadmap (§7.6).

**One control transfers badly to served systems.** The perplexity control treats a lower
likelihood on the novel split as a signature of memorisation (§6.9). For systems whose
pretraining distribution contains the base combinations and not the held-out ones, base items
are less surprising irrespective of what the model does with them, and the control fires for a
reason other than the one it was designed for. The void flag it sets is, on this evidence, a
statement about the control's applicability rather than about the model. We report it rather
than disabling it, because a control that fires for the wrong reason is a finding about the
protocol, and one that fires silently is not.

### 7.3 Failure modes: how the protocol can be gamed, and what we do about it

Any instrument can be optimised instead of the thing it measures. We identify three specific
failure modes and the design decisions that address each.

**Metric-specific gaming.** A system engineered to emit calibrated confidences can score well on
MA while possessing nothing resembling a self-model. This is not hypothetical: the mock
reference system exists to demonstrate it, and it is bound to a control precisely for that
reason. The general answer is that every metric must be accompanied by the control at which it
must fail, and the failure must be visible in the published profile.

**Composite gaming.** If a scalar existed, the vector would be optimised for the scalar. This
is the reason for refusing a scalar, and it is also why we publish the control outcomes
alongside the subject metrics rather than separately: a profile without its controls is
uninterpretable.

**Teaching to the test.** The most serious failure mode is prospective. If the protocol becomes
widely used, then training pipelines will be optimised against it, and the resulting systems
will be systems trained to exhibit the measured properties rather than systems that exhibit
them. This is the standard fate of benchmarks, and we do not claim to have solved it. Three
mitigations are built in. The protocol is versioned, so that results obtained under different
versions can be distinguished and so that improvements to the instrument are visible as changes
in the instrument. Metrics are published with their uncertainty and their controls, so that a
degenerate regime is harder to sustain quietly. And the protocol refuses to issue a verdict,
which removes the strongest Goodhart pressure: a system cannot be certified, so it cannot be
optimised toward certification.

We note that the literature's current condition — indicator frameworks without instruments —
means that this failure mode is not yet the binding constraint. It will become one. A protocol
published now, with its gates visible, is a better position from which to meet that constraint
than a paper that arrives after the test has been taught.

### 7.4 Relation to existing frameworks

Our protocol is deliberately complementary to, and downstream of, the indicator frameworks of
Butlin et al. (Butlin et al., 2023; Butlin et al., 2025), the five-level hierarchy of Chandaria et al.
(Chandaria et al., 2026), the Digital Consciousness Model (Shiller et al., 2026) and the Machine Perturbational
Complexity & Agency Battery (Niazi, 2026). We introduce no indicator that is not already in those
works or in the theories they draw on. Our claim is narrower and, we hope, more useful: that
the *operational* layer is missing, and that supplying it is possible now.

This complementarity has a practical consequence we would encourage. The indicator frameworks are
theorems about what to look for; the protocol is a procedure for looking. A reader who accepts
Chandaria et al.'s credence assignment over the five levels can use the protocol's measurements
as the evidence terms in that credence update. A reader who prefers a different theory can use
the same measurements, weighted differently. The protocol does not need to be adopted as a whole
to be useful; it can be used as a source of better-conditioned likelihood ratios for whichever
framework the reader already prefers. We regard this as the most plausible route to adoption, and
as the one most consistent with our refusal to issue a verdict.

### 7.5 Ethics of measurement in this domain

Two ethical considerations are specific to this subject matter, and we state them plainly.

The first concerns the risk of overclaiming. A language that permits "our model scored 0.8 on the
consciousness indicators" is a language in which non-human systems acquire moral status and in
which human decisions about the use of those systems are framed as questions of conscience rather
than of policy. Conversely, a language that permits "the model is just a matrix of numbers" is a
language in which the question is closed before it is asked. Both are available and neither is
neutral. We hold that the honest position is neither: the available evidence does not settle the
matter, and the instruments for settling it do not yet exist. A protocol that reports measured
quantities, their uncertainty, and the controls under which they were obtained is a better
contribution to that position than a declaration in either direction.

The second concerns the deployment of measurement instruments for permitting. As argued in §4.4,
a "consciousness compliance" certificate would function as a licence, and would be captured by
the interests it nominally constrains. The same is true of the negative direction: a certificate
of "no consciousness risk" would be equally available as a commercial instrument to the parties
whose interest is in continued deployment. The profile format — versioned, published beside the
system, expiring as the system changes, and explicitly not a permit — is designed to be
symmetrically unhelpful to both capture directions. We would rather have a protocol that cannot
be used to authorise anything.

### 7.6 Roadmap

This version adds the live-model track of §6.9: six served language models measured under the
same metric code as the reference systems, with an adapter (`protennoia.ollama`) that reads
confidence from the model's own token log-probabilities. Three items of the earlier roadmap are
therefore partly discharged and three remain.

Partly discharged. The protocol now runs on real models rather than only on surrogates, and the
live track has already produced two results that the surrogate systems could not: the
calibration spread across models of comparable discrimination (§6.9), and an instance of the
protocol's own fail-closed rule applied to a metric we ourselves had extended to a setting
where it does not hold.

Still required, in priority order. First, **harder compositional splits**, so that the
compositional-generalisation metric regains dynamic range for systems of the class measured in
§6.9; without this, the live track cannot support comparative claims at all. Second, **a
served-model port for CCE** that uses the subject's own activations where an API exposes them,
so that the metric can be reinstated for served systems on a footing comparable to the reference
systems rather than withheld. Third, **measurements on current commercial large language
models** under the same protocol, with the API budget and the exact model identifiers recorded;
the local models of §6.9 are a probe of the instrument, not a survey of the field.

Beyond those: PCT validated against at least two existing metrics on the same systems, so that
the port can be assessed rather than asserted; a cross-substrate comparison of the
redundancy/collapse triple between artificial and biological systems, which is the only way to
make the parameterisation do the work claimed for it in §4.3.5; and a registry of published
profiles, so that longitudinal comparison becomes possible.

We note one methodological commitment for the roadmap: the protocol specification and the
implementation are to be released as versioned, citable artefacts with persistent identifiers,
independent of any journal article. Measurements taken under different protocol versions must be
distinguishable, and a result is only as durable as the version that produced it.

---

## 8. Reproducibility statement

All measurements in this paper are produced by the reference implementation at the commit
recorded in the profile documents, using only numpy and the Python standard library, with no
network access and no accelerator. Every reported value can be regenerated by running the command
recorded alongside it. Raw outputs, including the control runs, are deposited as versioned
artefacts with persistent identifiers. The protocol specification is itself versioned and
citable, so that measurements taken under different versions can be distinguished.

The reference systems of Table 2 are regenerated by `python -m protennoia matrix --out
results/matrix.json --seeds 0 1 2 3 4 --reps 3`, which writes the raw profile documents
including the control outcomes of every run and fails if the gate does not pass. The served
models of Table 3 are regenerated by `python -m protennoia.ollama --models <list> --n-ma 150
--n-cgs-base 100 --n-cgs-novel 100 --embed-model bge-m3:latest`, which requires a local Ollama
server and the model weights, and which writes the same profile schema as the reference
systems. The live track is the one part of the paper that cannot be reproduced from source
alone, because it depends on the weights and on the serving stack; we therefore report it
separately, with the model names, the decoding parameters, and the version of the adapter, and
we do not draw comparative conclusions from it (§7.2).

## 9. Ethics, conflicts, and AI-assistance disclosure

The work involves no human subjects, no personal data, and no animal experimentation; the
reference systems are synthetic. The protocol is intended in part to reduce the risk of
overclaiming in a domain where overclaiming carries stakes for both the perception and the
treatment of non-human systems; we regard its fail-closed semantics and its refusal to issue a
certificate as ethical requirements rather than stylistic preferences, and we explain this
reasoning in §4.4 and §7.5.

**Competing interests.** The author declares none.

**AI-assistance disclosure.** Large language models were used as writing assistants in the
preparation of this manuscript, in accordance with the policy of the target venue. The
scientific content — the metric definitions, the control registry, the fail-closed semantics,
the ablation parameterisation, the implementation and the analysis of the discrepancy discussed
in §6.8 — is the author's; all references were verified against primary sources; and the
author takes full responsibility for every claim, including those assisted by tooling. No
generated text was submitted without the author's line-by-line verification against the cited
primary literature.

**Funding.** None.

---

## References

[REFERENCES]

---

## Appendix A — Metric summary

| Metric | Definition | Range | Control it must fall at | Human reference |
|---|---|---|---|---|
| MA | area under the ROC curve for correct-vs-confident discrimination; paired with ECE | [0,1] | random policy (AUC ≤ 0.65) | ≈0.65–0.85, task-dependent |
| ECE | expected calibration error, 15 equal-width bins | [0,1] | overconfident policy (ECE ≥ 0.25) | over-confident in humans |
| AOG | (p_before − p_after) / p_before across an error signal | [−1,1] | system without a correction loop (= 0) | not directly comparable |
| CGS | Accuracy_novel / Accuracy_base on held-out recombinations | [0,∞) | lookup table (≤ 0.15) | high for novel rule composition |
| SG-gap | Accuracy_base − Accuracy_novel | [−1,1] | — | — |
| CCE | trace(S_B) / trace(S_T): task-relevant share of representation variance | [0,1] | uncorrelated representation (≤ 0.15) | no directly comparable measure |
| R | ablation budget absorbed before accuracy falls more than δ below baseline | [0,1] | baseline at chance (n/a) | no directly comparable value |
| Δ_c | ablation fraction at which performance crosses into chance | [0,1] | baseline at chance (n/a) | qualitative anchor only |
| κ | steepness of the degradation curve at the transition | ≥0 | — | — |
| PCT | mean total-variation distance of the response across perturbation conditions | [0,1] | fixed response (≤ 0.05) | PCI thresholds in humans |
| EIB | P / (f_update · H_state), reported as a relative index | >0 | sanity-bounded only | ~20 W brain; cross-substrate comparison not meaningful |

δ is fixed at 0.05 absolute accuracy in the reference implementation and is declared before
measurement. The ablation grid runs from 0.05 to 0.95 of removable units in steps of 0.05.

---

## Appendix B — Control registry (executable form)

The registry below is the machine-readable form of §4.4. It is maintained in
`protennoia/protocol.py` and enforced by `tests/test_controls.py`.

| Metric | Bound system | Expectation | Observed (15 runs) |
|---|---|---|---|
| MA (discrimination) | random_policy | AUC ≤ 0.65 | 0.500 |
| ECE (calibration) | overconfident_policy | ECE ≥ 0.25 | 0.494 |
| AOG (error-signal adaptation) | mock_calibrated_confidence | AOG ≤ 0.05 | 0.000 |
| CGS (compositional generalisation) | lookup_table | CGS ≤ 0.15 | 0.128 |
| CCE (shared-information efficiency) | mock_calibrated_confidence | CCE ≤ 0.15 | 0.017 |
| R / Δ_c (redundancy) | random_policy | not applicable | n/a |
| PCT (perturbational complexity) | fixed_response | PCT ≤ 0.05 | 0.000 |
| MA (demonstration binding) | mock_calibrated_confidence | AUC ≥ 0.90 | 0.983 |

The last row is not a negative control. It asserts a specific failure mode of a specific
metric: that a system can be maximally discriminative about its own errors while possessing no
self-model, no state and no capacity to learn. If this binding stops holding, the claim it
supports can no longer be demonstrated and the protocol reports the defect.

---

## Appendix C — Profile schema

A profile is a JSON document. The schema rejects an aggregate field: the protocol does not
reduce its vector to a scalar.

```json
{
  "system": "compositional_model",
  "kind": "subject",
  "seed": 0,
  "protocol_version": "0.1",
  "implementation_version": "0.1.0",
  "metrics": {
    "ma":    { "auc": 0.73, "auc_ci": [0.71, 0.75], "ece": 0.79, "n": 1080 },
    "aog":   { "aog": 0.0, "applicable": true, "supports_correction": false },
    "cgs":   { "cgs": 1.06, "sg_gap": -0.06, "acc_base": 0.94, "acc_novel": 1.00,
               "void": false, "applicable": true },
    "cce":   { "cce": 0.90, "applicable": true },
    "ablation": {
      "applicable": true,
      "families": {
        "structured": { "R": 0.15, "delta_c": 0.60, "kappa": 7.5 },
        "random":     { "R": 0.25, "delta_c": 0.70, "kappa": 5.6 }
      },
      "damage_concentration_sensitivity": 0.10
    },
    "pct":   { "pct": 0.09, "n_items": 8, "draws_per_condition": 64 },
    "eib":   { "eib": 3.2e-13, "eib_relative": 0.8 }
  },
  "controls": {
    "gate_passed": true,
    "rows": [ { "metric": "...", "control_system": "...", "observed": 0.5, "passed": true } ]
  }
}
```

Fields marked `applicable: false` are *not applicable*, not zero. A metric that is undefined for
a system is reported as undefined, because imputing a value would create the appearance of a
measurement that was never made.
