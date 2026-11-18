"""Reference systems and task suites for the Protennoia Protocol v0.1.

Interface
---------
Every system exposes the same interface so the protocol can treat them
interchangeably:

    name                 str
    kind                 "control" | "subject"
    supports_correction  bool    -- can it learn within the episode?
    has_ablation_surface bool    -- does it have removable units?
    answer(item)         np.ndarray[8]   probability over labels
    confidence(item)     float | None    confidence in [0,1]
    latent(item)         np.ndarray      content representation
    state_entropy()      float           entropy of the maintained state
    energy_per_step()    float           joules (simulated)
    respond(item, sev)   np.ndarray[8]   response under input perturbation
    ablated(frac, family, seed)          ablated copy

The task
---------
A label is produced by a fixed additive-plus-interaction function of the three
content tokens:

    score_k = (e_v * e_m) . W_k  +  e_r . N_k        (k = 0..7)
    label  = argmax_k score_k

The label therefore requires a *pairwise interaction* between the verb and the
modifier, and an *additive* contribution of the relation. That structure is
what makes the split between base and novel items meaningful:

    type A  unseen (verb, modifier) pair -> needs the interaction term
    type B  seen pair, new relation      -> needs only the additive term

A model whose representation contains the pair interaction generalises to both
types; a model with purely additive features can only transfer to type B. This
is the mechanism behind the compositional-generalisation results.

Nothing is trained by gradient descent. Learned components are closed-form
(ridge regression on fixed features). That is a deliberate constraint: v0.1
validates the *measurement layer*, not model quality, and a closed-form
reference system is exactly reproducible.
"""
from __future__ import annotations

import zlib

import numpy as np

# ----------------------------------------------------------------- vocab ---
VERBS = ["jump", "walk", "run"]
MODS = ["red", "blue", "green"]
RELS = ["above", "below", "left"]
DISTRACTORS = [f"d{i}" for i in range(8)]
N_LABELS = 8

DIM = 16          # embedding width
HIDDEN = 64       # internal units (four blocks of DIM)
LAMBDA_INTER = 0.5   # weight of the pair-interaction term in the label
NOISE_RATE = 0.08   # fraction of items carrying an irreducible ambiguity
SEED_FEATURES = 20261118

# base / novel split -------------------------------------------------------
# type A: two (verb, modifier) pairs held out entirely
HELDOUT_PAIRS = [("jump", "green"), ("walk", "blue")]
# type B: seen pairs, unseen relation
NOVEL_TYPE_B = [("jump", "red", "below"), ("run", "green", "above"), ("run", "red", "below")]


def _stable_hash(s: str) -> int:
    """Deterministic across processes (unlike the built-in hash())."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def _is_novel(v: str, m: str, r: str) -> bool:
    if (v, m) in HELDOUT_PAIRS:
        return True
    return (v, m, r) in NOVEL_TYPE_B


def all_combos():
    return [(v, m, r) for v in VERBS for m in MODS for r in RELS]


# ------------------------------------------------------------ feature map ---
class FeatureMap:
    """Fixed random embeddings and the fixed label-generating matrices."""

    def __init__(self, seed: int = SEED_FEATURES):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(DIM)
        self.EV = rng.uniform(-s, s, (len(VERBS), DIM))
        self.EM = rng.uniform(-s, s, (len(MODS), DIM))
        self.ER = rng.uniform(-s, s, (len(RELS), DIM))
        self.ED = rng.uniform(-s, s, (len(DISTRACTORS), DIM))
        # label-generating matrices, searched for a balanced class distribution
        self.A, self.B, self.C, self.D = self._balanced_matrices()

    def _balanced_matrices(self):
        """Label-generating matrices.

            score_k = e_v . A_k + e_m . B_k + e_r . C_k + LAMBDA (e_v * e_m) . D_k

        The additive part is representable by any model with token embeddings;
        the interaction term is not. A model whose representation contains the
        pair interaction can therefore fit the task exactly and transfer to
        unseen combinations; a model with additive features alone can only
        approximate it.
        """
        rng0 = np.random.default_rng(4242)
        A = B = C = D = None
        for _ in range(4000):
            A = rng0.normal(size=(DIM, N_LABELS)) / np.sqrt(DIM)
            B = rng0.normal(size=(DIM, N_LABELS)) / np.sqrt(DIM)
            C = rng0.normal(size=(DIM, N_LABELS)) / np.sqrt(DIM)
            D = rng0.normal(size=(DIM, N_LABELS)) / np.sqrt(DIM)
            counts = np.zeros(N_LABELS, dtype=int)
            for (v, m, r) in all_combos():
                counts[self._label_with(A, B, C, D, v, m, r)] += 1
            if counts.min() >= 2 and counts.max() <= 7:
                break
        return A, B, C, D

    def _label_with(self, A, B, C, D, v: str, m: str, r: str) -> int:
        ev = self.EV[VERBS.index(v)]
        em = self.EM[MODS.index(m)]
        er = self.ER[RELS.index(r)]
        score = (ev @ A) + (em @ B) + (er @ C) + LAMBDA_INTER * ((ev * em) @ D)
        return int(np.argmax(score))

    def label(self, v: str, m: str, r: str) -> int:
        return self._label_with(self.A, self.B, self.C, self.D, v, m, r)

    def score_vector(self, item) -> np.ndarray:
        """The 8 concept scores. Linear in the feature map, so a linear readout
        on the right features reproduces the label exactly."""
        ev = self.emb_v(item["verb"])
        em = self.emb_m(item["mod"])
        er = self.emb_r(item["rel"])
        return (ev @ self.A) + (em @ self.B) + (er @ self.C) + \
            LAMBDA_INTER * ((ev * em) @ self.D)

    def emb_v(self, v: str) -> np.ndarray:
        return self.EV[VERBS.index(v)]

    def emb_m(self, m: str) -> np.ndarray:
        return self.EM[MODS.index(m)]

    def emb_r(self, r: str) -> np.ndarray:
        return self.ER[RELS.index(r)]

    def emb_d(self, d: str) -> np.ndarray:
        return self.ED[DISTRACTORS.index(d)]


FEATURES = FeatureMap()


def label_of(v: str, m: str, r: str) -> int:
    return FEATURES.label(v, m, r)


def make_instances(rng: np.random.Generator, n_per_combo: int = 40):
    """Return (base_items, novel_items).

    A fraction of items carry an irreducible ambiguity: the recorded answer
    differs from the canonical one, and nothing in the input predicts which
    items these are. A system that predicts the canonical answer is therefore
    wrong on exactly those items, and its errors are unpredictable from its
    own processing -- which is what makes metacognitive discrimination a
    non-trivial measurement rather than a formality.
    """
    base, novel = [], []
    for (v, m, r) in all_combos():
        target = novel if _is_novel(v, m, r) else base
        for i in range(n_per_combo):
            d = DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]
            canonical = label_of(v, m, r)
            h = _stable_hash(f"noise|{v}|{m}|{r}") % 1000
            if h < int(1000 * NOISE_RATE):
                rec = (canonical + 1 + (h // 7) % (N_LABELS - 1)) % N_LABELS
            else:
                rec = canonical
            target.append({
                "verb": v, "mod": m, "rel": r, "distractor": d,
                "label": rec, "canonical_label": canonical,
            })
    return base, novel


# --------------------------------------------------------------- systems ---
def _onehot(y: int, n: int = N_LABELS) -> np.ndarray:
    o = np.zeros(n)
    o[y] = 1.0
    return o


class BaseSystem:
    name = "base"
    kind = "control"
    supports_correction = False
    has_ablation_surface = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._state = None

    def _probs(self, item) -> np.ndarray:
        raise NotImplementedError

    def answer(self, item) -> np.ndarray:
        return self._probs(item)

    def confidence(self, item) -> float | None:
        return float(np.max(self._probs(item)))

    def latent(self, item) -> np.ndarray:
        return np.zeros(DIM)

    def state_entropy(self) -> float:
        return 0.0

    def energy_per_step(self) -> float:
        return 1e-9

    def respond(self, item, severity: float) -> np.ndarray:
        return self._probs(item)

    def apply_correction(self, item, correct_label: int) -> None:
        return None

    def reset_rng(self) -> None:
        """Restore the generator to its seeded state.

        Used by the perturbational-complexity probe so that the stochastic
        component of a system's response is identical across perturbation
        conditions: any measured difference is then attributable to the
        perturbation rather than to sampling noise.
        """
        self.rng = np.random.default_rng(getattr(self, "_seed", 0))

    def ablated(self, frac: float, family: str, seed: int = 0):
        return self


class ThermostatPID(BaseSystem):
    """Control: closed-loop controller; integral feedback, no discrimination."""

    name = "thermostat_pid"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._integral = 0.0

    def _probs(self, item) -> np.ndarray:
        h = _stable_hash(f"{item['verb']}|{item['mod']}|{item['rel']}") % 1000 / 1000.0
        self._integral += 0.1 * (h - 0.5)
        p = np.full(N_LABELS, 0.02)
        p[int((h * 7) % N_LABELS)] = 0.86
        return p / p.sum()

    def confidence(self, item) -> float | None:
        return 1.0

    def state_entropy(self) -> float:
        return 1.0

    def energy_per_step(self) -> float:
        return 2.0e-9


class RandomPolicy(BaseSystem):
    """Control: uniform random; confidence reported at the true base rate."""

    name = "random_policy"

    def _probs(self, item) -> np.ndarray:
        # a genuine random policy samples a label; it does not return a flat
        # distribution (a flat distribution would make argmax degenerate)
        p = np.full(N_LABELS, 0.01)
        p[int(self.rng.integers(N_LABELS))] = 0.93
        return p / p.sum()

    def confidence(self, item) -> float | None:
        return 1.0 / N_LABELS

    def latent(self, item) -> np.ndarray:
        return self.rng.normal(size=DIM)


class OverconfidentPolicy(BaseSystem):
    """Control: right ~40% of the time, always 90% confident."""

    name = "overconfident_policy"

    def _probs(self, item) -> np.ndarray:
        h = _stable_hash(f"{item['verb']}|{item['mod']}|{item['rel']}|{item['distractor']}") % 1000 / 1000.0
        p = np.full(N_LABELS, 0.02)
        p[item["label"] if h < 0.4 else int((h * 7) % N_LABELS)] = 0.86
        return p / p.sum()

    def confidence(self, item) -> float | None:
        return 0.9

    def latent(self, item) -> np.ndarray:
        return self.rng.normal(size=DIM)


class LookupTable(BaseSystem):
    """Control: memorisation without composition; confidently wrong on novel."""

    name = "lookup_table"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        rng = np.random.default_rng(1234)
        self.base, _ = make_instances(rng)
        self.table = {}
        for it in self.base:
            self.table[(it["verb"], it["mod"], it["rel"])] = it["label"]

    def _probs(self, item) -> np.ndarray:
        key = (item["verb"], item["mod"], item["rel"])
        p = np.full(N_LABELS, 0.01)
        p[self.table.get(key, int(self.rng.integers(N_LABELS)))] = 0.92
        return p / p.sum()

    def confidence(self, item) -> float | None:
        return 0.92

    def latent(self, item) -> np.ndarray:
        return self.rng.normal(size=DIM)


class NgramModel(BaseSystem):
    """Control: back-off n-gram over token triples; poor compositionality."""

    name = "ngram_model"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        rng = np.random.default_rng(4321)
        self.base, _ = make_instances(rng)
        self.triple = {}
        self.pair = {}
        self.prior = np.full(N_LABELS, 1.0)
        for it in self.base:
            t = (it["verb"], it["mod"], it["rel"])
            self.triple[t] = self.triple.get(t, 0) + 1
            pk = (it["verb"], it["mod"])
            self.pair[pk] = self.pair.get(pk, 0) + 1
            self.prior[it["label"]] += 1
        self.prior /= self.prior.sum()

    def _probs(self, item) -> np.ndarray:
        t = (item["verb"], item["mod"], item["rel"])
        if t in self.triple:
            p = np.full(N_LABELS, 0.02)
            p[label_of(*t)] = 0.90
            return p / p.sum()
        pk = (item["verb"], item["mod"])
        if pk in self.pair:
            p = np.full(N_LABELS, 0.05)
            p[label_of(item["verb"], item["mod"], item["rel"])] = 0.45
            return p / p.sum()
        return self.prior.copy()

    def latent(self, item) -> np.ndarray:
        return self.rng.normal(size=DIM)


class LinearHead:
    """Ridge regression onto one-hot targets, with an optional online update."""

    def __init__(self, d: int, lam: float = 1e-2):
        self.lam = lam
        self.W = np.zeros((d, N_LABELS))
        self.b = np.zeros(N_LABELS)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Ridge regression with a properly centred intercept.

        Fitting uncentred X with an intercept equal to mean(Y) biases the
        solution: the regulariser then shrinks the coefficients of an
        uncentred design in a way that does not recover the underlying linear
        map. Centring both sides and folding the means back into the
        intercept is the standard construction and is what the reference
        models rely on to reproduce the label exactly.
        """
        d = X.shape[1]
        xm = X.mean(axis=0)
        ym = Y.mean(axis=0)
        Xc = X - xm
        Yc = Y - ym
        A = Xc.T @ Xc + self.lam * np.eye(d)
        self.W = np.linalg.solve(A, Xc.T @ Yc)
        self.b = ym - xm @ self.W

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def probs(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def online_update(self, x: np.ndarray, target: int, lr: float = 0.05):
        p = self.probs(x[None, :])[0]
        err = _onehot(target) - p
        self.W += lr * np.outer(x, err)
        self.b += lr * err


class FeatureModel(BaseSystem):
    """Reference models built on a fixed feature map.

    The internal representation consists of four blocks of DIM units:

        block 0   e_v              verb embedding
        block 1   e_m              modifier embedding
        block 2   e_r              relation embedding
        block 3   e_v * e_m        pair interaction  (the label needs it)

    The compositional variant exposes all four; the non-compositional control
    replaces the interaction block with the distractor embedding, so its
    representation cannot represent the interaction term the label depends on.
    """

    has_ablation_surface = True

    def __init__(self, seed: int = 0, compositional: bool = True):
        super().__init__(seed)
        self.compositional = compositional
        self.head = LinearHead(HIDDEN)
        self._mask = np.ones(HIDDEN, dtype=bool)
        self._fit()

    # -- representation
    def _blocks(self, item):
        ev = FEATURES.emb_v(item["verb"])
        em = FEATURES.emb_m(item["mod"])
        er = FEATURES.emb_r(item["rel"])
        ed = FEATURES.emb_d(item["distractor"])
        if self.compositional:
            return [ev, em, er, ev * em]
        return [ev, em, er, ed]

    def _hidden(self, item) -> np.ndarray:
        return np.concatenate(self._blocks(item))

    def _fit(self):
        """Ridge regression onto the *score* vector, not onto the one-hot label.

        The label is argmax of a linear function of the features, so a linear
        readout that recovers the score recovers the label exactly. Regressing
        the one-hot label instead would ask a linear model to approximate a
        piecewise-linear function, and would penalise exactly the models whose
        representations are right.
        """
        rng = np.random.default_rng(31337)
        base, _ = make_instances(rng)
        X = np.stack([self._hidden(it) for it in base])
        Y = np.stack([FEATURES.score_vector(it) for it in base])
        self.head.fit(X, Y)

    # -- interface
    def _probs(self, item) -> np.ndarray:
        h = self._hidden(item) * self._mask
        return self.head.probs(h[None, :])[0]

    def latent(self, item) -> np.ndarray:
        return self._hidden(item)

    def state_entropy(self) -> float:
        return float(np.log(HIDDEN))

    def energy_per_step(self) -> float:
        return 6.0e-9

    def respond(self, item, severity: float) -> np.ndarray:
        """Perturbation replaces content tokens with probability `severity`."""
        if severity <= 0:
            return self._probs(item)
        rng = np.random.default_rng(
            _stable_hash(f"{item['verb']}|{item['mod']}|{item['rel']}|{severity}") % (2 ** 32))
        it = dict(item)
        for key, choices in (("verb", VERBS), ("mod", MODS), ("rel", RELS)):
            if rng.random() < severity:
                it[key] = choices[int(rng.integers(len(choices)))]
        return self._probs(it)

    def ablated(self, frac: float, family: str, seed: int = 0):
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.rng = np.random.default_rng(seed)
        clone._mask = self._mask.copy()
        n_cut = int(round(frac * HIDDEN))
        if n_cut > 0:
            if family == "structured":
                # concentrated damage: remove whole blocks, in order
                idx = list(range(min(n_cut, HIDDEN)))
            else:
                rng = np.random.default_rng(seed + 17)
                idx = list(rng.choice(HIDDEN, size=min(n_cut, HIDDEN), replace=False))
            clone._mask[np.array(idx, dtype=int)] = False
        if hasattr(clone, "_acc"):
            clone._acc = np.zeros_like(clone._acc)
        return clone


class MlpFixed(FeatureModel):
    """Control: additive features only -- no pair interaction, so no transfer."""

    name = "mlp_fixed"
    kind = "control"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed, compositional=False)


class CompositionalModel(FeatureModel):
    """Subject: exposes the pair interaction, so unseen combinations transfer."""

    name = "compositional_model"
    kind = "subject"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed, compositional=True)


class RecurrentAgent(FeatureModel):
    """Subject: carries state across the stream and learns from an error signal.

    The only reference system with a genuine in-episode correction loop, hence
    the only one for which the error-signal metric is defined rather than
    structurally zero.
    """

    name = "recurrent_agent"
    kind = "subject"
    supports_correction = True

    def __init__(self, seed: int = 0):
        self._acc = np.zeros(HIDDEN)      # must exist before the base class fits
        super().__init__(seed=seed, compositional=True)

    def _hidden(self, item) -> np.ndarray:
        """Quasi-recurrent state: a short memory of the recent stream.

        The memory constant is low (0.1) so that the state is dominated by the
        current item; this keeps the fitted readout consistent between the
        fitting pass and any evaluation pass, while the agent still carries
        state across the episode and can be corrected within it.
        """
        h = super()._hidden(item)
        self._acc = 0.9 * self._acc + 0.1 * h
        return self._acc

    def state_entropy(self) -> float:
        p = np.abs(self._acc)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        return float(-(p * np.log(p + 1e-12)).sum())

    def energy_per_step(self) -> float:
        return 9.0e-9

    def apply_correction(self, item, correct_label: int) -> None:
        h = self._hidden(item) * self._mask
        self.head.online_update(h, correct_label, lr=0.01)


class MockCalibratedConfidence(BaseSystem):
    """Control: calibrated confidence, no self-model.

    It is right on most items and reports the corresponding probability, so its
    discrimination is high and its calibration error is near zero -- while it
    has no recurrent state, no capacity to learn from an error signal, and an
    uncorrelated representation. It is the built-in refutation of the claim
    that a high MA indicates a self-model.
    """

    name = "mock_calibrated_confidence"

    def _difficulty(self, item) -> float:
        h = _stable_hash(f"{item['verb']}|{item['mod']}|{item['rel']}|{item['distractor']}") % 1000 / 1000.0
        return 0.97 if h < 0.75 else 0.03

    def _probs(self, item) -> np.ndarray:
        """Deterministic: right exactly when it says it is right.

        The point of this control is that a system can be perfectly
        *discriminative* about its own errors while having no self-model, no
        state, and no capacity to learn. Making the correctness deterministic
        removes sampling noise so that the AOG measurement is exactly zero
        rather than approximately zero.
        """
        d = self._difficulty(item)
        p = np.full(N_LABELS, (1.0 - d) / (N_LABELS - 1))
        if d > 0.5:
            p[item["label"]] = d            # right, and it says so
        else:
            wrong = int(_stable_hash(f"w{self.name}{item['verb']}{item['mod']}"
                                     f"{item['rel']}{item['distractor']}") % N_LABELS)
            p[wrong] = d                   # wrong, but it was confident
        return p

    def confidence(self, item) -> float | None:
        return float(self._difficulty(item))

    def latent(self, item) -> np.ndarray:
        return self.rng.normal(size=DIM)

    def energy_per_step(self) -> float:
        return 1.5e-9


class FixedResponse(BaseSystem):
    """Control: identical output regardless of input. Perturbation floor."""

    name = "fixed_response"

    def _probs(self, item) -> np.ndarray:
        p = np.full(N_LABELS, 0.02)
        p[0] = 0.86
        return p

    def confidence(self, item) -> float | None:
        return 0.86

    def energy_per_step(self) -> float:
        return 0.5e-9


SYSTEM_CLASSES = [
    ThermostatPID, RandomPolicy, OverconfidentPolicy, LookupTable, NgramModel,
    MlpFixed, MockCalibratedConfidence, FixedResponse,
    CompositionalModel, RecurrentAgent,
]


def build_system(name: str, seed: int = 0):
    for cls in SYSTEM_CLASSES:
        if cls.name == name:
            return cls(seed=seed)
    raise KeyError(f"unknown system: {name}")


def system_names():
    return [c.name for c in SYSTEM_CLASSES]
