"""Adapter for real language models served locally by Ollama.

This module lets the Protennoia Protocol measure actual large language models
rather than only the analytic reference systems. It exists because a protocol
validated only on surrogates validates the instrument, not the question.

Confidence source
-----------------
Confidence is taken from the model's own token log-probabilities over the
answer options, obtained from a single forward pass with the answer
constrained to one option letter. This is the source the protocol calls
"permitted": it is derived from the model's processing, it cannot see the
ground truth, and it is not a post-hoc self-report. Thinking mode is disabled
so that the first generated token is the option letter.

Task
----
A slot-binding composition task. The query states three slots (verb,
modifier, relation) in a structured form; the options are natural-language
sentences with the same slots filled in. The correct option is the one whose
slots match the query, so solving the item requires binding the three query
slots into the option template rather than retrieving a stored association.
Base and novel splits differ in which combinations occur.

Latents
-------
The internal representation is taken from the Ollama embedding endpoint for
the query text. CCE is then computed with the same estimator used for the
reference systems, so the two families are directly comparable.

Energy
------
Ollama reports evaluation counts and durations, so throughput in tokens per
second is measurable. Absolute joules are not measurable on this host, and
the protocol's energy metric is therefore reported as a throughput proxy, with
the substitution stated rather than hidden.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import urllib.error

import numpy as np

from . import metrics as M
from .systems import (MODS, NOVEL_TYPE_B, RELS, VERBS, BaseSystem, make_instances)

DEFAULT_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OPTION_LETTERS = "ABCDEFGH"

# The eight slot combinations used as distractors for every item, together with
# the held-out design of the LLM track.
ALL_TRIPLES = [(v, m, r) for v in VERBS for m in MODS for r in RELS]
NOVEL_TRIPLES = sorted(set(NOVEL_TYPE_B) | {(v, m, r) for (v, m) in
                                             [("jump", "green"), ("walk", "blue")]
                                             for r in RELS})


def sentence(v: str, m: str, r: str) -> str:
    return f"The {m} one is {r} the one that {v}s."


def query_text(v: str, m: str, r: str) -> str:
    return f"verb: {v}; modifier: {m}; relation: {r}"


class OllamaError(RuntimeError):
    pass


def _normalise_url(url: str) -> str:
    """Accept 'host', 'host:port', 'http://host:port' and the 0.0.0.0 form."""
    u = (url or "").strip()
    if not u:
        u = "http://127.0.0.1:11434"
    if "://" not in u:
        u = "http://" + u
    u = u.replace("//0.0.0.0", "//127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    scheme, _, rest = u.partition("://")
    host, _, _port = rest.partition(":")
    if ":" not in rest:
        rest = f"{host}:11434"
    return f"{scheme}://{rest}"


class OllamaClient:
    """Minimal stdlib client for the Ollama HTTP API."""

    def __init__(self, url: str = DEFAULT_URL, timeout: int = 300):
        self.url = _normalise_url(url).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def tags(self):
        try:
            return self._post("/api/tags", {}) if False else json.loads(
                urllib.request.urlopen(self.url + "/api/tags", timeout=60).read().decode())
        except Exception as exc:  # pragma: no cover - diagnostics only
            raise OllamaError(f"cannot reach Ollama at {self.url}: {exc}") from exc

    def chat_logprobs(self, model: str, prompt: str, top_logprobs: int = 20,
                      num_predict: int = 2, think: bool = False):
        """One chat call; returns (first_token, {token: logprob}, usage)."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "options": {"num_predict": num_predict, "temperature": 0.0},
        }
        if not think:
            payload["think"] = False
        try:
            resp = self._post("/api/chat", payload)
        except urllib.error.HTTPError:
            # models without a thinking mode may reject the field
            payload.pop("think", None)
            resp = self._post("/api/chat", payload)
        lps = resp.get("logprobs") or []
        first = lps[0] if lps else None
        dist = {}
        if first:
            for entry in first.get("top_logprobs", []):
                tok = entry.get("token", "").strip()
                if tok:
                    dist[tok] = max(dist.get(tok, -1e9), float(entry.get("logprob", -1e9)))
        usage = {
            "eval_count": resp.get("eval_count", 0),
            "eval_duration_ns": resp.get("eval_duration", 0),
            "prompt_eval_count": resp.get("prompt_eval_count", 0),
        }
        return (first or {}).get("token", ""), dist, usage

    def embed(self, model: str, inputs):
        payload = {"model": model, "input": inputs}
        resp = self._post("/api/embed", payload)
        return resp.get("embeddings", [])


class OllamaModel(BaseSystem):
    """A real LLM measured under the same protocol as the reference systems."""

    kind = "subject"
    supports_correction = False
    has_ablation_surface = False

    def __init__(self, model: str, seed: int = 0, client: OllamaClient | None = None,
                 embed_model: str | None = None, temperature: float = 0.0):
        super().__init__(seed)
        self.seed = seed
        self.model = model
        self.name = f"ollama:{model}"
        self.client = client or OllamaClient()
        self.embed_model = embed_model or model
        self.temperature = temperature
        self._tokens = 0
        self._nanos = 0
        self._prompts = 0
        self._cache: dict = {}

    # -- item construction ------------------------------------------------
    def _options(self, item, rng):
        """Eight options: the correct triple plus seven distractors, shuffled."""
        correct = (item["verb"], item["mod"], item["rel"])
        pool = [t for t in ALL_TRIPLES if t != correct]
        idx = rng.choice(len(pool), size=7, replace=False)
        opts = [pool[i] for i in idx] + [correct]
        order = rng.permutation(len(opts))
        shuffled = [opts[i] for i in order]
        return shuffled, shuffled.index(correct)

    def _prompt(self, item, opts):
        lines = [
            "A system is described by three slots: a verb, a modifier and a relation.",
            "Choose the sentence that uses exactly those three slot values.",
            "",
            f"verb: {item['verb']}",
            f"modifier: {item['mod']}",
            f"relation: {item['rel']}",
            "",
            "Options:",
        ]
        for letter, (v, m, r) in zip(OPTION_LETTERS, opts):
            lines.append(f"{letter}. {sentence(v, m, r)}")
        lines.append("")
        lines.append("Answer with the single letter of the correct option.")
        return "\n".join(lines)

    # -- interface --------------------------------------------------------
    def _probs(self, item):
        rng = np_random(self.seed, item)
        opts, correct = self._options(item, rng)
        prompt = self._prompt(item, opts)
        token, dist, usage = self.client.chat_logprobs(self.model, prompt)
        self._tokens += usage.get("eval_count", 0) or 0
        self._nanos += usage.get("eval_duration_ns", 0) or 0
        self._prompts += 1

        probs = {letter: 0.0 for letter in OPTION_LETTERS}
        for tok, lp in dist.items():
            key = tok.strip().strip(".)").strip()
            if key in probs:
                probs[key] = math.exp(lp)
        total = sum(probs.values())
        if total <= 0:
            # fall back to the emitted token if the top-k contained no letter
            key = (token or "").strip().strip(".)").strip()
            vec = np.zeros(len(OPTION_LETTERS))
            vec[OPTION_LETTERS.index(key) if key in OPTION_LETTERS else 0] = 1.0
            return vec
        vec = np.array([probs[l] for l in OPTION_LETTERS])
        return vec / vec.sum()

    def _label_of_options(self, item):
        rng = np_random(self.seed, item)
        opts, correct = self._options(item, rng)
        return opts, correct

    def answer(self, item):
        """Predict the option index among the shuffled options."""
        opts, correct = self._label_of_options(item)
        probs = self._probs(item)
        return probs

    def confidence(self, item):
        probs = self._probs(item)
        return float(np.max(probs))

    def latent(self, item):
        key = (item["verb"], item["mod"], item["rel"], item["distractor"])
        if key not in self._cache:
            texts = [query_text(item["verb"], item["mod"], item["rel"])]
            try:
                self._cache[key] = np.array(self.client.embed(self.embed_model, texts)[0], dtype=float)
            except Exception:
                self._cache[key] = np.zeros(8)
        return self._cache[key]

    def state_entropy(self) -> float:
        return float(np.log(max(len(self._cache), 1)))

    def energy_per_step(self) -> float:
        """Throughput proxy: inverse tokens-per-second, normalised.

        Not joules: this host cannot measure energy. The substitution is declared
        so that the value is never read as an energy measurement.
        """
        if self._nanos <= 0 or self._tokens <= 0:
            return 1e-9
        tps = self._tokens / (self._nanos / 1e9)
        return 1.0 / max(tps, 1e-6) * 1e-9

    def throughput(self) -> float:
        if self._nanos <= 0:
            return 0.0
        return self._tokens / (self._nanos / 1e9)

    def respond(self, item, severity: float) -> np.ndarray:
        return self._probs(item)


def np_random(seed: int, item):
    """Deterministic per-item RNG so that options are stable across calls."""
    import numpy as np
    key = f"{seed}|{item['verb']}|{item['mod']}|{item['rel']}|{item['distractor']}"
    h = zlib_crc32(key)
    return np.random.default_rng(h)


def zlib_crc32(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


# --------------------------------------------------------------- runner ---
def evaluate_llm(model: str, client: OllamaClient, n_ma: int = 120,
                 n_cgs_base: int = 60, n_cgs_novel: int = 60, seed: int = 0,
                 embed_model: str | None = None) -> dict:
    """Measure one LLM under the protocol. Same metric code as the surrogates."""
    rng = np_random(seed, {"verb": "jump", "mod": "red", "rel": "above", "distractor": "d0"})
    base, novel = make_instances(rng)
    base = [it for it in base if (it["verb"], it["mod"], it["rel"]) not in NOVEL_TRIPLES]
    novel = [it for it in novel if (it["verb"], it["mod"], it["rel"]) in NOVEL_TRIPLES]
    rng.shuffle(base)
    rng.shuffle(novel)

    system = OllamaModel(model, seed=seed, client=client, embed_model=embed_model)

    # ---- MA over the full task distribution (reduced n for local models) --
    ma_items = (base + novel)[:n_ma]
    conf, correct, logits_rows = [], [], []
    for it in ma_items:
        opts, correct_idx = system._label_of_options(it)
        probs = system._probs(it)
        logits_rows.append(probs)
        conf.append(float(np.max(probs)))
        correct.append(int(int(np.argmax(probs)) == correct_idx))
    ma = M.compute_ma(np.array(conf), np.array(correct), seed=seed)

    # ---- CGS ------------------------------------------------------------
    def acc_and_nll(items):
        c = 0
        nll = 0.0
        for it in items:
            opts, correct_idx = system._label_of_options(it)
            probs = system._probs(it)
            c += int(int(np.argmax(probs)) == correct_idx)
            nll += -float(math.log(max(probs[correct_idx], 1e-12)))
        n = max(len(items), 1)
        return c / n, nll / n

    acc_base, nll_base = acc_and_nll(base[:n_cgs_base])
    acc_novel, nll_novel = acc_and_nll(novel[:n_cgs_novel])
    cgs = M.compute_cgs(acc_base, acc_novel, nll_base, nll_novel)

    # ---- CCE from the model's own embeddings ----------------------------
    probe_items = (base + novel)[:120]
    try:
        texts = [query_text(it["verb"], it["mod"], it["rel"]) for it in probe_items]
        embs = client.embed(embed_model or model, texts)
        z = np.array(embs, dtype=float)
        # The covariance-based estimator needs n > d + 2; embedding models
        # produce far more dimensions than the number of items, so the latent is
        # projected onto the leading principal components first.
        k = int(min(32, z.shape[0] - 3, z.shape[1]))
        if k >= 2:
            zc = z - z.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(zc, full_matrices=False)
            z = zc @ vt[:k].T
        labels = []
        for it in probe_items:
            # The class label is the content of the item (the canonical triple),
            # not the position of the correct option, which is shuffled per item
            # and therefore carries no information about the item.
            labels.append(ALL_TRIPLES.index((it["verb"], it["mod"], it["rel"])))
        labels = np.array(labels)
        # The covariance estimator needs more samples per class than dimensions.
        _, counts = np.unique(labels, return_counts=True)
        k = int(max(2, min(16, counts.min() - 2)))
        if z.shape[1] > k:
            zc = z - z.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(zc, full_matrices=False)
            z = zc @ vt[:k].T
        cce = M.compute_cce(z, labels)
    except Exception as exc:
        cce = {"cce": None, "applicable": False, "note": f"embedding failed: {exc}"}

    return {
        "system": system.name,
        "kind": "subject",
        "seed": seed,
        "protocol_version": "0.1",
        "implementation_version": "0.1.0",
        "metrics": {
            "ma": ma,
            "aog": {"aog": 0.0, "applicable": False,
                    "note": "no in-episode correction loop for a served LLM"},
            "cgs": cgs,
            "cce": cce,
            "ablation": {"applicable": False,
                         "note": "no ablation surface exposed by the serving API"},
            "pct": {"pct": None, "applicable": False,
                    "note": "perturbation probe not defined for a served model"},
            "eib": {"eib": None, "applicable": False,
                    "throughput_tokens_per_s": system.throughput()},
        },
        "n_ma_items": len(ma_items),
        "n_cgs_base": min(n_cgs_base, len(base)),
        "n_cgs_novel": min(n_cgs_novel, len(novel)),
        "notes": [
            "Confidence is the model's own token log-probability over the option letters.",
            "Energy is a throughput proxy, not joules: absolute energy is not measurable on this host.",
            "Ablation and perturbational-complexity metrics are not applicable to a served model.",
        ],
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="protennoia.ollama")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--n-ma", type=int, default=120)
    ap.add_argument("--n-cgs-base", type=int, default=60)
    ap.add_argument("--n-cgs-novel", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ollama_matrix.json")
    a = ap.parse_args(argv)

    client = OllamaClient()
    out = {"protocol_version": "0.1", "implementation_version": "0.1.0",
           "systems": {}, "config": vars(a)}
    for m in a.models:
        print(f"[protennoia] measuring {m} ...", flush=True)
        try:
            out["systems"][f"ollama:{m}"] = evaluate_llm(
                m, client, n_ma=a.n_ma, n_cgs_base=a.n_cgs_base,
                n_cgs_novel=a.n_cgs_novel, seed=a.seed, embed_model=a.embed_model)
            r = out["systems"][f"ollama:{m}"]["metrics"]
            print(f"[protennoia] {m}: MA={r['ma'].get('auc')} ECE={r['ma'].get('ece')} "
                  f"CGS={r['cgs'].get('cgs')} CCE={r['cce'].get('cce')} "
                  f"tps={out['systems'][f'ollama:{m}']['metrics']['eib'].get('throughput_tokens_per_s')}",
                  flush=True)
        except Exception as exc:
            print(f"[protennoia] {m} FAILED: {exc}", flush=True)
    path = a.out
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print(f"[protennoia] wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
