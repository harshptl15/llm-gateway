"""Offline experiment: can we beat embedding-only caching? Free (no API calls).

Baseline: MiniLM nearest neighbour, serve if cosine >= t.
Challengers:
  1. a different embedder (bge-small-en-v1.5), same policy;
  2. retrieve-then-verify: MiniLM finds the nearest cached prompt with a LOWER
     cosine bar (more candidates -> more recall), then a cross-encoder reads the
     query and the candidate TOGETHER and must also approve it (precision).

A bi-encoder (MiniLM) embeds each question alone, so "become an introvert" and
"become less of an introvert" land close together. A cross-encoder attends across
both texts at once, so it can notice that one small word flips the meaning. It's
too slow to compare against every cached prompt, but cheap on the single
candidate the vector search already found.

    python -m bench.improve --pairs 2000
"""
import argparse
import json
import time

from sentence_transformers import CrossEncoder

from .sweep import RESULTS, embed_all, score, simulate
from .workload import QQP

CROSS_ENCODERS = {
    # Trained on STS-B (general sentence similarity) — has NOT seen Quora data.
    "stsb": "cross-encoder/stsb-distilroberta-base",
    # Trained on Quora duplicate questions (train split) — the right task, but
    # in-domain for this benchmark, so its numbers here are optimistic.
    "quora": "cross-encoder/quora-distilroberta-base",
}


class MemoVerifier:
    """Cross-encoder score with memoization (the same pair recurs across configs)."""

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name, device="cpu")
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, a: str, b: str) -> float:
        key = (a, b)
        if key not in self.cache:
            self.cache[key] = float(self.model.predict([(a, b)], show_progress_bar=False)[0])
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qqp = QQP()
    stream = QQP.stream(qqp.sample(args.pairs, args.seed), args.seed)
    texts = [r.text for r in stream]
    rows = []

    def run(name, vecs, t, verify=None, **extra):
        r = {"config": name, "retrieval_threshold": t, **extra, **score(stream, simulate(stream, vecs, t, verify), qqp)}
        rows.append(r)
        print(f"{name:<28} t={t:.2f} {str(extra.get('verify_threshold', '')):>5}  hit {r['hit_rate']:6.1%}  "
              f"false {r['false_hit_rate_low']:5.1%}-{r['false_hit_rate_high']:5.1%}  recall {r['dup_recall']:5.1%}",
              flush=True)

    minilm = embed_all(texts)
    for t in (0.90, 0.95, 0.97):
        run("minilm (baseline)", minilm, t)

    bge = embed_all(texts, "BAAI/bge-small-en-v1.5")
    for t in (0.90, 0.93, 0.95, 0.97):
        run("bge-small", bge, t)

    for key, model_name in CROSS_ENCODERS.items():
        v = MemoVerifier(model_name)
        t0 = time.time()
        for rt in (0.80, 0.85, 0.90):
            for vt in (0.5, 0.7, 0.8, 0.9):
                run(f"minilm + verify[{key}]", minilm, rt, lambda q, c: v.score(q, c) >= vt, verify_threshold=vt)
        n = len(v.cache)
        print(f"  {key}: {n} unique pairs scored, {1000 * (time.time() - t0) / max(n, 1):.1f} ms/pair incl. simulation\n")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "improve.json").write_text(json.dumps({"pairs": args.pairs, "seed": args.seed, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
