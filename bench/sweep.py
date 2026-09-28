"""Offline threshold sweep: hit rate vs false-hit rate, 0.80 -> 0.99. Free (no API calls).

Why offline is legitimate: given the embeddings, the gateway's hit/miss decision
is deterministic — nearest cached prompt, compare to threshold, store on miss.
`simulate()` replays exactly that policy with exact (brute-force) cosine search.
The only difference from production is that pgvector's HNSW index is approximate;
bench/live.py measures how often the two actually agree.

    python -m bench.sweep --pairs 2000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.embeddings import SentenceTransformerEmbedder

from .workload import QQP, Request

RESULTS = Path(__file__).parent.parent / "results"


def embed_all(texts: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict[str, np.ndarray]:
    emb = SentenceTransformerEmbedder(model)
    unique = sorted(set(texts))
    vecs = emb.model.encode(unique, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    return dict(zip(unique, vecs.astype(np.float32)))


def simulate(stream: list[Request], vecs: dict[str, np.ndarray], threshold: float, verify=None):
    """Replay the gateway's cache policy. Returns one (hit, similarity, matched_text) per request.

    `verify(query, candidate) -> bool`, if given, is a second-stage check that must
    also pass before a candidate above `threshold` is served (retrieve-then-verify).
    """
    dim = next(iter(vecs.values())).shape[0]
    cache = np.empty((len(stream), dim), dtype=np.float32)
    cached_texts: list[str] = []
    out = []
    for req in stream:
        q = vecs[req.text]
        n = len(cached_texts)
        if n:
            sims = cache[:n] @ q                       # unit vectors: dot product == cosine similarity
            j = int(np.argmax(sims))
            best = float(sims[j])
            if best >= threshold and (verify is None or verify(req.text, cached_texts[j])):
                out.append((True, best, cached_texts[j]))
                continue                               # hit: nothing is stored
        cache[n] = q                                   # miss: provider answers, answer is stored
        cached_texts.append(req.text)
        out.append((False, best if n else None, None))
    return out


def score(stream: list[Request], decisions, qqp: QQP) -> dict:
    hits = correct = wrong = unlabeled = 0
    dup_followups = dup_caught = 0
    for req, (hit, _, matched) in zip(stream, decisions):
        if req.role == "followup" and req.is_dup:
            dup_followups += 1
        if not hit:
            continue
        hits += 1
        verdict = qqp.judge(req.text, matched)
        correct += verdict == "correct"
        wrong += verdict == "wrong"
        unlabeled += verdict == "unlabeled"
        if req.role == "followup" and req.is_dup and verdict == "correct":
            dup_caught += 1
    false_hits = wrong + unlabeled
    return {
        "requests": len(stream),
        "hits": hits,
        "hit_rate": hits / len(stream),
        "correct_hits": correct,
        "false_hits": false_hits,
        "false_hits_unlabeled": unlabeled,
        # Of the answers served from cache, what fraction were for a different question.
        # Reported as a range: QQP only labels the pairs it contains, so a hit on an
        # unlabeled pair is unknown. Lower bound counts only human-labeled mismatches;
        # upper bound also counts every unlabeled hit as wrong.
        "false_hit_rate_low": wrong / hits if hits else 0.0,
        "false_hit_rate_high": false_hits / hits if hits else 0.0,
        # Of follow-ups that humans say ARE paraphrases, what fraction we served from cache.
        "dup_recall": dup_caught / dup_followups if dup_followups else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qqp = QQP()
    stream = QQP.stream(qqp.sample(args.pairs, args.seed), args.seed)
    t0 = time.time()
    vecs = embed_all([r.text for r in stream])
    print(f"embedded {len(vecs)} unique questions in {time.time() - t0:.1f}s")

    thresholds = [round(0.80 + 0.01 * i, 2) for i in range(20)]
    rows = []
    for t in thresholds:
        rows.append({"threshold": t, **score(stream, simulate(stream, vecs, t), qqp)})

    print(f"\n{args.pairs} QQP pairs -> {len(stream)} requests "
          f"({sum(r.role == 'followup' and r.is_dup for r in stream)} follow-ups are true paraphrases)\n")
    print("| threshold | hit rate | false-hit rate (labeled – conservative) | paraphrase recall | hits (wrong / unlabeled) |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['threshold']:.2f} | {r['hit_rate']:.1%} | {r['false_hit_rate_low']:.1%} – "
              f"{r['false_hit_rate_high']:.1%} | {r['dup_recall']:.1%} "
              f"| {r['hits']} ({r['false_hits'] - r['false_hits_unlabeled']} / {r['false_hits_unlabeled']}) |")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sweep.json"
    out.write_text(json.dumps({"pairs": args.pairs, "seed": args.seed, "requests": len(stream),
                               "embedding_model": "all-MiniLM-L6-v2", "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
