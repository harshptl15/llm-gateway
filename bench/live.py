"""Live benchmark through the running gateway (docker compose up). COSTS REAL MONEY.

Pass 1 (measured): the QQP request stream, sent one request at a time so each
    latency is the request itself, not queueing behind others. Gives hit rate,
    uncached latency, and real $ spent/saved from the gateway's usage_log.
Pass 2 (replay, free): the same queries again. Everything is cached by now, so
    this yields ~N extra cache-hit latency samples at $0 — enough for a stable p99.
    Replay hits are NOT counted in hit rate or savings.

    python -m bench.live --pairs 300          # prints an estimate and asks before spending
    python -m bench.live --pairs 300 --analyze-existing   # recover an interrupted run; replay is free
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import asyncpg
import httpx
import numpy as np
import redis

from .sweep import RESULTS, embed_all, simulate
from .workload import QQP

RAW = Path(__file__).parent / "data" / "live_raw.json"  # contains QQP text -> gitignored
DB_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway@localhost:5432/gateway")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def pct(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"n": len(values), "p50": round(float(p50), 1), "p95": round(float(p95), 1), "p99": round(float(p99), 1)}


async def reset_db():
    conn = await asyncpg.connect(DB_URL)
    await conn.execute("TRUNCATE semantic_cache, usage_log")
    await conn.close()


async def read_ledger(key_id: str) -> list[dict]:
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        """SELECT u.cache_hit, u.similarity, u.cost_usd, u.saved_usd, u.latency_ms,
                  u.input_tokens, u.output_tokens, c.prompt AS matched
           FROM usage_log u LEFT JOIN semantic_cache c ON c.id = u.cache_entry_id
           WHERE u.key_id = $1 ORDER BY u.id""",
        key_id,
    )
    await conn.close()
    return [dict(r) for r in rows]


def send(client: httpx.Client, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = client.post("/v1/complete", json={"prompt": prompt, "max_tokens": max_tokens})
    ms = (time.perf_counter() - t0) * 1000
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    return {"status": r.status_code, "ms": ms, "cached": body.get("cached"), "similarity": body.get("similarity")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--sample-from", type=int, default=2000, help="take the first N pairs of the sweep's sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="bench-key-456")
    ap.add_argument("--key-id", default="bench", help="the name GATEWAY_API_KEYS maps --key to")
    ap.add_argument("--provider-rpm", type=float, default=45, help="cap on provider calls/min (Anthropic tier 1 = 50)")
    ap.add_argument("--replay-rps", type=float, default=8, help="replay pacing; stays under the gateway's 600/min limit")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    ap.add_argument("--analyze-existing", action="store_true",
                    help="don't send pass 1; recover it from usage_log (e.g. after the run was interrupted)")
    args = ap.parse_args()

    client = httpx.Client(base_url=args.url, headers={"x-api-key": args.key}, timeout=120)
    health = client.get("/health").json()
    threshold, model = health["similarity_threshold"], health["model"]

    qqp = QQP()
    pairs = qqp.sample(args.sample_from, args.seed)[: args.pairs]
    stream = QQP.stream(pairs, args.seed)
    vecs = embed_all([r.text for r in stream])
    predicted = simulate(stream, vecs, threshold)
    est_misses = sum(not hit for hit, _, _ in predicted)
    # Haiku 4.5: ~20 input tokens/question at $1/M, up to max_tokens output at $5/M.
    est_cost = est_misses * (20 * 1.0 + args.max_tokens * 5.0) / 1e6
    print(f"gateway: model={model} threshold={threshold}")
    print(f"{len(stream)} requests, ~{est_misses} predicted misses -> est. cost <= ${est_cost:.2f}, "
          f"~{est_misses / args.provider_rpm:.0f} min (provider calls capped at {args.provider_rpm:.0f}/min)")
    if args.analyze_existing:
        # Recover an interrupted run from the gateway's own ledger instead of
        # re-spending. Rows are in request order; once the provider started failing,
        # only cache hits still got logged, so the clean prefix ends at the last miss.
        ledger = asyncio.run(read_ledger(args.key_id))
        k = max(i for i, r in enumerate(ledger) if not r["cache_hit"]) + 1
        measured, led1, client1 = stream[:k], ledger[:k], None  # client timings weren't saved
        stopped_at, errors, skip = k, None, len(ledger)
        print(f"recovered the first {k} requests from usage_log ({len(ledger) - k} later rows ignored)")
    else:
        if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
            return
        asyncio.run(reset_db())
        redis.Redis.from_url(REDIS_URL).flushdb()

        # ---- pass 1: measured
        pass1, fails = [], 0
        min_gap = 60.0 / args.provider_rpm
        t_start = time.time()
        for i, req in enumerate(stream):
            rec = send(client, req.text, args.max_tokens)
            pass1.append(rec)
            fails = fails + 1 if rec["status"] != 200 else 0
            if fails >= 10:  # e.g. provider out of credit: stop instead of blasting through the stream
                print(f"  aborting at request {i + 1}: 10 consecutive errors (last status {rec['status']})")
                break
            if rec["status"] == 200 and not rec["cached"]:
                time.sleep(max(0.0, min_gap - rec["ms"] / 1000))  # pacing happens OUTSIDE the timed region
            if (i + 1) % 50 == 0:
                hits = sum(bool(r["cached"]) for r in pass1)
                print(f"  pass 1: {i + 1}/{len(stream)}  hits={hits}  elapsed={time.time() - t_start:.0f}s", flush=True)
        # Failed requests raise before logging, so ledger rows line up with the 200s.
        ok = [(req, rec) for req, rec in zip(stream, pass1) if rec["status"] == 200]
        measured, client1 = [req for req, _ in ok], [rec["ms"] for _, rec in ok]
        led1 = asyncio.run(read_ledger(args.key_id))
        assert len(led1) == len(measured), (len(led1), len(measured))
        stopped_at = len(pass1) if len(pass1) < len(stream) else None
        errors, skip = len(pass1) - len(ok), len(led1)

    # ---- pass 2: replay the measured requests for cache-hit latency samples (free)
    pass2 = []
    for req in measured:
        pass2.append(send(client, req.text, args.max_tokens))
        time.sleep(1.0 / args.replay_rps)
    ok2 = [rec for rec in pass2 if rec["status"] == 200]
    led2 = asyncio.run(read_ledger(args.key_id))[skip:]
    assert len(led2) == len(ok2), (len(led2), len(ok2))
    print(f"  pass 2 (replay): {sum(bool(r['cached']) for r in ok2)}/{len(pass2)} hits")

    verdicts = {"correct": 0, "wrong": 0, "unlabeled": 0}
    for req, row in zip(measured, led1):
        if row["cache_hit"]:
            verdicts[qqp.judge(req.text, row["matched"])] += 1
    hits = sum(r["cache_hit"] for r in led1)

    # Does the live gateway (HNSW, approximate) decide the same as the exact simulation?
    pred_by_idx = {id(req): hit for req, (hit, _, _) in zip(stream, predicted)}
    agree = sum(pred_by_idx[id(req)] == row["cache_hit"] for req, row in zip(measured, led1))

    spent = float(sum(r["cost_usd"] for r in led1))
    saved = float(sum(r["saved_usd"] for r in led1))
    miss_rows = [r for r in led1 if not r["cache_hit"]]
    avg_miss_cost = spent / len(miss_rows) if miss_rows else 0.0
    replay_pairs = [(rec["ms"], row["latency_ms"]) for rec, row in zip(ok2, led2) if row["cache_hit"]]

    results = {
        "model": model,
        "threshold": threshold,
        "pairs": args.pairs,
        "max_tokens": args.max_tokens,
        "stream_requests": len(stream),
        "requests": len(measured),
        "stopped_at": stopped_at,
        "errors": errors,
        "hits": hits,
        "hit_rate": hits / len(measured),
        "predicted_hit_rate_full_stream": sum(h for h, _, _ in predicted) / len(stream),
        "false_hit_rate_low": verdicts["wrong"] / hits if hits else 0.0,
        "false_hit_rate_high": (verdicts["wrong"] + verdicts["unlabeled"]) / hits if hits else 0.0,
        "hit_verdicts": verdicts,
        "sim_live_agreement": agree / len(measured),
        "latency_client_ms": {
            "uncached": pct([ms for ms, row in zip(client1, led1) if not row["cache_hit"]]) if client1 else {"n": 0},
            "cached_replay": pct([c for c, _ in replay_pairs]),
        },
        "latency_server_ms": {
            "uncached": pct([r["latency_ms"] for r in led1 if not r["cache_hit"]]),
            "cached_replay": pct([s for _, s in replay_pairs]),
        },
        # HTTP + serialization overhead on localhost: client-side minus server-side, same requests.
        "client_minus_server_ms_p50": round(float(np.median([c - s for c, s in replay_pairs])), 2) if replay_pairs else None,
        "cost_usd": {
            "spent": round(spent, 4),
            "saved": round(saved, 4),
            "saved_pct_of_uncached_spend": saved / (spent + saved) if spent + saved else 0.0,
            "avg_cost_per_miss": round(avg_miss_cost, 6),
            "projected_saved_per_1M_requests": round(1e6 * (hits / len(measured)) * avg_miss_cost, 2),
        },
        "replay_misses": sum(not r["cache_hit"] for r in led2) + (len(pass2) - len(ok2)),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "live.json").write_text(json.dumps(results, indent=2))
    RAW.write_text(json.dumps({"pass2": pass2}, indent=0))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
