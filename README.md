# LLM Gateway with Semantic Caching

[![tests](https://github.com/harshptl15/llm-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/harshptl15/llm-gateway/actions/workflows/tests.yml)

A FastAPI proxy in front of the Anthropic API. Clients call the gateway instead of the provider; the gateway
authenticates them, rate-limits them, and **semantically caches** responses — a new prompt that *means* the same
thing as a previously answered one is served from Postgres/pgvector instead of paying for another LLM call.

The point of this project is the measurement: how much a semantic cache actually saves, how fast a hit is, and —
the part most write-ups skip — **how often it serves an answer to a different question**, measured against
human-labeled ground truth.

## Results

- **Cache hits are ~100× faster and free:** 23 ms vs 2.3 s at p50 (gateway-side), $0 vs $0.00075 per request.
- **The hard part is correctness, not speed.** Against human labels, at a 0.90 similarity threshold (a common
  default) **17–31% of cache hits served an answer to a different question**, e.g. *"How do I become an
  introvert?"* matched *"How do I become **less of** an introvert?"* at 0.95. The gateway ships at **0.95**
  (11–23% false hits, catching 22% of paraphrases): precision over hit rate.
- **Savings depend on how repetitive your traffic is.** On QQP, which has no exact repeats, the cache served 3.4% of
  live requests. Treat that as a floor for paraphrase-only traffic, not a typical number.
- **The offline sweep is trustworthy:** it matched the live gateway on **384 / 384** hit/miss decisions, so the
  20-threshold sweep stands in for 20 live runs.
- **The cache degrades gracefully.** The live run stopped at request 384 of 600 when the API account ran out of
  credit. Requests that were cache hits kept being served; only misses failed.

### Live run: 384 real requests, claude-haiku-4-5, threshold 0.95

| Latency (gateway-side) | p50 | p95 | p99 | n |
|---|---:|---:|---:|---:|
| Cache miss (calls the AI) | 2,327 ms | 2,750 ms | 3,213 ms | 371 |
| Cache hit | 22.9 ms | 33.7 ms | 40.1 ms | 384 |
| **Speedup** | **102×** | 82× | 80× | |

Measured inside the gateway, from request received to response ready (`usage_log.latency_ms`), so both rows are timed the same way. Seen from the client over localhost, a cache hit is p50 28.2 ms / p99 49.3 ms (HTTP overhead: median 4.9 ms).

| Result | Value |
|---|---:|
| Requests measured | 384 (run stopped at request 384 of 600: API credit ran out) |
| Hit rate | 3.4% (13 / 384); simulation predicts 4.0% over all 600 |
| Wrong-question hits | 3 of 13 (small sample; see the sweep's 237 hits at this threshold) |
| Spent on the AI | $0.2784 |
| Saved by the cache | $0.0095 (3.3% of what it would have cost) |
| Avg cost per cache miss | $0.00075 |
| Live vs. offline-simulation agreement | 100% of hit/miss decisions |

### Threshold sweep: 4,000 requests (2,000 QQP pairs), all-MiniLM-L6-v2

| Threshold | Hit rate | False-hit rate (labeled – conservative) | Paraphrase recall |
|---:|---:|---:|---:|
| 0.80 | 24.7% | 25.8% – 45.2% | 70.8% |
| 0.81 | 23.6% | 25.0% – 43.9% | 69.2% |
| 0.82 | 22.1% | 23.4% – 41.6% | 67.3% |
| 0.83 | 20.9% | 23.2% – 41.3% | 64.1% |
| 0.84 | 19.3% | 22.9% – 40.0% | 60.4% |
| 0.85 | 18.2% | 22.0% – 38.7% | 57.9% |
| 0.86 | 16.7% | 21.4% – 37.6% | 54.0% |
| 0.87 | 15.4% | 20.5% – 36.4% | 50.7% |
| 0.88 | 14.3% | 19.7% – 34.6% | 48.2% |
| 0.89 | 13.0% | 17.9% – 32.2% | 45.2% |
| 0.90 | 12.0% | 16.7% – 31.2% | 42.1% |
| 0.91 | 10.8% | 15.7% – 30.0% | 38.8% |
| 0.92 | 9.8% | 14.2% – 28.2% | 35.6% |
| 0.93 | 8.3% | 13.2% – 26.7% | 30.4% |
| 0.94 | 7.0% | 12.5% – 26.0% | 25.5% |
| **0.95** | 5.9% | 11.4% – 23.2% | 21.8% |
| 0.96 | 4.8% | 9.3% – 18.7% | 18.4% |
| 0.97 | 3.5% | 7.0% – 14.1% | 13.6% |
| 0.98 | 2.6% | 4.8% – 12.4% | 9.5% |
| 0.99 | 1.7% | 1.5% – 6.1% | 5.3% |

Raw numbers: [`results/live.json`](results/live.json), [`results/sweep.json`](results/sweep.json). The tables
above are generated from them by [`bench/report.py`](bench/report.py).


## How it works

```
client ──x-api-key──▶ POST /v1/complete {"prompt", "max_tokens"}
  1. auth        static API keys (env), constant-time compare              → 401
  2. limits      Redis: fixed-window requests/min + daily token quota        → 429 + Retry-After
  3. embed       all-MiniLM-L6-v2 (384-d, CPU, in a threadpool)
     search      pgvector: ORDER BY embedding <=> $q LIMIT 1 (HNSW, cosine)
  4. HIT  (similarity ≥ threshold) → stored response, $0 provider cost
     MISS → Anthropic (behind a Provider interface) → store prompt + embedding + response
  5. ledger      one usage_log row per request: key, hit/miss, served entry, tokens, cost, $ saved, latency
```

| File | Role |
|---|---|
| [`app/main.py`](app/main.py) | App factory, the five-step request handler, `/v1/stats` |
| [`app/static/index.html`](app/static/index.html) | Dashboard + playground (no build step, no JS dependencies) |
| [`app/cache.py`](app/cache.py) | Nearest-neighbour lookup and insert (pgvector) |
| [`app/embeddings.py`](app/embeddings.py) | sentence-transformers wrapper; detects prompts the model would truncate |
| [`app/ratelimit.py`](app/ratelimit.py) | Redis rate limit + token quota |
| [`app/providers/`](app/providers/base.py) | `Provider` interface, Anthropic implementation, fake for tests |
| [`db/init.sql`](db/init.sql) | Schema: `semantic_cache` (HNSW index), `usage_log` |
| [`bench/`](bench/) | Workload, offline threshold sweep, live benchmark, report generator |

## Methodology

**Workload.** [Quora Question Pairs](https://huggingface.co/datasets/nyu-mll/glue) (GLUE QQP, validation split):
question pairs where human annotators labeled whether both ask the same thing (36.8% are duplicates). Sampled pairs
are interleaved into one request stream: each question's partner arrives at a random later point, with unrelated
traffic in between.

**Ground truth for a hit.** When the gateway serves prompt *A*'s cached answer to query *B*, the hit is **correct**
if *A* and *B* are identical or QQP labels them duplicates, and **wrong** if QQP labels them non-duplicates. If QQP
never labeled that pair, the hit is **unlabeled**. False-hit rates are reported as a range: the low end counts only
human-labeled mismatches; the high end also counts every unlabeled hit as wrong. Spot-checking suggests most
unlabeled hits are genuine paraphrases, so the truth is closer to the low end.

**Threshold sweep (offline, free).** Given the embeddings, the cache decision is deterministic, so
[`bench/sweep.py`](bench/sweep.py) replays the gateway's exact policy (nearest neighbour → compare → store on miss)
in numpy for 20 thresholds. The live run checks this against the real system: the simulation and the live gateway
(HNSW, approximate) made the same hit/miss decision on the percentage of requests shown in the results table.

**Live run.** [`bench/live.py`](bench/live.py) sends the stream (300 pairs = 600 requests, a subset of the
sweep's pairs) through the Dockerized gateway with real Claude Haiku 4.5 calls (`max_tokens=150`), **one request at a time**, so each latency sample is the request itself and not
time spent queueing. Provider calls are paced under Anthropic's tier-1 rate limit; the pacing sleep happens outside
the timed region. Hit rate, spend and savings come from the gateway's own `usage_log`, not from benchmark-side
bookkeeping. The stream is then **replayed** — every query is now cached — to collect a stable number of cache-hit
latency samples at no cost; the replay is excluded from hit rate and savings. This run was cut off at request 384
when the API account's credit ran out. The first 384 requests had all succeeded, so they were recovered from the
gateway's ledger (`--analyze-existing`) rather than re-bought; their client-side timings were lost with the killed
process, which is why the latency table uses gateway-side timing for both rows.

**Caveats.**
- QQP was built from *confusable* question pairs, so it's harder than typical traffic: false-hit rates here are
  pessimistic. It also has no exact repeats, which real FAQ/support traffic has many of, so the hit rate is
  conservative.
- QQP labels are noisy in both directions (e.g. a question and the same question with a typo, labeled "different").
- all-MiniLM-L6-v2's training data includes Quora duplicate pairs, so it may have seen some of these questions.
  If anything, that makes these numbers optimistic.
- Latency is measured on one laptop (Apple M4, Docker via OrbStack) over localhost; provider latency depends on
  Anthropic's load at the time.

## Design decisions

- **Postgres + pgvector instead of a dedicated vector DB.** Cache entries, the usage ledger, and the vectors live in
  one transactional store. At this scale a separate vector DB is another service to run for no measurable gain.
- **HNSW index.** An approximate nearest-neighbour graph: lookup cost grows roughly logarithmically instead of scanning
  every row. It can miss the true nearest neighbour; the sim-vs-live agreement number measures how often that
  mattered.
- **The similarity is returned and logged even on a miss.** That's what makes threshold tuning possible from
  production traffic.
- **Embedding runs in a threadpool.** `encode()` is CPU-bound and synchronous; calling it on the event loop would
  stall every in-flight request.
- **Prompts longer than the embedder's 256-token window bypass the cache.** MiniLM silently truncates, so two long
  prompts that differ only at the end would embed identically, a guaranteed false hit.
- **The cache is scoped by model.** An answer is only reused for the model that wrote it.
- **Fixed-window rate limiting** (`INCR` + `EXPIRE`, one pipelined round trip). Simple and atomic; allows up to 2×
  the limit in a burst straddling a window boundary. A sliding window or token bucket fixes that for more Redis work.
- **Cache hits don't consume token quota.** They cost nothing upstream.
- **Tests hit real Postgres and Redis; only the LLM provider is faked.** The SQL and the Redis pipeline are where bugs
  live. A fake embedder with hand-chosen vectors lets tests set exact similarities around the threshold. Injecting
  bugs on purpose (always-hit, never-store, charge-quota-on-hit) turns the suite red, so the tests are known to catch
  them.

## Known limitations

- **False hits are the real problem.** Topic similarity isn't answer equivalence: "How do I become an introvert?"
  matched "How do I become *less of* an introvert?" at 0.95. Next steps would be a stronger embedding model or a
  verification step (a cross-encoder re-scoring the top match) before serving a hit.
- **Cache stampede.** Two identical prompts arriving at the same moment both miss and both call the provider.
  The fix is request coalescing ("single-flight").
- **`max_tokens` is not part of the cache key**, so a request can receive a cached answer longer than it asked for.
- **The cache is shared across API keys.** That maximizes hit rate, but a multi-tenant deployment would partition it.
- **No TTL or eviction.** Answers never expire and the table grows without bound.
- **Single-turn prompts only.** No streaming, multi-provider failover, or user management (all out of scope).

## Running it

```bash
cp .env.example .env               # then set ANTHROPIC_API_KEY
docker compose up -d --wait        # gateway :8000, Postgres :5432, Redis :6379
```

```bash
curl -s localhost:8000/v1/complete -H "x-api-key: dev-key-123" -H "content-type: application/json" \
  -d '{"prompt": "What is the best way to learn Python?", "max_tokens": 100}'
```

**Dashboard: http://localhost:8000** is a plain-English page with live hit rate, request volume, money saved,
latency, and a box to try questions yourself. Raw interactive API docs are at http://localhost:8000/docs. All settings (threshold, rate limits, model, API keys) are env
vars; see [`app/config.py`](app/config.py).

**Tests** (need the compose Postgres and Redis running):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

**Benchmark:**

```bash
pip install -r requirements-bench.txt
mkdir -p bench/data && curl -L -o bench/data/qqp_validation.parquet \
  https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet
python -m bench.sweep --pairs 2000   # free, ~10 s
python -m bench.live --pairs 300     # real API calls: prints a cost estimate and asks first
python -m bench.report               # the tables above, from results/*.json
```
