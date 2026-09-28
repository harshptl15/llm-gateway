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
