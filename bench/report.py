"""Render results/*.json as the markdown tables used in README.md.

    python -m bench.report > results/tables.md
"""
import json

from .sweep import RESULTS


def main():
    live = json.loads((RESULTS / "live.json").read_text())
    sweep = json.loads((RESULTS / "sweep.json").read_text())
    srv, cli, cost = live["latency_server_ms"], live["latency_client_ms"], live["cost_usd"]
    miss, hit = srv["uncached"], srv["cached_replay"]
    v = live["hit_verdicts"]

    print(f"### Live run: {live['requests']} real requests, {live['model']}, threshold {live['threshold']}\n")
    print("| Latency (gateway-side) | p50 | p95 | p99 | n |")
    print("|---|---:|---:|---:|---:|")
    print(f"| Cache miss (calls the AI) | {miss['p50']:,.0f} ms | {miss['p95']:,.0f} ms | {miss['p99']:,.0f} ms | {miss['n']} |")
    print(f"| Cache hit | {hit['p50']:.1f} ms | {hit['p95']:.1f} ms | {hit['p99']:.1f} ms | {hit['n']} |")
    print(f"| **Speedup** | **{miss['p50'] / hit['p50']:.0f}×** | {miss['p95'] / hit['p95']:.0f}× | {miss['p99'] / hit['p99']:.0f}× | |\n")
    c = cli["cached_replay"]
    print(f"Measured inside the gateway, from request received to response ready (`usage_log.latency_ms`), so both rows "
          f"are timed the same way. Seen from the client over localhost, a cache hit is p50 {c['p50']:.1f} ms / "
          f"p99 {c['p99']:.1f} ms (HTTP overhead: median {live['client_minus_server_ms_p50']:.1f} ms).\n")

    print("| Result | Value |")
    print("|---|---:|")
    stop = (f" (run stopped at request {live['stopped_at']} of {live['stream_requests']}: API credit ran out)"
            if live.get("stopped_at") else "")
    print(f"| Requests measured | {live['requests']}{stop} |")
    print(f"| Hit rate | {live['hit_rate']:.1%} ({live['hits']} / {live['requests']}); "
          f"simulation predicts {live['predicted_hit_rate_full_stream']:.1%} over all {live['stream_requests']} |")
    print(f"| Wrong-question hits | {v['wrong'] + v['unlabeled']} of {live['hits']} (small sample; see the sweep's "
          f"{next(r for r in sweep['rows'] if r['threshold'] == live['threshold'])['hits']} hits at this threshold) |")
    print(f"| Spent on the AI | ${cost['spent']:.4f} |")
    print(f"| Saved by the cache | ${cost['saved']:.4f} ({cost['saved_pct_of_uncached_spend']:.1%} of what it would have cost) |")
    print(f"| Avg cost per cache miss | ${cost['avg_cost_per_miss']:.5f} |")
    print(f"| Live vs. offline-simulation agreement | {live['sim_live_agreement']:.0%} of hit/miss decisions |\n")

    print(f"### Threshold sweep: {sweep['requests']:,} requests ({sweep['pairs']:,} QQP pairs), {sweep['embedding_model']}\n")
    print("| Threshold | Hit rate | False-hit rate (labeled – conservative) | Paraphrase recall |")
    print("|---:|---:|---:|---:|")
    for r in sweep["rows"]:
        b = "**" if r["threshold"] == live["threshold"] else ""
        print(f"| {b}{r['threshold']:.2f}{b} | {r['hit_rate']:.1%} | "
              f"{r['false_hit_rate_low']:.1%} – {r['false_hit_rate_high']:.1%} | {r['dup_recall']:.1%} |")


if __name__ == "__main__":
    main()
