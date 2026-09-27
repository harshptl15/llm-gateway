import pytest

from app.pricing import cost_usd
from app.providers.base import Completion, ProviderError

from .conftest import HEADERS, FakeEmbedder, ledger, vec_with_similarity


def ask(client, prompt, **kw):
    return client.post("/v1/complete", json={"prompt": prompt, **kw}, headers=HEADERS)


# ---------------------------------------------------------------- cache behaviour

def test_first_request_is_a_miss_and_calls_provider(make_client):
    client, provider = make_client()
    r = ask(client, "hello")
    assert r.status_code == 200
    assert r.json()["cached"] is False
    assert r.headers["X-Cache"] == "MISS"
    assert provider.calls == ["hello"]


def test_identical_prompt_is_a_hit_and_skips_provider(make_client):
    client, provider = make_client()
    first = ask(client, "hello").json()
    second = ask(client, "hello")
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["text"] == first["text"]
    assert second.json()["similarity"] == pytest.approx(1.0)
    assert provider.calls == ["hello"]  # the provider was called exactly once


def test_similar_prompt_above_threshold_is_a_hit(make_client):
    emb = FakeEmbedder({"original": [1, 0], "paraphrase": vec_with_similarity(0.95)})
    client, provider = make_client(embedder=emb, similarity_threshold=0.90)
    ask(client, "original")
    r = ask(client, "paraphrase").json()
    assert r["cached"] is True
    assert r["similarity"] == pytest.approx(0.95, abs=1e-4)
    assert r["text"] == "answer to: original"  # served the ORIGINAL prompt's answer
    assert provider.calls == ["original"]


def test_similar_prompt_below_threshold_is_a_miss(make_client):
    emb = FakeEmbedder({"original": [1, 0], "different": vec_with_similarity(0.85)})
    client, provider = make_client(embedder=emb, similarity_threshold=0.90)
    ask(client, "original")
    r = ask(client, "different").json()
    assert r["cached"] is False
    assert r["similarity"] == pytest.approx(0.85, abs=1e-4)  # best match is still reported
    assert provider.calls == ["original", "different"]


@pytest.mark.parametrize("threshold, expect_hit", [(0.80, True), (0.90, True), (0.95, False)])
def test_threshold_is_configurable(make_client, threshold, expect_hit):
    emb = FakeEmbedder({"a": [1, 0], "b": vec_with_similarity(0.92)})
    client, _ = make_client(embedder=emb, similarity_threshold=threshold)
    ask(client, "a")
    assert ask(client, "b").json()["cached"] is expect_hit


def test_nearest_neighbour_wins_among_multiple_entries(make_client):
    emb = FakeEmbedder({"far": [1, 0], "near": [0, 1], "query": [0.3, 0.954]})
    client, _ = make_client(embedder=emb, similarity_threshold=0.90)
    ask(client, "far")
    ask(client, "near")
    r = ask(client, "query").json()
    assert r["cached"] is True
    assert r["text"] == "answer to: near"


def test_cache_is_scoped_to_model(make_client):
    client_a, _ = make_client(model="claude-haiku-4-5")
    ask(client_a, "hello")
    client_b, provider_b = make_client(model="claude-sonnet-5")
    assert ask(client_b, "hello").json()["cached"] is False
    assert provider_b.calls == ["hello"]


def test_prompt_too_long_to_embed_bypasses_cache(make_client):
    client, provider = make_client(embedder=FakeEmbedder(too_long={"huge"}))
    ask(client, "huge")
    r = ask(client, "huge").json()
    assert r["cached"] is False and r["similarity"] is None
    assert provider.calls == ["huge", "huge"]


def test_real_embedder_matches_paraphrase_not_unrelated(make_client):
    """End-to-end with the actual MiniLM model: semantic, not lexical, matching."""
    from app.embeddings import SentenceTransformerEmbedder

    client, _ = make_client(embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
                            similarity_threshold=0.85)
    ask(client, "How do I learn Python quickly?")
    assert ask(client, "What is the fastest way to learn Python?").json()["cached"] is True
    assert ask(client, "What is the capital of France?").json()["cached"] is False


# ---------------------------------------------------------------- usage ledger / cost

def test_usage_logged_with_cost_on_miss_and_savings_on_hit(make_client, settings):
    client, _ = make_client()
    miss = ask(client, "one two three").json()
    hit = ask(client, "one two three").json()

    expected_cost = cost_usd("claude-haiku-4-5", 3, 20)  # FakeProvider: words in, 20 out
    assert miss["cost_usd"] == pytest.approx(expected_cost)
    assert hit["cost_usd"] == 0 and hit["saved_usd"] == pytest.approx(expected_cost)

    rows = ledger(settings)
    assert [r["cache_hit"] for r in rows] == [False, True]
    assert all(r["key_id"] == "tester" for r in rows)
    assert rows[0]["input_tokens"] == 3 and rows[1]["input_tokens"] == 0
    assert rows[0]["cache_entry_id"] is None and rows[1]["cache_entry_id"] is not None
    assert float(rows[1]["saved_usd"]) == pytest.approx(expected_cost)


# ---------------------------------------------------------------- auth / limits / errors

@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}])
def test_bad_or_missing_key_is_401(make_client, headers):
    client, provider = make_client()
    r = client.post("/v1/complete", json={"prompt": "hi"}, headers=headers)
    assert r.status_code == 401
    assert provider.calls == []


def test_rate_limit_returns_429_with_retry_after(make_client):
    client, _ = make_client(rate_limit_per_minute=3)
    codes = [ask(client, f"q{i}").status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
    assert "Retry-After" in ask(client, "q").headers


def test_daily_token_quota_counts_provider_tokens_only(make_client):
    # FakeProvider bills 1 input + 20 output = 21 tokens for a one-word prompt.
    client, _ = make_client(daily_token_quota=42)  # room for exactly two misses
    assert ask(client, "first").status_code == 200                 # miss: 21 used
    for _ in range(3):
        assert ask(client, "first").json()["cached"] is True      # hits: still 21 used
    assert ask(client, "second").status_code == 200                # miss: 42 used -> full
    assert ask(client, "third").status_code == 429


def test_provider_error_maps_to_gateway_status(make_client):
    class Failing:
        async def complete(self, prompt, *, model, max_tokens) -> Completion:
            raise ProviderError("upstream down", status=502)

    client, _ = make_client(provider=Failing())
    r = ask(client, "hi")
    assert r.status_code == 502
    # A failed call must not poison the cache.
    client2, provider = make_client()
    assert ask(client2, "hi").json()["cached"] is False


def test_request_validation(make_client):
    client, _ = make_client()
    assert ask(client, "").status_code == 422
    assert ask(client, "hi", max_tokens=0).status_code == 422
