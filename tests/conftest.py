"""Test fixtures.

Postgres and Redis are REAL (from docker-compose locally, service containers in
CI) — the pgvector query and the Redis pipeline are exactly the code most likely
to be wrong, so mocking them would test nothing. Only the LLM provider is faked,
because it costs money and needs the network.
"""
import dataclasses
import os

import asyncpg
import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.fake import FakeProvider

API_KEY = "test-key"
HEADERS = {"x-api-key": API_KEY}


class FakeEmbedder:
    """Maps text to hand-chosen vectors so a test can dictate exact similarities.

    `vectors` maps a prompt to a vector; any other prompt gets a vector orthogonal
    to everything registered (similarity 0). All vectors are normalized, matching
    the real embedder's contract.
    """

    dim = 384

    def __init__(self, vectors: dict[str, list[float]] | None = None, too_long: set[str] | None = None):
        self.vectors = vectors or {}
        self.too_long = too_long or set()
        self._fallback = {}

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if text in self.vectors:
            raw = self.vectors[text]
            v[: len(raw)] = raw
        else:
            # Unique one-hot in the upper dimensions -> orthogonal to all registered vectors.
            idx = self._fallback.setdefault(text, 200 + len(self._fallback))
            v[idx] = 1.0
        return v / np.linalg.norm(v)

    def fits(self, text: str) -> bool:
        return text not in self.too_long


def vec_with_similarity(sim: float) -> list[float]:
    """A 2-d vector whose cosine similarity with [1, 0] is exactly `sim`."""
    return [sim, float(np.sqrt(1 - sim**2))]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://gateway:gateway@localhost:5432/gateway"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        api_keys={API_KEY: "tester"},
        similarity_threshold=0.90,
        provider="fake",
        model="claude-haiku-4-5",
        rate_limit_per_minute=1000,
        daily_token_quota=1_000_000,
    )


@pytest.fixture(autouse=True)
def clean_state(settings):
    """Every test starts with an empty cache, empty ledger and empty Redis."""

    async def truncate():
        conn = await asyncpg.connect(settings.database_url)
        await conn.execute("TRUNCATE semantic_cache, usage_log")
        await conn.close()

    import asyncio

    asyncio.run(truncate())
    redis.Redis.from_url(settings.redis_url).flushdb()
    yield


@pytest.fixture
def make_client(settings):
    """Build a TestClient with a given embedder/provider/settings overrides.

    Using the client as a context manager runs the app's lifespan (DB pool,
    Redis connection) exactly like uvicorn would.
    """
    clients = []

    def _make(embedder=None, provider=None, **overrides):
        s = dataclasses.replace(settings, **overrides)
        provider = provider or FakeProvider()
        app = create_app(s, provider=provider, embedder=embedder or FakeEmbedder())
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, provider

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def ledger(settings) -> list[dict]:
    import asyncio

    async def fetch():
        conn = await asyncpg.connect(settings.database_url)
        rows = await conn.fetch("SELECT * FROM usage_log ORDER BY id")
        await conn.close()
        return [dict(r) for r in rows]

    return asyncio.run(fetch())
