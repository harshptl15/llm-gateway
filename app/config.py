"""All runtime knobs, read once from the environment.

Plain os.environ instead of pydantic-settings: fewer moving parts, and every
setting is visible in one screen.
"""
import os
from dataclasses import dataclass, field


def _parse_keys(raw: str) -> dict[str, str]:
    """"key1:alice,key2:bob" -> {"key1": "alice", "key2": "bob"}."""
    out = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, name = pair.partition(":")
        out[key] = name or key
    return out


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://gateway:gateway@localhost:5432/gateway")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # api key -> key_id (a human-readable name used in logs/usage).
    api_keys: dict[str, str] = field(
        default_factory=lambda: _parse_keys(os.getenv("GATEWAY_API_KEYS", "dev-key-123:dev"))
    )

    # Cosine similarity at or above which a cached answer is served.
    similarity_threshold: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    provider: str = os.getenv("PROVIDER", "anthropic")  # "anthropic" | "fake"
    model: str = os.getenv("MODEL", "claude-haiku-4-5")

    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "600"))
    daily_token_quota: int = int(os.getenv("DAILY_TOKEN_QUOTA", "2000000"))


settings = Settings()
