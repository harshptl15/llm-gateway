from dataclasses import dataclass

import asyncpg
import numpy as np


@dataclass
class CacheEntry:
    id: int
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    similarity: float


class SemanticCache:
    def __init__(self, pool: asyncpg.Pool, threshold: float):
        self.pool = pool
        self.threshold = threshold

    async def nearest(self, embedding: np.ndarray, model: str) -> CacheEntry | None:
        """Closest cached prompt for this model, regardless of threshold.

        `<=>` is pgvector's cosine *distance* (1 - cosine similarity). Ordering by
        it ascending with LIMIT 1 is what lets Postgres use the HNSW index; the
        index returns an approximate nearest neighbour, not a guaranteed exact one.
        """
        row = await self.pool.fetchrow(
            """
            SELECT id, prompt, response, input_tokens, output_tokens,
                   1 - (embedding <=> $1) AS similarity
            FROM semantic_cache
            WHERE model = $2
            ORDER BY embedding <=> $1
            LIMIT 1
            """,
            embedding,
            model,
        )
        return CacheEntry(**dict(row)) if row else None

    async def lookup(self, embedding: np.ndarray, model: str) -> tuple[CacheEntry | None, float | None]:
        """Returns (hit_or_None, best_similarity). The similarity is returned even on
        a miss so it can be logged — that's what makes threshold tuning possible."""
        best = await self.nearest(embedding, model)
        if best is None:
            return None, None
        return (best if best.similarity >= self.threshold else None), best.similarity

    async def store(self, *, model: str, prompt: str, embedding: np.ndarray, response: str,
                    input_tokens: int, output_tokens: int) -> None:
        await self.pool.execute(
            """
            INSERT INTO semantic_cache (model, prompt, embedding, response, input_tokens, output_tokens)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            model, prompt, embedding, response, input_tokens, output_tokens,
        )
