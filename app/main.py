import time
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field

from .auth import authenticate
from .cache import SemanticCache
from .config import Settings, settings as default_settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .pricing import cost_usd
from .providers import Provider, ProviderError, make_provider
from .ratelimit import RateLimited, RateLimiter


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_tokens: int = Field(default=256, ge=1, le=4096)


class CompleteResponse(BaseModel):
    text: str
    cached: bool
    similarity: float | None  # best match found, hit or not
    model: str
    input_tokens: int         # billed by the provider for THIS request (0 on a hit)
    output_tokens: int
    cost_usd: float
    saved_usd: float


def create_app(settings: Settings = default_settings, *, provider: Provider | None = None,
               embedder: Embedder | None = None) -> FastAPI:
    """Factory so tests can inject a FakeProvider and a deterministic embedder."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # register_vector teaches asyncpg to send/receive numpy arrays as pgvector values.
        pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10, init=register_vector)
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        app.state.settings = settings
        app.state.pool = pool
        app.state.cache = SemanticCache(pool, settings.similarity_threshold)
        app.state.limiter = RateLimiter(redis_client, settings.rate_limit_per_minute, settings.daily_token_quota)
        # Loading the embedding model takes ~1-2s; do it once at startup, not per request.
        app.state.embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
        # torch initializes lazily; without this the first real request pays ~100ms extra.
        app.state.embedder.embed("warmup")
        app.state.provider = provider or make_provider(settings.provider)
        yield
        await pool.close()
        await redis_client.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        s = request.app.state.settings
        return {"ok": True, "model": s.model, "similarity_threshold": s.similarity_threshold}

    @app.post("/v1/complete", response_model=CompleteResponse)
    async def complete(body: CompleteRequest, request: Request, response: Response,
                       key_id: str = Depends(authenticate)):  # step 1: auth runs before the handler body
        state = request.app.state
        model = state.settings.model
        start = time.perf_counter()

        # Step 2: rate limit / quota.
        try:
            await state.limiter.check(key_id)
        except RateLimited as e:
            raise HTTPException(429, detail=str(e), headers={"Retry-After": str(e.retry_after)})

        # Step 3: embed + nearest-neighbour search. encode() is CPU-bound and
        # synchronous; calling it directly would block the event loop and stall
        # every other in-flight request, so it goes to a worker thread.
        cacheable = state.embedder.fits(body.prompt)
        hit, similarity, embedding = None, None, None
        if cacheable:
            embedding = await run_in_threadpool(state.embedder.embed, body.prompt)
            hit, similarity = await state.cache.lookup(embedding, model)

        # Step 4: HIT -> serve stored answer. MISS -> provider, then store.
        if hit:
            text, in_tok, out_tok, cost = hit.response, 0, 0, 0.0
            saved = cost_usd(model, hit.input_tokens, hit.output_tokens)
        else:
            try:
                completion = await state.provider.complete(body.prompt, model=model, max_tokens=body.max_tokens)
            except ProviderError as e:
                raise HTTPException(e.status, detail=str(e))
            text, in_tok, out_tok = completion.text, completion.input_tokens, completion.output_tokens
            cost, saved = cost_usd(model, in_tok, out_tok), 0.0
            if cacheable:
                await state.cache.store(model=model, prompt=body.prompt, embedding=embedding, response=text,
                                        input_tokens=in_tok, output_tokens=out_tok)
            await state.limiter.record_tokens(key_id, in_tok + out_tok)

        latency_ms = (time.perf_counter() - start) * 1000

        # Step 5: per-key usage ledger.
        await state.pool.execute(
            """INSERT INTO usage_log (key_id, model, cache_hit, cache_entry_id, similarity, input_tokens,
                                      output_tokens, cost_usd, saved_usd, latency_ms)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            key_id, model, hit is not None, hit.id if hit else None, similarity, in_tok, out_tok,
            cost, saved, latency_ms,
        )

        response.headers["X-Cache"] = "HIT" if hit else "MISS"
        return CompleteResponse(text=text, cached=hit is not None, similarity=similarity, model=model,
                                input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost, saved_usd=saved)

    return app


app = create_app()
