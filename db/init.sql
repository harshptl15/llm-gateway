-- Runs once, the first time the Postgres container starts with an empty volume.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per provider response we are willing to serve again.
CREATE TABLE IF NOT EXISTS semantic_cache (
    id            BIGSERIAL PRIMARY KEY,
    model         TEXT        NOT NULL,   -- a cached answer is only valid for the model that wrote it
    prompt        TEXT        NOT NULL,
    embedding     vector(384) NOT NULL,   -- all-MiniLM-L6-v2 output size
    response      TEXT        NOT NULL,
    input_tokens  INT         NOT NULL,   -- what the original call cost; a hit "saves" this much
    output_tokens INT         NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW = approximate nearest-neighbour graph index. Without it every lookup is a
-- full scan computing cosine distance against every row (fine at 1k rows, not at 1M).
-- vector_cosine_ops makes the index serve the `<=>` (cosine distance) operator.
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_hnsw
    ON semantic_cache USING hnsw (embedding vector_cosine_ops);

-- One row per request through the gateway. Cost numbers are computed at write
-- time from the pricing table in app/pricing.py.
CREATE TABLE IF NOT EXISTS usage_log (
    id            BIGSERIAL PRIMARY KEY,
    key_id        TEXT        NOT NULL,
    model         TEXT        NOT NULL,
    cache_hit     BOOLEAN     NOT NULL,
    similarity    REAL,                   -- best match similarity (NULL if cache was empty)
    input_tokens  INT         NOT NULL,   -- tokens actually billed by the provider (0 on a hit)
    output_tokens INT         NOT NULL,
    cost_usd      NUMERIC(12, 8) NOT NULL,
    saved_usd     NUMERIC(12, 8) NOT NULL, -- on a hit: what the original call cost
    latency_ms    REAL        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_log_key_time ON usage_log (key_id, created_at);
