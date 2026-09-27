import time
from datetime import datetime, timezone

import redis.asyncio as redis


class RateLimited(Exception):
    def __init__(self, reason: str, retry_after: int):
        super().__init__(reason)
        self.retry_after = retry_after


class RateLimiter:
    """Two checks per request, both in Redis so every gateway replica shares them:

    1. Requests per minute — fixed window. Key = rl:{key_id}:{minute number}.
       INCR is atomic, so concurrent requests can't both read "99" and both pass.
       Known weakness: a burst straddling a minute boundary can get up to 2x the
       limit. Sliding-window or token-bucket fixes that at the cost of more Redis work.
    2. Provider tokens per UTC day — a quota. Cache hits don't count against it,
       because they didn't cost anything.
    """

    def __init__(self, client: redis.Redis, per_minute: int, daily_tokens: int):
        self.r = client
        self.per_minute = per_minute
        self.daily_tokens = daily_tokens

    @staticmethod
    def _quota_key(key_id: str) -> str:
        return f"quota:{key_id}:{datetime.now(timezone.utc):%Y-%m-%d}"

    async def check(self, key_id: str) -> None:
        now = time.time()
        window_key = f"rl:{key_id}:{int(now // 60)}"
        # Pipeline = one round trip for all three commands.
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.incr(window_key)
            pipe.expire(window_key, 60)  # old windows delete themselves
            pipe.get(self._quota_key(key_id))
            count, _, used = await pipe.execute()

        if count > self.per_minute:
            raise RateLimited("rate limit exceeded", retry_after=60 - int(now % 60))
        if used is not None and int(used) >= self.daily_tokens:
            raise RateLimited("daily token quota exhausted", retry_after=3600)

    async def record_tokens(self, key_id: str, tokens: int) -> None:
        key = self._quota_key(key_id)
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.incrby(key, tokens)
            pipe.expire(key, 2 * 86400)
            await pipe.execute()
