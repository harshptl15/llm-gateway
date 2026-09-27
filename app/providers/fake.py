import asyncio

from .base import Completion


class FakeProvider:
    """Deterministic stand-in for tests: no network, no cost.

    `calls` lets a test assert "the provider was NOT called" on a cache hit,
    which is the whole point of the cache.
    """

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = latency_s
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, model: str, max_tokens: int) -> Completion:
        self.calls.append(prompt)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return Completion(text=f"answer to: {prompt}", input_tokens=len(prompt.split()), output_tokens=20)
