from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """Upstream call failed. `status` is what the gateway returns to its client."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class Provider(Protocol):
    """The only thing the gateway knows about an LLM backend.

    Keeping it this narrow means tests (and the benchmark's dry runs) swap in
    FakeProvider without touching request-handling code.
    """

    async def complete(self, prompt: str, *, model: str, max_tokens: int) -> Completion: ...
