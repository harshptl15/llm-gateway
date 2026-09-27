from .base import Completion, Provider, ProviderError


def make_provider(name: str) -> Provider:
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider  # lazy: tests never import the SDK path

        return AnthropicProvider()
    if name == "fake":
        from .fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"unknown provider {name!r}")


__all__ = ["Completion", "Provider", "ProviderError", "make_provider"]
