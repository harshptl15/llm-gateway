import anthropic

from .base import Completion, ProviderError


class AnthropicProvider:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        # Zero-arg client reads ANTHROPIC_API_KEY from the environment.
        self.client = client or anthropic.AsyncAnthropic()

    async def complete(self, prompt: str, *, model: str, max_tokens: int) -> Completion:
        try:
            msg = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        # Most specific first: a 429 upstream is "try again later" for our client
        # too; any other non-2xx or network failure is a bad gateway.
        except anthropic.RateLimitError as e:
            raise ProviderError(f"provider rate limited: {e}", status=503) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"provider error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"provider unreachable: {e}") from e

        text = "".join(block.text for block in msg.content if block.type == "text")
        return Completion(text, msg.usage.input_tokens, msg.usage.output_tokens)
