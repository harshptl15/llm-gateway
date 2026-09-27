import hmac

from fastapi import Header, HTTPException, Request


async def authenticate(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """Resolve the x-api-key header to a key_id, or 401.

    compare_digest over every key (instead of `dict.get`) keeps the comparison
    constant-time, so response timing doesn't leak how much of a guessed key matched.
    """
    if x_api_key:
        for key, key_id in request.app.state.settings.api_keys.items():
            if hmac.compare_digest(key.encode(), x_api_key.encode()):
                return key_id
    raise HTTPException(status_code=401, detail="invalid or missing x-api-key")
