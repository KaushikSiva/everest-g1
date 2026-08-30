"""Optional Bright Data research adapter, kept outside every control loop."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlencode


class BrightDataConfigurationError(RuntimeError):
    """Raised when the optional research connector is not configured."""


@dataclass(frozen=True, repr=False)
class BrightDataSettings:
    api_token: str
    endpoint: str = "https://mcp.brightdata.com/mcp"

    @classmethod
    def from_env(cls) -> BrightDataSettings:
        token = os.getenv("BRIGHT_DATA_API_TOKEN", "")
        if not token:
            raise BrightDataConfigurationError("BRIGHT_DATA_API_TOKEN is required")
        return cls(api_token=token)

    def __repr__(self) -> str:
        return f"BrightDataSettings(api_token='<redacted>', endpoint={self.endpoint!r})"

    def remote_url(self) -> str:
        query = urlencode(
            {
                "token": self.api_token,
                "tools": "search_engine,scrape_as_markdown",
            }
        )
        return f"{self.endpoint}?{query}"


async def search_public_conditions(query: str, settings: BrightDataSettings) -> list[str]:
    """Return text blocks from a Bright Data search without interpreting control values."""

    if not query.strip() or len(query) > 300:
        raise ValueError("query must contain 1-300 non-whitespace characters")

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        async with (
            streamablehttp_client(settings.remote_url()) as (reader, writer, _),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "search_engine",
                {"query": query.strip(), "engine": "google"},
            )
    except Exception as exc:
        # The remote URL contains a credential. Never include the underlying message.
        raise RuntimeError(f"Bright Data request failed ({type(exc).__name__})") from exc

    return [block.text for block in result.content if getattr(block, "type", None) == "text"]
