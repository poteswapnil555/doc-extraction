"""LLM client abstraction. Real client is built from env variables only."""
from __future__ import annotations

import re
import sys
import time
from typing import Protocol

from .config import azure_auth_mode, env, llm_provider

TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
MAX_RATE_LIMIT_RETRIES = 6
DEFAULT_RETRY_SECONDS = 15.0
MAX_WAIT_SECONDS = 120.0
_sleep = time.sleep  # replaced in tests


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429


def retry_delay_seconds(exc: Exception) -> float:
    """Wait time from the Retry-After header or the error text ("retry in 12.5s",
    "retryDelay: 12s"); 15 seconds if none is shown. Capped at 120 seconds."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    candidates = [headers.get("retry-after")] if hasattr(headers, "get") else []
    text = str(exc)
    for pattern in (r"retry in ([\d.]+)\s*s", r"retryDelay\W+([\d.]+)\s*s", r"retry after ([\d.]+)\s*s"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidates.append(m.group(1))
    for c in candidates:
        try:
            return min(max(float(c), 0.0), MAX_WAIT_SECONDS)
        except (TypeError, ValueError):
            continue
    return DEFAULT_RETRY_SECONDS


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def build_token_provider():
    """Sign-in mode: service principal from AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET."""
    from azure.identity import ClientSecretCredential, get_bearer_token_provider

    credential = ClientSecretCredential(
        tenant_id=env("AZURE_TENANT_ID"),
        client_id=env("AZURE_CLIENT_ID"),
        client_secret=env("AZURE_CLIENT_SECRET"),
    )
    return get_bearer_token_provider(credential, TOKEN_SCOPE)


class OpenAICompatibleClient:
    """Works with Azure OpenAI or any OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        from openai import AzureOpenAI, OpenAI

        if llm_provider() == "azure":
            auth = (
                {"api_key": env("AZURE_OPENAI_API_KEY")}
                if azure_auth_mode() == "key"
                else {"azure_ad_token_provider": build_token_provider()}
            )
            self._client = AzureOpenAI(
                azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
                api_version=env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                **auth,
            )
            self._model = env("AZURE_OPENAI_DEPLOYMENT", "")
        else:
            self._client = OpenAI(api_key=env("OPENAI_API_KEY"), base_url=env("OPENAI_BASE_URL"))
            self._model = env("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if not is_rate_limit(e) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = retry_delay_seconds(e)
                print(f"Rate limited (429). Waiting {wait:.0f}s, retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}...", file=sys.stderr)
                _sleep(wait)
