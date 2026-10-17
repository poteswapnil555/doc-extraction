"""LLM client abstraction. Real client is built from env variables only."""
from __future__ import annotations

from typing import Protocol

from .config import env, llm_provider


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    """Works with Azure OpenAI or any OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        from openai import AzureOpenAI, OpenAI

        if llm_provider() == "azure":
            self._client = AzureOpenAI(
                azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
                api_key=env("AZURE_OPENAI_API_KEY"),
                api_version=env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
            self._model = env("AZURE_OPENAI_DEPLOYMENT", "")
        else:
            self._client = OpenAI(api_key=env("OPENAI_API_KEY"), base_url=env("OPENAI_BASE_URL"))
            self._model = env("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
