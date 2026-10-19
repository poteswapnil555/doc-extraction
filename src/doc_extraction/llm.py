"""LLM client abstraction. Real client is built from env variables only."""
from __future__ import annotations

from typing import Protocol

from .config import azure_auth_mode, env, llm_provider

TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


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
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
