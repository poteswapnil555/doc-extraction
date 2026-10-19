"""All configuration comes from environment variables. Nothing is read from files here."""
from __future__ import annotations

import os


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value else default


def azure_auth_mode() -> str | None:
    """'key' if AZURE_OPENAI_API_KEY is set, else 'token' if the service-principal
    variables (AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET) are all set, else None."""
    if not env("AZURE_OPENAI_ENDPOINT"):
        return None
    if env("AZURE_OPENAI_API_KEY"):
        return "key"
    if env("AZURE_TENANT_ID") and env("AZURE_CLIENT_ID") and env("AZURE_CLIENT_SECRET"):
        return "token"
    return None


def llm_provider() -> str:
    """'azure' if Azure endpoint + credentials are set, 'openai' if OPENAI_API_KEY is set."""
    if azure_auth_mode():
        return "azure"
    if env("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY/"
        "AZURE_OPENAI_DEPLOYMENT/AZURE_OPENAI_API_VERSION plus AZURE_OPENAI_API_KEY or "
        "AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET, or OPENAI_API_KEY (+ optional OPENAI_BASE_URL)."
    )


def llm_configured() -> bool:
    try:
        llm_provider()
        return True
    except RuntimeError:
        return False
