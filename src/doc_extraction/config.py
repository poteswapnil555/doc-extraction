"""All configuration comes from environment variables. Nothing is read from files here."""
from __future__ import annotations

import os


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value else default


def llm_provider() -> str:
    """'azure' if Azure variables are set, 'openai' if OPENAI_API_KEY is set."""
    if env("AZURE_OPENAI_ENDPOINT") and env("AZURE_OPENAI_API_KEY"):
        return "azure"
    if env("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY/"
        "AZURE_OPENAI_DEPLOYMENT/AZURE_OPENAI_API_VERSION, or OPENAI_API_KEY (+ optional OPENAI_BASE_URL)."
    )


def llm_configured() -> bool:
    try:
        llm_provider()
        return True
    except RuntimeError:
        return False
