"""Auth-mode selection, with the Azure/OpenAI SDKs replaced by fakes. No key or network needed."""
import openai
import pytest

from doc_extraction import llm
from doc_extraction.config import azure_auth_mode, llm_provider

ALL = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
       "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL"]


@pytest.fixture
def clean_env(monkeypatch):
    for v in ALL:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    return monkeypatch


@pytest.fixture
def fake_azure(monkeypatch):
    captured = {}

    class FakeAzure:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openai, "AzureOpenAI", FakeAzure)
    monkeypatch.setattr(llm, "build_token_provider", lambda: "TOKEN_PROVIDER")
    return captured


def _sp(mp):
    mp.setenv("AZURE_TENANT_ID", "t")
    mp.setenv("AZURE_CLIENT_ID", "c")
    mp.setenv("AZURE_CLIENT_SECRET", "s")


def test_key_mode_when_key_set(clean_env, fake_azure):
    clean_env.setenv("AZURE_OPENAI_API_KEY", "fake")
    _sp(clean_env)  # key wins even if sign-in variables exist
    assert azure_auth_mode() == "key"
    llm.OpenAICompatibleClient()
    assert fake_azure["api_key"] == "fake" and "azure_ad_token_provider" not in fake_azure


def test_token_mode_when_no_key(clean_env, fake_azure):
    _sp(clean_env)
    assert azure_auth_mode() == "token" and llm_provider() == "azure"
    llm.OpenAICompatibleClient()
    assert fake_azure["azure_ad_token_provider"] == "TOKEN_PROVIDER" and "api_key" not in fake_azure


def test_incomplete_sign_in_variables_not_configured(clean_env):
    clean_env.setenv("AZURE_TENANT_ID", "t")
    assert azure_auth_mode() is None
    with pytest.raises(RuntimeError):
        llm_provider()


def test_token_provider_uses_cognitive_services_scope(clean_env, monkeypatch):
    import azure.identity as ai

    seen = {}
    _sp(clean_env)
    monkeypatch.setattr(ai, "ClientSecretCredential", lambda **kw: ("cred", kw))
    monkeypatch.setattr(ai, "get_bearer_token_provider", lambda cred, scope: seen.update(cred=cred, scope=scope) or "p")
    assert llm.build_token_provider() == "p"
    assert seen["scope"] == "https://cognitiveservices.azure.com/.default"
    assert seen["cred"][1] == {"tenant_id": "t", "client_id": "c", "client_secret": "s"}
