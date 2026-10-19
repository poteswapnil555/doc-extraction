import pytest

from conftest import SAMPLE
from doc_extraction import llm as llm_mod
from doc_extraction.evaluate import MockLLM, evaluate, main
from doc_extraction.ocr import TesseractOcr


class Boom(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


class _Message:
    content = '{"ok": true}'


class _Choice:
    message = _Message()


class _Resp:
    choices = [_Choice()]


class FlakyClient:
    """Fake OpenAI client: raises the given errors in order, then succeeds."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _Resp()


@pytest.fixture
def waits(monkeypatch):
    w = []
    monkeypatch.setattr(llm_mod, "_sleep", w.append)
    return w


def _client(fake):
    c = object.__new__(llm_mod.OpenAICompatibleClient)
    c._client, c._model = fake, "m"
    return c


def test_retries_after_429_using_time_in_error(waits):
    fake = FlakyClient(Boom("quota exceeded. Please retry in 12.5s.", 429))
    assert _client(fake).complete("s", "u") == '{"ok": true}'
    assert fake.calls == 2 and waits == [12.5]


def test_reads_retry_delay_field(waits):
    fake = FlakyClient(Boom('{"retryDelay": "21s"}', 429))
    _client(fake).complete("s", "u")
    assert waits == [21.0]


def test_default_wait_is_15_seconds(waits):
    fake = FlakyClient(Boom("too many requests", 429))
    _client(fake).complete("s", "u")
    assert waits == [15.0]


def test_gives_up_after_6_retries(waits):
    fake = FlakyClient(*[Boom("slow down", 429) for _ in range(10)])
    with pytest.raises(Boom):
        _client(fake).complete("s", "u")
    assert fake.calls == 7 and len(waits) == 6


def test_other_errors_are_not_retried(waits):
    fake = FlakyClient(Boom("bad request", 400))
    with pytest.raises(Boom):
        _client(fake).complete("s", "u")
    assert fake.calls == 1 and waits == []


def test_delay_and_progress_in_evaluate():
    slept, seen = [], []
    evaluate(SAMPLE, TesseractOcr(), MockLLM(), "mock", delay=2.5, sleep=slept.append,
             progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (1, 20) and seen[-1] == (20, 20)
    assert slept == [2.5] * 19


def test_cli_prints_progress_to_stderr(capsys):
    assert main(["--mock", "--dir", str(SAMPLE), "--delay", "0"]) == 0
    err = capsys.readouterr().err
    assert "3/20 done" in err and "20/20 done" in err
