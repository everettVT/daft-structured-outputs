import os
import re

import pytest

# Mark as integration; this test requires a live OpenAI-compatible endpoint
pytestmark = pytest.mark.integration
from dotenv import load_dotenv

import daft
from daft.functions import llm_generate

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # Tests will skip if OpenAI client is unavailable


# Load local .env for developer machines; CI should use environment/secrets
load_dotenv()


def _get_env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value


def _get_openai_client():
    if OpenAI is None:
        pytest.skip("openai package not available; skipping integration tests")
    base_url = _get_env("OPENAI_BASE_URL", "http://0.0.0.0:8000/v1")
    api_key = _get_env("OPENAI_API_KEY", "none")
    return OpenAI(base_url=base_url, api_key=api_key)


def _ensure_server_or_skip() -> None:
    client = _get_openai_client()
    try:
        _ = client.models.list()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Skipping: vLLM OpenAI server not reachable ({exc})")


@pytest.fixture(scope="module", autouse=True)
def daft_session():
    with daft.session() as session:
        session.set_provider("openai", api_key=_get_env("OPENAI_API_KEY", "none"))
        yield session


def _model_id() -> str:
    return _get_env("MODEL_ID", "google/gemma-3n-e4b-it")


def _base_url() -> str:
    return _get_env("OPENAI_BASE_URL", "http://0.0.0.0:8000/v1")


def _api_key() -> str:
    return _get_env("OPENAI_API_KEY", "none")


def test_guided_choice_returns_one_of_choices():
    _ensure_server_or_skip()

    df = daft.from_pylist([
        {"text": "Classify this sentiment: Daft is fast!"},
    ])

    choices = ["positive", "negative"]
    df_with_choice = df.with_column(
        "result",
        llm_generate(
            df["text"],
            model=_model_id(),
            provider="openai",
            extra_body={"guided_choice": choices},
            base_url=_base_url(),
            api_key=_api_key(),
        ),
    )

    rows = df_with_choice.to_pylist()
    output = rows[0]["result"].strip()
    assert output in choices


def test_guided_regex_matches_pattern():
    _ensure_server_or_skip()

    df = daft.from_pylist([
        {
            "text": (
                "Generate an email address for Alan Turing, who works at Enigma. "
                "End in .com and new line. Example result: 'alan.turing@enigma.com\\n'"
            )
        }
    ])

    # Allow optional trailing newline
    pattern = re.compile(r"^[a-z0-9.]{1,20}@\w{6,10}\.com\n?$")

    df_with_email = df.with_column(
        "result",
        llm_generate(
            df["text"],
            model=_model_id(),
            provider="openai",
            extra_body={"guided_regex": r"[a-z0-9.]{1,20}@\\w{6,10}\\.com\\n"},
            base_url=_base_url(),
            api_key=_api_key(),
        ),
    )

    rows = df_with_email.to_pylist()
    output = rows[0]["result"]
    assert pattern.match(output) is not None, f"Output {output!r} did not match regex"