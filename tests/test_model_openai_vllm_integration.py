import os
import re
from typing import Optional

import pytest

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package is required for integration tests") from exc


def _get_client() -> OpenAI:
    """Construct an OpenAI client pointed at the vLLM server.

    Reads OPENAI_BASE_URL and OPENAI_API_KEY from environment.
    Defaults align with the project's makefile.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "http://0.0.0.0:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "none")
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_model_id() -> str:
    """Resolve the model id to target for tests.

    Checks MODEL, then OPENAI_MODEL_ID, then defaults to gemma 3n IT.
    """
    return (
        os.environ.get("MODEL")
        or os.environ.get("OPENAI_MODEL_ID")
        or "google/gemma-3n-e4b-it"
    )


def _ensure_server_or_skip(client: OpenAI) -> None:
    """Skip tests if the server is unreachable or not yet started."""
    try:
        # Small, fast call to verify connectivity
        _ = client.models.list()
    except Exception as exc:
        pytest.skip(f"Skipping: vLLM OpenAI server not reachable ({exc})")


def _get_timeout_client(client: OpenAI, timeout_s: float = 30.0) -> OpenAI:
    """Return a client with a per-request timeout."""
    return client.with_options(timeout=timeout_s)


def _get_test_image_url() -> Optional[str]:
    return os.environ.get("TEST_IMAGE_URL")


def test_models_list_returns_models():
    """Models listing returns at least one model from the vLLM OpenAI endpoint."""
    client = _get_client()
    _ensure_server_or_skip(client)

    models = _get_timeout_client(client, 10).models.list()
    assert hasattr(models, "data"), "Response should have a 'data' attribute"
    assert isinstance(models.data, list) and len(models.data) >= 1


def test_chat_text_only_returns_content():
    """Basic chat completion with text-only prompt returns non-empty content."""
    client = _get_client()
    _ensure_server_or_skip(client)
    model_id = _get_model_id()

    chat = _get_timeout_client(client, 30).chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "What's the coolest thing about daft dataframes?"}
        ],
    )
    content = chat.choices[0].message.content
    assert isinstance(content, str) and content.strip(), "Expected non-empty string content"


def test_chat_guided_choice_is_one_of_choices():
    """Chat completion with guided_choice returns one of the specified options."""
    client = _get_client()
    _ensure_server_or_skip(client)
    model_id = _get_model_id()

    choices = ["positive", "negative"]
    completion = _get_timeout_client(client, 30).chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "Classify this sentiment: Daft is wicked fast!"}
        ],
        extra_body={"guided_choice": choices},
    )
    content = completion.choices[0].message.content.strip()
    assert content in choices, f"Output '{content}' not in {choices}"


def test_chat_guided_regex_matches_pattern():
    """Chat completion with guided_regex conforms to the provided regex pattern."""
    client = _get_client()
    _ensure_server_or_skip(client)
    model_id = _get_model_id()

    # Allow optional trailing newline
    regex = re.compile(r"^\w+@\w+\.com\n?$")

    completion = _get_timeout_client(client, 30).chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate an example email address for Alan Turing, who works at Enigma. "
                    "End in .com and new line. Example result: alan.turing@enigma.com\n"
                ),
            }
        ],
        extra_body={"guided_regex": r"\w+@\w+\.com\n", "stop": ["\n"]},
    )
    content = completion.choices[0].message.content
    assert regex.match(content) is not None, f"Output '{content!r}' did not match regex"


@pytest.mark.skipif(
    not _get_test_image_url(), reason="Set TEST_IMAGE_URL to exercise vision test"
)
def test_chat_vision_when_image_url_set():
    """Vision chat completion returns text content when TEST_IMAGE_URL is provided.

    This test is optional and will be skipped unless TEST_IMAGE_URL is set.
    """
    client = _get_client()
    _ensure_server_or_skip(client)
    model_id = _get_model_id()
    image_url = _get_test_image_url()

    completion = _get_timeout_client(client, 60).chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ],
    )
    content = completion.choices[0].message.content
    assert isinstance(content, str) and content.strip(), "Expected non-empty string content"


