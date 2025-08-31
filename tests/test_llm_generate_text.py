import daft
from daft import col 
from daft.functions import llm_generate, format 

import pytest





@pytest.fixture(scope="module", autouse=True)
def skip_no_credential(pytestconfig):
    if not pytestconfig.getoption("--credentials"):
        pytest.skip(reason="OpenAI integration tests require the `--credentials` flag.")
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(reason="OpenAI integration tests require the OPENAI_API_KEY environment variable.")

    

@pytest.fixture(scope="module", autouse=True)
def session(skip_no_credential):
    """Configures the session to be used for all tests."""
    with daft.session() as session:
        # the key is not explicitly needed, but was added with angry lookup for clarity.
        session.set_provider("openai", api_key=os.environ["OPENAI_API_KEY"])
        yield session

        


df = daft.from_pylist([
    {"text":"Classify this sentiment: Daft is fast!"},
])

df_result = df.with_column("result", llm_generate(
        df["text"],
        model=model_id,
        provider="openai",
        extra_body={"guided_choice": ["positive", "negative"]},
        base_url=base_url,
        api_key=api_key
    )
)

df_result.to_pylist()