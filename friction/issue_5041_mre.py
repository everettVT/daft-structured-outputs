import os
import daft
from daft.functions import llm_generate
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

p = "daft dataframes"
df = daft.from_pylist([
    {"prompt": p},
])


# try openai
response = OpenAI().chat.completions.create(
    model="gpt-5-nano-2025-08-07",
    messages=[
        {"role": "user", "content": f"Generate a short made-up story about: {p}"}
    ]
)
print(response.choices[0].message.content)


# with OPENAI_API_KEY set
df = df.with_column("results", llm_generate(
    "Generate a short made-up story about:"+ df["prompt"],
    model="gpt-5-nano-2025-08-07",
    provider="openai",
))
df.collect()
print(df.to_pydict())