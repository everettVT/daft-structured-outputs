# Import Dependencies
import asyncio
import base64
import daft
from daft import col, lit
from daft.functions import format, llm_generate
from openai import OpenAI, AsyncOpenAI
import time

# Step 0 Define Variables and Inference UDF ------------------------------------
model_id = 'google/gemma-3n-e4b-it'
api_key = "none"
base_url = "http://0.0.0.0:8000/v1"
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

@daft.func(return_dtype=daft.DataType.string())
async def struct_output_rowwise(model_id: str, text_col: str, image_col: str, extra_body: dict | None = None) -> str:

    content = [{"type": "text", "text": text_col}]
    if image_col:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_col}"},
        })


    result = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        model=model_id,
        extra_body=extra_body,
    )
    return result.choices[0].message.content

@daft.udf(return_dtype=daft.DataType.string())
def struct_output_batch(
        model_id: str,
        text_col: daft.Series,
        image_col: daft.Series,
        extra_body: dict | None = None
    ) -> list[str]:


    async def generate(model_id: str, text: str, image: str) -> str:

        content = [{"type": "text", "text": text}]
        if image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            })

        result = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model=model_id,
            extra_body=extra_body,
        )
        return result.choices[0].message.content

    texts = text_col.to_pylist()
    images = image_col.to_pylist()

    async def gather_completions() -> list[str]:
        tasks = [generate(model_id, t,i) for t,i in zip(texts,images) ]
        return await asyncio.gather(*tasks)

    return asyncio.run(gather_completions())

# Step 1 - Load Dataset --------------------------------------------------------
df = daft.read_parquet('hf://datasets/HuggingFaceM4/the_cauldron/ai2d/train-00000-of-00001-2ce340398c113b79.parquet')

# Step 2 - Preprocessing -------------------------------------------------------
# Convert byte string to base64
df = df.explode(col("images")).with_column("image_base64", df["images"].struct.get("bytes").apply(
        lambda x: base64.b64encode(x).decode('utf-8'),
        return_dtype=daft.DataType.string()
    )
)

# Explode Questions
df = df.explode(col("texts")).with_columns({
    "user": df["texts"].struct.get("user"),
    "assistant": df["texts"].struct.get("assistant")
})

# Parse the Question/Answer Strings
df = df.with_columns({
    "question": df["user"]
        .str.extract(r"(?s)Question:\s*(.*?)\s*Choices:")
        .str.replace("Choices:", "")
        .str.replace("Question:",""),
    "choices_string": df["user"]
        .str.extract(r"(?s)Choices:\s*(.*?)\s*Answer?\.?")
        .str.replace("Choices:\n", "")
        .str.replace("Answer",""),
    "answer": df["assistant"]
        .str.extract(r"Answer:\s*(.*)$")
        .str.replace("Answer:",""),
})


# Step 3 - Inference -----------------------------------------------------------
df = df.with_column("result", struct_output_batch(
                                model_id = model_id,
                                text_col = format("{} \n {}", col("question"), col("choices_string")), # Prompt Template
                                image_col = col("image_base64"),
                                extra_body={"guided_choice": ["A", "B", "C", "D"]}
))

# Step 4 - Evaluation ----------------------------------------------------------
df = df.with_column("is_correct", col("result").str.lstrip().str.rstrip() == col("answer").str.lstrip().str.rstrip())



## Materialization ------------------------------------------------------------
start = time.time()
df_eval = df.limit(200).collect() # Limit Num Rows as desired
end = time.time()
num_rows = df_eval.count_rows()
print(f"Processed {num_rows} rows in {end-start} seconds")


# Calculate
pass_fail_rate = df_eval.where(col("is_correct")).count_rows() / df_eval.count_rows()
print(f"Pass/Fail Rate: {pass_fail_rate}")


# Validation 