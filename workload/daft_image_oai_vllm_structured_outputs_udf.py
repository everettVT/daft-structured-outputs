"""

For running gemma-3n-e4b-it with vllm, you need to run the following command (A100 recommended):
 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3n-e4b-it \
  --guided-decoding-backend guidance \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000


"""
# Import Dependencies
import asyncio
import base64
import daft
from daft import col, lit
from daft.functions import format, llm_generate
from openai import OpenAI, AsyncOpenAI
import time

# Define the UDF ---------------------------------------------------------------
@daft.udf(return_dtype=daft.DataType.string())
class StructuredOutputsProdUDF:
    def __init__(self, base_url: str, api_key: str, max_conn: int = 32):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_conn)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)


    def __call__(self, model_id: str, text_col: daft.Series, image_col: daft.Series, extra_body: dict) -> list[str]:

        async def generate(text: str, image: str) -> str:
                content = [{"type": "text", "text": text}]
                if image:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    })

                result = await self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    model=model_id,
                    extra_body=extra_body
                )
                return result.choices[0].message.content

        async def infer_with_semaphore(t, i):
            async with self.semaphore:
                return await generate(t,i)

        async def gather_completions(texts,images) -> list[str]:
            tasks = [infer_with_semaphore(t,i) for t,i in zip(texts,images)]
            return await asyncio.gather(*tasks)

        texts = text_col.to_pylist()
        images = image_col.to_pylist()

        return self.loop.run_until_complete(gather_completions(texts,images))


class DaftStructuredOutputsBench:
    def __init__(self, model_id: str, base_url: str, api_key: str, max_conn: int = 32, num_rows: int = 200):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_conn = max_conn

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

if __name__ == "__main__":
    # Load Environment Variables ---------------------------------------------------
    import os 
    from dotenv import load_dotenv

    load_dotenv()

    # Define Variables ------------------------------------------------------------
    model_id = 'google/gemma-3n-e4b-it'
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    concurrency = 4
    max_conn = 32 

    # Step 1 - Load Dataset --------------------------------------------------------
    df_raw = daft.read_parquet('hf://datasets/HuggingFaceM4/the_cauldron/ai2d/train-00000-of-00001-2ce340398c113b79.parquet')

    # Step 2 - Preprocessing -------------------------------------------------------
    # Convert byte string to base64
    df = df_raw.explode(col("images")).with_column("image_base64", df_raw["images"].struct.get("bytes").apply(
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
    df_prepped = df.with_columns({
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
    start = time.time()
    df_prod_udf = df_prepped.with_column("result", StructuredOutputsProdUDF.with_init_args(
        base_url=base_url,
        api_key=api_key,
        max_conn=max_conn
    ).with_concurrency(concurrency)(
        model_id = model_id,
        text_col = format("{} \n {}", col("question"), col("choices_string")), # Prompt Template
        image_col = col("image_base64"),
        extra_body={"guided_choice": ["A", "B", "C", "D"]}
    )).collect()
    end = time.time()
    print(f"Batch UDF (Image + Text + Prompt Template) \n Processed {df_prod_udf.count_rows()} rows in {end-start} seconds")
     

    # Step 4 - Evaluation ----------------------------------------------------------
    df_eval = df_prod_udf.with_column("is_correct", col("result").str.lstrip().str.rstrip() == col("answer").str.lstrip().str.rstrip())


    # Calculate Pass/Fail Rate 
    pass_fail_rate = df_eval.where(col("is_correct")).count_rows() / df_eval.count_rows()
    print(f"Pass/Fail Rate: {pass_fail_rate}")


