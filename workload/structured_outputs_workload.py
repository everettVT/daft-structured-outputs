"""

For running gemma-3n-e4b-it on vllm online server, you need to run the following command:
 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3n-e4b-it \
  --guided-decoding-backend guidance \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000

The above memory utilization was configured for google colab's A100  Instance: 

If you are using a different instance, you WILL need to adjust the memory utilization accordingly.
"""
# Import Dependencies & Define Variables

import time
from typing import Any
import asyncio
import base64

import daft
from daft import col, lit
from daft.functions import format
from openai import AsyncOpenAI

import logging

logger = logging.getLogger(__name__)
@daft.udf(return_dtype=daft.DataType.string(), concurrency=4)
class StructuredOutputsProdUDF:
    def __init__(self, base_url: str, api_key: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)


    def __call__(self,
        model_id: str,
        text_col: daft.Series,
        image_col: daft.Series,
        sampling_params: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None
        ):

        async def generate(text: str, image: str) -> str:
                content = []
                if image:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    })
                if text:
                    content.append({"type": "text", "text": text})

                result = await self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": content # Dataset prefers image first
                        }
                    ],
                    model=model_id,
                    extra_body=extra_body,
                    **sampling_params
                )
                return result.choices[0].message.content

        async def infer_with_semaphore(t, i):
            return await generate(t,i)

        async def gather_completions(texts,images) -> list[str]:
            tasks = [infer_with_semaphore(t,i) for t,i in zip(texts,images)]
            return await asyncio.gather(*tasks)

        texts = text_col.to_pylist()
        images = image_col.to_pylist()

        return self.loop.run_until_complete(gather_completions(texts,images))

class TheCauldronImageUnderstandingEvaluationPipeline:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def __call__(self,
        model_id: str,
        dataset_uri: str,
        sampling_params: dict[str,Any] | None = None,
        concurrency: int = 4,
        row_limit: int | None = None,
        is_eager: bool = False,
    ) -> daft.DataFrame:
        """Executes dataset loading, preprocessing, inference, and post-processing.
        Evalutation must be run seperately since it requires materialization. 

        Args:
            model_id: The ID of the model to use
            dataset_uri: The URI of the dataset to use
            sampling_params: The sampling parameters to use
            concurrency: The number of concurrent requests to make
            row_limit: The number of rows to limit the dataset to
            is_eager: Whether to eager load the dataset
        """

        if is_eager:
            # Load Dataset and Materialize
            df = self.load_dataset(dataset_uri)
            df = df.limit(row_limit) if row_limit else df
            df = self._log_processing_time(df)

            # Preprocess
            df = self.preprocess(df)
            df = self._log_processing_time(df)

            # Perform Inference
            df = self.infer(df, model_id, sampling_params)
            df = self._log_processing_time(df)

            # Post-Process
            df = self.postprocess(df)
            df = self._log_processing_time(df)
        else:
            df = self.load_dataset(dataset_uri)
            df = self.preprocess(df)
            df = self.infer(df, model_id, sampling_params)
            df = self.postprocess(df)
            df = df.limit(row_limit) if row_limit else df

        return df

    @staticmethod
    def _log_processing_time(df: daft.DataFrame):
        start = time.time()
        df_materialized = df.collect()
        end = time.time()
        num_rows = df_materialized.count_rows()
        logger.info(f"Processed {num_rows} rows in {end-start} sec, {num_rows/(end-start)} rows/s")
        return df_materialized

    def load_dataset(self, uri: str) -> daft.DataFrame:
        return daft.read_parquet(uri)

    def preprocess(self, df: daft.DataFrame) -> daft.DataFrame:

        # Convert png image byte string to base64
        df = df.explode(col("images")).with_column("image_base64", df["images"].struct.get("bytes").apply(
        lambda x: base64.b64encode(x).decode('utf-8'),
        return_dtype=daft.DataType.string()
            )
        )

        # Explode Lists of User Prompts and Assistant Answer Pairs
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
        return df

    def infer(self,
        df: daft.DataFrame,
        model_id: str = 'google/gemma-3n-e4b-it',
        sampling_params: dict[str,Any] = {"temperature": 0.0},
        concurrency: int = 4,
        extra_body: dict[str, Any] = {"guided_choice": ["A", "B", "C", "D"]}
    ) -> daft.DataFrame:

        return df.with_column("result", StructuredOutputsProdUDF.with_init_args(
            base_url=self.base_url,
            api_key=self.api_key,
        ).with_concurrency(concurrency)(
            model_id = model_id,
            text_col = format("{} \n {}", col("question"), col("choices_string")), # Prompt Template
            image_col = col("image_base64"),
            sampling_params = sampling_params,
            extra_body=extra_body
        ))


    def postprocess(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column("is_correct", col("result").str.lstrip().str.rstrip() == col("answer").str.lstrip().str.rstrip())
        return df

    def evaluate(self, df: daft.DataFrame) -> float:
        pass_fail_rate = df.where(col("is_correct")).count_rows() / df.count_rows()
        return pass_fail_rate

if __name__ == "__main__":
    # Load Environment Variables 
    import os 
    from dotenv import load_dotenv

    load_dotenv()

    # Define Variables
    model_id = 'google/gemma-3n-e4b-it'
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    dataset_uri = 'hf://datasets/HuggingFaceM4/the_cauldron/ai2d/train-00000-of-00001-2ce340398c113b79.parquet' # 7462 rows
    concurrency = 4
    row_limit = 10

    # Instantiate the pipeline
    pipeline = TheCauldronImageUnderstandingEvaluationPipeline(
        api_key  = api_key, 
        base_url = base_url
    )

    df = pipeline(
        model_id = model_id, 
        dataset_uri = dataset_uri, 
        sampling_params={"temperature": 0.0}, 
        row_limit = row_limit,
        concurrency = concurrency,
        is_eager=False,
    )

    # Materialize the dataframe
    df = df.collect() # Optionally measure performance with pipeline._log_processing_time(df)

    # Evaluate the results
    pass_fail_rate = pipeline.evaluate(df)
    print(f"Pass/Fail Rate: {pass_fail_rate}")




