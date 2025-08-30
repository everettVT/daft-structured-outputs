import asyncio
import daft
from daft import col
from openai import AsyncOpenAI
from dotenv import load_dotenv



@daft.udf(return_dtype= daft.DataType.string(), concurrency=3)
def sync_udf(model_id: str, text_col: daft.Series, image_col: daft.Series): 
    
    def combine(model_id: str, text: str, image: str) -> str:
        return model_id + text + image
    
    texts = text_col.to_pylist()
    images = image_col.to_pylist()

    return [combine(model_id,t,i) for t,i in zip(texts,images)]

# Works
@daft.udf(return_dtype= daft.DataType.string(), concurrency=3)
def async_udf(model_id: str, text_col: daft.Series, image_col: daft.Series): 
    
    async def combine(model_id: str, text: str, image: str) -> str:
        return model_id + text + image
    
    async def combine_all(model_id: str, texts: list[str], images: list[str]) -> list[str]:
        tasks = [combine(model_id,t,i) for t,i in zip(texts,images)]
        return await asyncio.gather(*tasks)

    texts = text_col.to_pylist()
    images = image_col.to_pylist()
    return asyncio.run(combine_all(model_id, texts, images))


@daft.udf(return_dtype=daft.DataType.string())
def image_inference_no_concurrency(
        model_id: str,
        text_col: daft.Series,
        image_col: daft.Series,
    ) -> list[str]:


    async def generate(model_id: str, text: str, image: str) -> str:

        content = [{"type": "text", "text": text}]
        if image:
            content.append({
                "type": "image_url",
                "image_url": {"url": image},
            })

        result = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model=model_id,
        )
        return result.choices[0].message.content

    texts = text_col.to_pylist()
    images = image_col.to_pylist()

    async def gather_completions() -> list[str]:
        tasks = [generate(model_id, t,i) for t,i in zip(texts,images) ]
        return await asyncio.gather(*tasks)

    return asyncio.run(gather_completions())



@daft.udf(return_dtype=daft.DataType.string(), concurrency=3)
class ImageInferenceWithConcurrencyClassUDF:
    def __init__(self, base_url: str, api_key: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, model_id: str, text_col: daft.Series, image_col: daft.Series) -> list[str]:

        async def generate(model_id: str, text: str, image: str) -> str:
            content = [{"type": "text", "text": text}]
            if image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image},
                })

            result = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                model=model_id,
            )
            return result.choices[0].message.content

        texts = text_col.to_pylist()
        images = image_col.to_pylist()

        async def gather_completions() -> list[str]:
            tasks = [generate(model_id, t,i) for t,i in zip(texts,images) ]
            return await asyncio.gather(*tasks)

        return asyncio.run(gather_completions())


@daft.udf(return_dtype=daft.DataType.string(), concurrency=3)
def image_inference_with_concurrency(
        model_id: str,
        text_col: daft.Series,
        image_col: daft.Series,
    ) -> list[str]:


    async def generate(model_id: str, text: str, image: str) -> str:

        content = [{"type": "text", "text": text}]
        if image:
            content.append({
                "type": "image_url",
                "image_url": {"url": image},
            })

        result = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model=model_id,
        )
        return result.choices[0].message.content

    texts = text_col.to_pylist()
    images = image_col.to_pylist()

    async def gather_completions() -> list[str]:
        tasks = [generate(model_id, t,i) for t,i in zip(texts,images) ]
        return await asyncio.gather(*tasks)

    return asyncio.run(gather_completions())

if __name__ == "__main__":
    load_dotenv()
    import os 

    model_id = 'gpt-5-nano'
    client = AsyncOpenAI()
    prompt = "what is in this image?"
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


    df = daft.from_pylist([{
        "user": prompt,
        "image_url": image_url
        }])
    
    # Works
    df1 = df.with_column("result", sync_udf(
                                        model_id = model_id,
                                        text_col = col("user"),
                                        image_col = col("image_url"),
        )).collect()
    
    # Works
    df2 = df.with_column("result", async_udf(
                                        model_id = model_id,
                                        text_col = col("user"),
                                        image_col = col("image_url"),
        )).collect()

    # Works
    df3 = df.with_column("result", image_inference_no_concurrency(
                                        model_id = model_id,
                                        text_col = col("user"), 
                                        image_col = col("image_url"),
        )).collect()
    
    # Works
    df4 = df.with_column("result", ImageInferenceWithConcurrencyClassUDF.with_init_args(
        base_url = os.getenv("OPENAI_BASE_URL"),
        api_key = os.getenv("OPENAI_API_KEY"),
    )(
                                        model_id = model_id,
                                        text_col = col("user"), 
                                        image_col = col("image_url"),
    )).collect()
    
    # Doesn't work
    df5 = df.with_column("result", image_inference_with_concurrency(
                                        model_id = model_id,
                                        text_col = col("user"),
                                        image_col = col("image_url"),
        )).collect()

