# Friction Log - Implementing Structured Outputs with Daft

- Date: Aug 25, 2025
- Author: Everett Kleven
- Size: XL
- Persona: UDF Naive User
- Notebooks: 
    * [Daft Canonical Workload - Multimodal Structured Outputs with Gemma 3 and vLLM](https://colab.research.google.com/drive/1-AJWGeTPhtIK_uXngPmxMFRePlU5rKRU#scrollTo=BOa4WVnVKJoD)
  
## Summary
Structured Ouputs work with Daft, but usage patterns aren't immediately obvious without examples. Multimodality currently isn't supported in `llm_generate` and a `daft.DataType.Image()` utility for base64 encoding would be helpful. Due to this, I had to build my own class-based batch UDF which took a while and turned out to be non-trivial. Initial attempts to use a row-wise udfs yielded poor performance, however transitioning the udf to Async Batch yielded unexpected errors and scaling issues. This combined with the complexity of implementing inference as a part of an actual non-trivial workload led to high cognitive load. Overall, due to the complex interplay of marrying multiple interfaces and usage patterns, there is significant opportunity to streamline vectorized structured outputs for multimodal worklaods. 

## Context
This Friction Log Covers the many pain points and discoveries I made along the way while getting structured ouputs over images to work on vllm with gemma-3n-e4b-it using the online OpenAI compatible server. Structured outputs form the foundation of many of the most important use cases that AI powers today (Synthetic Data, Document OCR, Tool Use) and represents the most important non-default feature for batch AI workloads.

Why not just use OpenRouter or OpenAI directly? Companies deploy inference engines internally as a means of controlling costs, specifically to fill idle GPUs utilitzation on commited use discounts. Focusing on vllm's OpenAI compatible server helps us focus on the use-case that most companies actually face in scaling inference. Additionally, enabling structured generation for the average openai api compatible client is trivial, but implementing an example on your own inference engine is not.

Adding image inputs adds another layer of complexity, forcing engineers to encode images into base64 data urls if they plan to load image data themselves. Once you've figured all of that out, then you can begin developing your workload in daft, which comes with it's own set of challenges. This combination of factors make Local Structured Output Serving a highly intesive cognitive task for any developer and represents a use-case with real end-user pain worth solving.

I approached documenting friction points from the expectation that Daft makes it trivially easy for the naive user to implement structured outputs on images. While aspirational, this perspective helped identify every task that consumed my cognitive load and provides the basis of expectation that new users have in our AI era.

## Friction Points
The [unfiltered notebook](daft-structured-outputs/friction/full_notebook_unfiltered.ipynb) captures my experience end-to-end building the structured outputs workload iteself. It represents the full amount of effort that it took to develop the final workload, consuming dozens of hours. While there were many small inconveniences and learnings I experienced along the way, the following friction points stood out as urgent priorities worth focusing on. 

### Class-based UDF initialization syntax -> Could use better error messages [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)


###  No Base64 encoding for image dtype [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)

Image inputs work for image urls or data urls encoded with base64. Since Daft has the Image Datatype, we can steamline this preprocessing step so that users can pass in any column with an Image DataType to inference functions for fast composition. This may also inform the design of the new resources concept for new UDF designs or the `daft.File` object. Given that there are really two ways of provisioning images to inference servers in a distributed environment I imagine the team already has ideas for this.

###  Concurrency Param bug for python function based udfs with OpenAI client [Issue 5088](https://github.com/Eventual-Inc/Daft/issues/5088)

For some reason, when we use an openai client inside of a UDF, we can't add the concurrency parameter for function based user defined funtions. We get this runtime error referring to pickle serialization. I ran into this while I was initially developing the batch udf. I also started a [thread on slack](https://dist-data.slack.com/archives/C052CA6Q9N1/p1756400464828409) due to my confusion on whether or not the daft.func supported a concurrency argument. 


###  Event loop Exceptions when scaling inference without a set event loop and Hanging Inference calls when running 2000+ rows

For the average user who is looking to leverage daft to perform ai inference using an http client, most users will try either a row_wise UDF or a synchronous Batch UDF. These implementations work at small scale but run into issues once users attempt to run them at 2000 + rows. Regardless of how they arrive at the conclusion, eventually they will attempt to run their inference calls asynchronously which will non-blocking errors at the 200-1500 row limit range. Essentially the client will return exceptions it see's, but completes the entire workload, failing after the full job is completed. This means that if you have a follow on transformation the code never executes and the process exits throwing a long errors chain. 

Daft scales easily, too easily as it turns out. The `asyncio.event_loop` can become starved, most likely from limitations in the openai client, which can create opaque errors. These errors only emerge when you try to scale This combined The simplest implementations of row-wise and batch user-defined functions both failed to process more than 2000 rows with default concurrency. Once I added the event loop attachment logic in a class based udf, the issues went away and I was able to process the full ~8000 row dataset. 

### Issues Created: 
- [5083](https://github.com/Eventual-Inc/Daft/issues/5083) Improve error messages for improper UDF initialization
- [5088](https://github.com/Eventual-Inc/Daft/issues/5088) Setting concurrency or batch_size on python functions decorated with @daft.udf raises RuntimeError on daft native runner
- [5089](https://github.com/Eventual-Inc/Daft/issues/5089) Add base64 encoding support for daft.DataType.Image()

 
## Raw Take

Overall I'd say I ran into just as many issues trying to get vllm setup correctly than I did working through the workload itself. The shear number of hidden incompatibilities across models, engine configuration parameters, and openai client args far exceeds the average cognitive load of single day.

It takes about 7.5 minutes for vllm to be ready, and installing the library alone takes several minutes. That means that if you misconfigure or need to tweak any of the runtime options, you are looking at least a minimum of a 10 minute redo cycle.

I really wish I didn't spend as much time just getting vllm to run a model I could use for this workload, but it turned out to be highly non-trivial. I'm not even talking about optimization here. The Engine config for vllm has something like 10 billion cli args on it and if you don't read the fine print for the model card on HF, it could take you hours to discover  your model has no default chat-template, or doesn't support images, or isn't trained to handle structured outputs well.

While I initally really wanted to work with gemma-3n-270m, a tiny model operating at the frontier of intelligence cost performance, I eventually went with gemma-3n-e4b-it. It supports text, image, and audio inputs and can run on an L4 if your life depended on it. Naturally if you want decent inference speeds you size up, so I went with an A100 which turned out to do just fine.  


Not having Copy/Paste tutorials for implementing structured outputs made it confusing to decouple issues between daft and vllm at times, but in the end I think we arrived at something that looks reasonable. 

Friction points: 


