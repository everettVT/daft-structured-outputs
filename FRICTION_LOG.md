# Friction Log - Implementing Structured Outputs with Daft

- Date: Aug 25, 2025
- Author: Everett Kleven
- Size: XL
- Persona: UDF Naive User
- Notebook: [Daft Canonical Workload - Multimodal Structured Outputs with Gemma 3 and vLLM](https://colab.research.google.com/drive/1-AJWGeTPhtIK_uXngPmxMFRePlU5rKRU#scrollTo=BOa4WVnVKJoD)


### Issues Created: 
- [5083](https://github.com/Eventual-Inc/Daft/issues/5083) Improve error messages for improper UDF initialization
- [5088](https://github.com/Eventual-Inc/Daft/issues/5088) Setting concurrency or batch_size on python functions decorated with @daft.udf raises RuntimeError on daft native runner
- [5089](https://github.com/Eventual-Inc/Daft/issues/5089) Add base64 encoding support for daft.DataType.Image()

## Context
This Friction Log Covers the many pain points and discoveries I made along the way while getting structured ouputs over images to work on vllm with gemma-3n-e4b-it. Structured outputs form the foundation of many of the most important use cases that AI powers today (Synthetic Data, Document OCR, Tool Use).  Many companies have deploy inference engines internally as a means of controlling costs, and fill idle GPUs utilitzation to maximize commited use discounts.  

Simply stated, enabling structured generation for the average openai api compatible client is trivial, but implementing an example on your own inference engine is not. Adding image inputs adds another layer of complexity, forcing engineers to encode images into base64 data urls if they plan to load image data themselves. Once you've figured all of that out, then you can begin developing your workload in daft, which comes with it's own set of challenges.

I approached documenting friction points from the expectation that Daft makes it trivially easy for the naive user to implement structured outputs on images. While aspirational, this perspective helped identify every task that consumed my cognitive load. 

# Friction Points
The [unfiltered notebook](daft-structured-outputs/friction/full_notebook_unfiltered.ipynb) captures my experience end-to-end building the workload iteself. There is probably a hidden 10 hours of combined pain dedicated to just vllm alone. Ray 

Daft scales easily, too easily as it turns out. The `asyncio.event_loop` can become starved  limitations on the openai client can create unexpected errors. This combined The simplest implementations of row-wise and batch user-defined functions both failed to process more than 2000 rows with default concurrency, and with   

  I ran into few bugs and friction points that I've since created issues for (listed above).
 
## Raw Take

There is quite a bit to unpack here. Overall I'd say I ran into just as many issues trying to get vllm setup correctly than I did working through the workload itself. The shear number of hidden incompatibilities across models, engine configuration parameters, and openai client args far exceeds the average cognitive load of single day.

It takes about 7.5 minutes for vllm to be ready, and installing the library alone takes several minutes. That means that if you misconfigure or need to tweak any of the runtime options, you are looking at least a minimum of a 10 minute redo cycle. This idle time hurts.. excuse me... It fucking sucks. Like no wonder inference providers have a moat! It makes a lot of sense that Ray has decided to partner with vllm now, integrated it into ray serve natively. 

I really wish I didn't spend as much time just getting vllm to run a model I could use for this workload, but it turned out to be highly non-trivial. I'm not even talking about optimization here. The Engine config for vllm has something like 10 billion cli args on it and if you don't read the fine print for the model card on HF, it could take you hours to discover  your model has no default chat-template, or doesn't support images, or isn't trained to handle structured outputs well.

While I initally really wanted to work with gemma-3n-270m, a tiny model operating at the frontier of intelligence cost performance, I eventually went with gemma-3n-e4b-it. It supports text, image, and audio inputs and can run on an L4 if your life depended on it. Naturally if you want decent inference speeds you size up, so I went with an A100 which turned out to do just fine.  


Not having Copy/Paste tutorials for implementing structured outputs made it confusing to decouple issues between daft and vllm at times, but in the end I think we arrived at something that looks reasonable. 

Friction points: 
- Class-based UDF initialization syntax -> Could use better error messages [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)
- No Base64 encoding for image dtype [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)
- Concurrency Param bug for python function based udfs with OpenAI client [Issue 5088](https://github.com/Eventual-Inc/Daft/issues/5088)
- Event loop Exceptions when scaling inference without a set event loop. 
- Hanging Infernce calls when running 2000+ rows

