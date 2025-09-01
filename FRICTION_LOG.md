# Friction Log - Implementing Structured Outputs with Daft

- Date: Aug 25, 2025
- Author: Everett Kleven
- Size: L
- Persona: UDF Naive User
- [Notebook](/workload/Daft_Canonical_Workload_Gemma3_vLLM.ipynb)
- [Python Script](/workload/structured_outputs_workload.py)
  
## Summary
Structured Ouputs work with Daft, but usage patterns aren't immediately obvious without examples. Multimodality currently isn't supported in `llm_generate` and a `daft.DataType.Image()` utility for base64 encoding would be helpful. Due to this, I had to build my own class-based batch UDF which took a while and turned out to be non-trivial. Initial attempts to use a row-wise udfs yielded poor performance, however transitioning the udf to Async Batch yielded unexpected errors and scaling issues. This combined with the complexity of implementing inference as a part of an actual non-trivial workload led to high cognitive load. Overall, due to the complex interplay of marrying multiple interfaces and usage patterns, there is significant opportunity to streamline vectorized structured outputs for multimodal worklaods. 

## Context
This Friction Log Covers the many pain points and discoveries I made along the way while getting structured ouputs over images to work on vllm with gemma-3n-e4b-it using the online OpenAI compatible server. Structured outputs form the foundation of many of the most important use cases that AI powers today (Synthetic Data, Document OCR, Tool Use) and represents the most important non-default feature for batch AI workloads.

Why not just use OpenRouter or OpenAI directly? Companies deploy inference engines internally as a means of controlling costs, specifically to fill idle GPUs utilitzation on commited use discounts. Focusing on vllm's OpenAI compatible server centers our workload on a real use-case that most companies actually face in scaling inference. Additionally, enabling structured generation for the average openai api compatible client is trivial, but implementing an example on your own inference engine is not.

Adding image inputs adds another layer of complexity, forcing engineers to encode images into base64 data urls if they plan to load image data themselves. Once you've figured all of that out, then you can begin developing your workload in daft, which comes with it's own set of challenges. This combination of factors make Local Structured Output Serving a highly intesive cognitive task for any developer and represents a use-case with real end-user pain worth solving.

I approached documenting friction points from the expectation that Daft makes it trivially easy for the naive user to implement structured outputs on images. While aspirational, this perspective helped identify every task that consumed my cognitive load and provides the basis of expectation that new users have in our AI era.

## Friction Points
The [unfiltered notebook](daft-structured-outputs/friction/full_notebook_unfiltered.ipynb) captures my experience end-to-end building the structured outputs workload iteself. It represents the full amount of effort that it took to develop the final workload, consuming dozens of hours. While there were many small inconveniences and learnings I experienced along the way, the following friction points stood out as urgent priorities worth focusing on. 

### [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089) - No Base64 encoding for image dtype

Image inputs work for image urls or data urls encoded with base64. Since Daft has the Image Datatype, we can steamline this preprocessing step so that users can pass in any column with an Image DataType to inference functions for fast composition. This may also inform the design of the new resources concept for new UDF designs or the `daft.File` object. Given that there are really two ways of provisioning images to inference servers in a distributed environment I imagine the team already has ideas for this.

### [Issue 5083](https://github.com/Eventual-Inc/Daft/issues/5083) - Class-based UDF initialization syntax -> Could use better error messages

Basically, for class-based user defined functions, naive initialization usage pattern doesn't work. The `udf.with_init_args()` is a bit unintuitive, but it works. The Error messagage however is definitely misleading. Especially for the conventional python class initilaization usage pattern, it can be extremely confusing.

### [Issue 5088](https://github.com/Eventual-Inc/Daft/issues/5088) - Concurrency Param bug for python function based udfs with OpenAI client

For some reason, when we use an openai client inside of a function based batch UDF, we can't add the concurrency parameter. We get this runtime error referring to pickle serialization. I ran into this while I was initially developing the batch udf and it took a second to actually reproduce, but it looks like others have run into it. I also started a [thread on slack](https://dist-data.slack.com/archives/C052CA6Q9N1/p1756400464828409) due to my confusion on whether or not the daft.func supported a concurrency argument.

### [Issue 5090](https://github.com/Eventual-Inc/Daft/issues/5088) Scaling headaches 

For the average user who is looking to leverage daft to perform ai inference using a client (whether it would be openai or otherwise), most users will try either a row_wise UDF or a synchronous Batch UDF. These implementations work at small scale but run into issues once users attempt to run them at 2000 + rows. Regardless of how they arrive at the conclusion, eventually they will attempt to run their inference calls asynchronously which will produce non-blocking errors at the 200-1500 row limit range. 

From what I could tell, this was primarily due to how daft scales its workers. I was running on the native runner, and saw different behaviors as I increased the number of rows. 
- At 100 rows I saw no issues. Both the naive and batch implementations worked well without errors. 
- At 200 rows I began to see a few event loop exceptions and nothing following that line executes. 
- At 1000 rows I see more errors dumped at the end of the workload, again, nothing executes following the error, but the workload still finishes.  
- At 2000 rows, I see about 50 requests complete, but then the job just hangs. I have a video I can share if you are interested. 

Essentially the client will return exceptions it see's, but completes the entire workload for less than 2000 rows, failing after the UDF job is completed. If you have any follow on transformations, that code wont execute and the process exits throwing a long errors chain.

From what I can tell, Daft scales easily, too easily as it turns out. The `asyncio.event_loop` can become starved, most likely from limitations in the openai client, which can create opaque errors. These errors only emerge when you try to scale. The simplest implementations of row-wise and batch user-defined functions both failed to process more than 2000 rows with default concurrency. Once I added the event loop attachment logic in a class based udf, the issues went away and I was able to process the full ~8000 row dataset.

 
## Raw Take

### Decoupling vllm pains from daft pains
Overall I'd say I ran into just as many issues trying to get vllm setup correctly than I did working through the workload itself. The shear number of hidden incompatibilities across models, engine configuration parameters, and openai client args far exceeds the average cognitive load of single day. It takes about 7.5 minutes for vllm to be ready, and installing the library alone takes several minutes. That means that if you misconfigure or need to tweak any of the runtime options, you are looking at least a minimum of a 10 minute redo cycle.

While I initally really wanted to work with gemma-3n-270m, a tiny model operating at the frontier of intelligence cost performance, I eventually went with gemma-3n-e4b-it. It supports text, image, and audio inputs and can run on an L4 if your life depended on it. Naturally if you want decent inference speeds you size up, so I went with an A100 which turned out to do just fine.  

I really wish I didn't spend as much time just getting vllm to run a model I could use for this workload, but it turned out to be highly non-trivial. I'm not even talking about optimization here. The Engine config for vllm has something like 10 billion cli args on it and if you don't read the fine print for the model card on HF, it could take you hours to discover  your model has no default chat-template, or doesn't support images, or isn't trained to handle structured outputs well.

Daft can't control vllm's documentation or model compatiblity, but it can setup examples that are tuned for some of the most popular models. Each time a new frontier model is released, we can work on evaluating it and fine tuning the engine to get the best results. I think spending some time to get Gemma-3 working in a way that daft can fully exploit its strengths would be a worthwhile endeavor, and should inform the roadmap for both offline and online inference strategies. 

Not having Copy/Paste tutorials for implementing structured outputs made it confusing to decouple issues between daft and vllm at times, but in the end I think we arrived at something that looks reasonable. 

I know this might sound like a bit of a distraction, but the reality is every single engineer who looks to use vllm for inference will run into these types of issues. Its one of the reasons I chose this particular image understanding workload in the first place. Every week a new open-source model is released. That means teams are evaluating game-changing performance bumps with various model and inference server configurations about every month or every quarter. 

The opportunity cost of not trying out the latest frontier model is just too high. Not only for accuracy, but compute efficiency as well. In the end, its pretty clear why inference providers are becoming profitable while model developers are not. The main take-away here is that model serving comes with its own set of complexities which should be taken into consideration when prioritizing the roadmap for daft if it's going to become a part of the solution to simplify these pains. 

### Daft Pain points in the developing the multimodal structured outputs workload

#### Preprocessing

Theres a lot to unpack here. I ran into several "gotchas" that I wasn't expecting and had to create issues for. The issues are listed above, so I wont go into too much detail here on what they are so much as how frustrating they were to encounter. 

Setting up the preprocessing for the AI2D dataset was actaully pretty straightforward, and I implemented each of the steps pretty quickly with the help of GPT-5. The text parsing step was definitely going to be the most onerous since I had to implement regex, but the llm did a great job. 

It was super helpful to be able to have a full dataframe preview to grab sample data from. ONe thing that was frustrating was the fact that the cell previews didn't actually increase the amount of content I could see making the feature pretty much useless.

It's always amazing to see just how quickly I could download the parquet file. I know 500 mb isn't much but the fact that the processing time feels trivial just makes me feel like I chose the right tool for the job.

#### Inference UDFs

I approached this workload from the assumption that I would start with the simplest approach I could deriving the bare python OpenAI client call and turning it into a row-wise `daft.func` UDF. This felt trivial to implement and I was up and running quickly on a 100 rows. When I tried to run the full dataset however, the job hangs and I have to restart the notebook runtime (rerunning all of the preprocessing cells again) to debug what went wrong.

Naturally I was a bit confused at first. Especially since  I had no indication or exception as to why my vllm server wasn't processing any requests AND my notebook cells had stalled. I did leave the cell running for over 20 minutes once and saw no progress with the number of rows.

The next logical step for me was to run 200, 500, 1000, and then 2000 rows until I hit the same roadblock. It was at this point that I figured I was running into some sort of a concurrency limitation on the AsyncOpenAI client and decided to test out a few new approaches.

The first thing I tried was to instantiate the client inside the udf. Instantiating an AsyncOpenAI client yielded no calls to vllm, the job hung for about 30-60 seconds before it began processing requests and proceeded to spit out a ton of asyncio event loop errors.

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/daft/udf/row_wise.py", line 90, in __call_async_batch
    event_loop = asyncio.get_running_loop()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: no running event loop
```

Switching to a synchronous OpenAI client worked better. No errors and it completed the entire workload. Naturally, this feels a bit hacky, especially if we weren't using a local vllm server, but it did finish.

Movning from a synchronous mental model to a batch udf was fine for me, but it does take a few extra points of cognitive load to achieve. My very first implementation used a list comprehension with the Sync OpenAI client, but I quickly discovered that I wasn't entirely sure where the client needed to be instantiated from... Another quick note on the transitino between `daft.func` and `daft.udf` that feels obvious at first, but might not be directly obvious to new users is the fact that arguments are now daft.Series. I'm sure most users understand the implications, but when you aren't sure if your inference usage pattern needs the model_id to be vectorized or just a plain string argumnet, you can definitely spend some time in API signature dream land overoptimizing something you arent sure you need yet. 

For this workload I was only using one model_id and one extra_body arg, so I didn't need those parameters to be vectorized or unpacked from daft.series. Again, sounds trivial, but when you are in the moment and you are used to thinking about re-usability, it takes up at least one thread of cognitive load. 

Everytime I write `df.to_pylist()` I *wince* and my eye begins to twitch. We all know that using daft expressions as much as possible yields the best performance, but in this case it's especially true. This UDF is useful but is certainly unperfomant. The serialization back to python alone incurrs inter-generational guilt that I'm just not sure I can atone for.

ANYWAYS, once I had the list comprehension pattern down for zipping the texts and images together... Wait (I thought) ... what about passing lists of images? What about passing entire message histories? How is that supposed to come together anyways? The OpenAI client doesn't just support a single image or a single text content, it can handle a list of them! (This api design problem lives rent free in my head with the water running). I knew was facing emminent destruction of my focus if I played with the idea for too long, so I went with the explicit route.

Moving on... I already knew the general shape of the pattern I needed to implement from colin's async pattern implemented in Sashimi4Talent and llm_generate, so I proceeded to reproduce the same idea with an extra input. Really the only difference between what I ended up writing for this initial function based UDF and the final "production" class-based udf was how I instantiaed the client and event_loop attachment logic that ended up silencing all of the errors we were seeing earlier.

The batch implementation did turn out to be faster, but like I just said, no matter what I did I kept running into those pesky event_loop errors. I knew it had to do with concurreny of the client, so I tried adding a concurrency specifier in the udf decorator for the definition. This was super frustrating since it yielded a different runtime error relating to some sort of pickling issue. I've already included the minimum reproducible example in issue 5088 which was surprising to develop. I wonder if the OpenAI client itself is just this giant complicated thing that shouldn't be included in a UDF at all and we should just be using http requests explicitly. It would probably be simpler.

Regardless, the UDF had to evolve to a class based approach in order for me to run the entire workload in a reasonable amount of time without error. This "production" UDF implementation uses the same event_loop attachment/factory logic that exists in the `llm_generate` function and ended up working for me as well.

The only real differences between my final UDF and llm_generate is: 
1. I initialize openai client in `__init__` and assign it to self.
2. I accept image_col and extra_body explicitly as opposed to using the *generation_config. 
3. I pass model_id in `__call__` instead of at initialization, but still as a string (not series)
4. I have to zip the image and text input together in the wrapper methods for asyncio gather. 

`llm_generate` works with structured outputs on text inputs with no changes needed. It ran the entire dataset (all ~8000 rows), without issues. I actually end up using llm_generate to check image understanding with/without the actual images.  

Hilariously, I was getting the exact same performance between text only and text + image (pass/fail rate of ~55%) until about 30 minutes ago when I looked at the usage pattern and realized that the dataset has the image passed first before the questions. Now that I've run the full workload with images first, the pass/fail rate sits at ~70%. Turns out that we see only a 15% difference in meaningful image understanding with/without images.

The dataset I used to build the workload is actually just one of 50 datasets we could use to evaluate model performance or for training. The next logical step for our workload would be to parallelize the evaluation on ray across multiple subsets.

Arguably, this type of a workload sets the foundation for all sorts of opportunities for case studies in architecture design and could set the stage for Daft to rival Ray's Batch Structured Outputs example.

## Conclusion

My goal in developing this workload was to not only uncover friction points in scaling daft multimodal inference, but to also provide a genuinely useful piece of code. AI engineers are constantly evaluating new models, and rarely do you see examples that are this end to end.

I'd like to extend the workload to compare performance across model variants, or explore how different sampling parameters impact performance. There seems to be a new frontier open source model released every week nowadays, and I only expect that trend to continue.

I think there are a lot of pain points that I have uncovered that are easily adderessed with a new llm_generate api signature. Throughout this exercise I aimed to focus on the problems instead of trying to provide an all-encompassing solution. I hope you found this Friction log helpful and maybe learned a thing or two a long the way.