# Implementing Structued Outputs with Daft 

A Friction Log by Everett Kleven 

This Friction Log Covers the many pain points and discoveries I made along the way while getting structured ouputs over images to work on vllm with gemma-3n-e4b-it. There is actaully quite a bit to unpack here. Overall I'd say I ran into way more issues just trying to get vllm setup correctly than building the workload itself. 

The workload of focus for the week was structured outputs. Structured outputs form the foundation of many of the most important use cases that AI powers today. Synthetic Data, Document OCR, and Tool Use all are enabled with logit masking, a process that few AI engineers actually have experience with directly.  

Not having Copy/Paste tutorials for implementing structured outputs made it confusing to decouple issues between daft and vllm at times, but in the end I think we arrived at something that looks reasonable. 

There are a lot of gotchas when trying to run ai workloads, and each time you add a new variable, cognitive load dwindles.

 It takes about 7.5 minutes for vllm to be ready, and installing the library alone takes several minutes. That means that if you misconfigure or need to tweak any of the runtime options, you are looking at least a minimum of a 10 minute redo cycle. This idle time hurts.. excuse me... It fucking sucks. Like no wonder inference providers have a moat! 

I really wish I didn't spend as much time just getting vllm to run a model I could use for this workload, but it turned out to be highly non-trivial. I'm not even talking about optimization here. The Engine config for vllm has something like 10 billion cli args on it and if you don't read the fine print for the model card on HF, it could take you hours to discover  your model has no default chat-template, or doesn't support images, or isn't trained to handle structured outputs well.

While I initally really wanted to work with gemma-3n-270m, a tiny model operating at the frontier of intelligence cost performance, I eventually went with gemma-3n-e4b-it. It supports text, image, and audio inputs and can run on an L4 if your life depended on it. Naturally if you want decent inference speeds you size up, so I went with an A100 which turned out to do just fine.  

Friction points: 
- Class-based UDF initialization syntax -> Could use better error messages [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)
- No Base64 encoding for image dtype [Issue 5089](https://github.com/Eventual-Inc/Daft/issues/5089)
- Concurrency Param bug for python function based udfs with OpenAI client [Issue 5088](https://github.com/Eventual-Inc/Daft/issues/5088)
- Event loop Exceptions when scaling inference without a set event loop. 
- Hanging Infernce calls when running 2000+ rows

