# TODO
[x] ChatML format (actually need to add special tokens)
[x] Vicuna dataset merge (yahma/alpaca-cleaned)
[x] Phi-2 fine tuning
[x] Quantize /w llama.cpp
[x] Make custom component use llama.cpp + ChatML
[ ] Continued synthetic dataset improvements (there are a bunch of TODOs in there)
[x] Licenses + Attributions
[x] Finish Readme/docs for initial release
[x] Function calling as JSON
[ ] multi-turn prompts; better instruct dataset like dolphin/wizardlm?
[ ] Fine tune Phi-1 and Phi-1.5 versions
[ ] "context requests"
    - basically just let the model decide what RAG/extra context it wants
    - the model predicts special tokens as the first few tokens of its output
    - the requested content is added to the context after the request tokens and then generation continues
    - needs more complicated training b/c multi-turn + there will be some weird masking going on for training the responses properly
[ ] RAG for getting info for setting up new devices
    - set up vectordb
    - ingest home assistant docs
    - "context request" from above to initiate a RAG search
[x] make llama-cpp-python wheels for "llama-cpp-python>=0.2.24"
[ ] prime kv cache with current "state" so that requests are faster
[ ] make a proper evaluation framework to run. not just loss. should test accuracy on the function calling
[ ] add LocalAI backend
[ ] more config options for prompt template (allow other than chatml)