### Performance of running the model on a Raspberry Pi
The RPI4 4GB that I have was sitting right at 1.5 tokens/sec for prompt eval and 1.6 tokens/sec for token generation when running the `Q4_K_M` quant. I was reliably getting responses in 30-60 seconds after the initial prompt processing which took almost 5 minutes. It depends significantly on the number of devices that have been exposed as well as how many states have changed since the last invocation because llama.cpp caches KV values for identical prompt prefixes.

It is highly recommend to set up text-generation-webui on a separate machine that can take advantage of a GPU.

# Home 1B V2 GGUF Q4_K_M RPI5

christmas.txt
llama_print_timings:        load time =     678.37 ms
llama_print_timings:      sample time =      16.38 ms /    45 runs   (    0.36 ms per token,  2747.09 tokens per second)
llama_print_timings: prompt eval time =   31356.56 ms /   487 tokens (   64.39 ms per token,    15.53 tokens per second)
llama_print_timings:        eval time =    4868.37 ms /    44 runs   (  110.64 ms per token,     9.04 tokens per second)
llama_print_timings:       total time =   36265.33 ms /   531 tokens

climate.txt
llama_print_timings:        load time =     613.87 ms
llama_print_timings:      sample time =      20.62 ms /    55 runs   (    0.37 ms per token,  2667.96 tokens per second)
llama_print_timings: prompt eval time =   27324.34 ms /   431 tokens (   63.40 ms per token,    15.77 tokens per second)
llama_print_timings:        eval time =    5780.72 ms /    54 runs   (  107.05 ms per token,     9.34 tokens per second)
llama_print_timings:       total time =   33152.48 ms /   485 tokens

# Home 3B V2 GGUF Q4_K_M RPI5

climate.txt
llama_print_timings:        load time =    1179.64 ms
llama_print_timings:      sample time =      19.25 ms /    52 runs   (    0.37 ms per token,  2702.00 tokens per second)
llama_print_timings: prompt eval time =   52688.82 ms /   431 tokens (  122.25 ms per token,     8.18 tokens per second)
llama_print_timings:        eval time =   10206.12 ms /    51 runs   (  200.12 ms per token,     5.00 tokens per second)
llama_print_timings:       total time =   62942.85 ms /   482 tokens

sonnet.txt
llama_print_timings:        load time =    1076.44 ms
llama_print_timings:      sample time =    1225.34 ms /   236 runs   (    5.19 ms per token,   192.60 tokens per second)
llama_print_timings: prompt eval time =   60754.40 ms /   490 tokens (  123.99 ms per token,     8.07 tokens per second)
llama_print_timings:        eval time =   44885.82 ms /   213 runs   (  210.73 ms per token,     4.75 tokens per second)
llama_print_timings:       total time =  107127.16 ms /   703 tokens