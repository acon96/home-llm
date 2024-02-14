# early home-llm experiements (phi1.5)
### rev1 - original test
- 1 epoch
- train ctx 1900
- I think the learning rate was way too high (2e-4)
- the service invocation syntax is ambiguous causing repeat code blocks
- it doesn't get the device name right like ever
- eval dataset was disabled

### rev2 - it kinda works
- eval dataset at 10%
- 2 epochs
- train ctx 1200
- batch size 1
- learning rate linear 5e-5
- gradient accumulation steps 1
+ fixed invocation syntax
+ still repeatedly spits out code blocks but at least it closes them correctly
+ names are MUCH more accurate. will still hallucinate names that don't exist

### rev3 - ok it definitely works
- 4 epochs
- batch size 2
- learning rate cosine 5e-5
+ doesn't seem like there's much difference but the loss is lower
+ still halluncinates device names. (need to figure this one out)
+ need more examples for: garage_door, media_player, 

### rev4 - got to way lower loss. it tries really hard to stop generating text
- 4 epochs
- train ctx 512
- batch size 2
- learning rate cosine 1e-4
- added system prompt and moved services block before states block

### rev 4.1 - really doesn't work as well. loss dropped REALLY fast and then never got as low as rev4
- 4 epochs
- train ctx 512
- batch size 3
- learning rate cosine 1e-4
- proper pad token

### rev 4.2 - yeah nah it's the pad token
- batch size 2

### rev 5 - new dataset
- 3 epochs (4th epoch was overfit)
- train cx 512
- batch size 2
- learning rate cosine 1e-5
+ actually stops generating text. not at the right... place but still!
+ messing with temperature makes it generate some interesting output.

### rev 5.1 - gradient accumulation test
- 3 epochs
- train cx 512
- batch size 8
- learning rate cosine 1e-5
+ very meh

### rev 5.2 - learning rate test
- 3 epochs
- train cx 512
- batch size 8
- learning rate cosine 1e-4
+ higher learning rate really helped with the higher batch size
+ is able to more reliably generate the correct device name again
+ still need more examples for multi-device actions (really need room/group support in dataset)
+ need to have more variance in request format. need more informal + more formal versions

### rev 5.3 - learning rate test 2
- 4 epochs
- train cx 512
- batch size 8
- learning rate cosine 6e-5
+ lower learning rate seemed to not be as effective even though it ran for longer

### rev 6 - dataset revamp again
- 3 epochs
- train cx 512
- batch size 8
- learning rate cosine 1e-4
- all questions + responses are lowercase now
- ensured there are no duplicate entries in the states block
+ definitely a bit overfit
+ maybe not so overfit. able to 0 shot asking to do stuff to a christmas tree

### rev 6.1 - lower train rate
- 3 epochs
- train cx 512
- batch size 8
- learning rate cosine 6e-5
+ also definitely a bit overfit. can't generate names it hasn't seen before

### rev 6.2 - fewer epochs
- 2 epochs
- train cx 512
- batch size 8
- learning rate cosine 6e-5

### rev 6.3 - higher batch
- 2 epochs
- train cx 512
- batch size 12
- learning rate cosine 1e-4

### rev 7 - tweak dataset again
- 2 epochs
- train ctx 512
- batch size 8
- learning rate 1e-4
+ when generating results, don't end with a space. it works WAY better

### rev 7.1 - failed padding attempt

### rev 7.2 - try to overfit less + no newline at end
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-4
+ it definitly works with only one epoch

### rev 7.3 - try adding fake end of sentence token
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-4

### rev 8 - dataset tweaks. add status requests
+ service requests still mostly work but status requests are pretty broken

### rev 8.1 - tweak example counts + ratios
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-4
+ seems to have worked better with lower example counts

### rev 8.2 - try to fit learning rate so loss doesn't bottom out till the end of training
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 8e-5 (didn't change loss at all)
- learning rate 5e-5 (same)
- learning rate 1e-5 (wayyyy better)
+ pretty sure i've been overcranking most of these and destroying most of the model
+ oh yuuhhhhh it's overcranked. nails both request types (plus even ending generation)
+ needs ambiguous device name examples because I totally just asked it an ambiguous question and it answered the one I wasn't expecting

### rev 8.3 - further reduced training rate
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 8e-6
+ certainly not overfit like < rev7
+ has some creativity with how it repsonds
+ will often get the device name wrong on the first try

### rev 8.4 - re-ordered prompt
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 8e-6
- put actions before response and also made actions it's own "block"
+ it *works* but is incredibly open ended
+ basically never stops generating text

### rev 8.5 - tweaked prompt format again
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 8e-6
- re-orderd response before actions again but made actions less like a "block" so it might stop generation
+ that worked rather badly

### rev 8.6 - make prompt look more like other examples it has seen before
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 8e-6
- change ```done to just ``` and add 3 newlines at the end (idk it keeps doing that for other prompts before stopping)
+ it wants to generate the other prompt types much more with this config
+ only get the correct response about 50% of the time
+ it totally stops correctly when it DOES work

### rev 8.7 - try to fit a bit more. the last iteration jumps around on which format it chooses
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-5
+ similar issues as last model
+ altering the format (with newlines) makes it pick our format more often
+ comparing to 8.6 with modified format shows this one is better at getting device names right

### rev 8.8 - train with newlines instead of spaces in requets/response
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-5
+ definitely worse than the previous one
+ for some reason both 8.7 and 8.8 are horrible when using their actual template but if you deviate slightly it works a lot better on inference

### rev 8.9 - actually fix pad token
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-5
+ properly generates a response (+ terminates) when using the actual template

### rev 9 - reduced dataset size
- 1 epoch
- train ctx 512
- batch size 8
- learning rate 1e-5
+ didn't work as well
+ would often not generate a service call
+ went back to 8.9

------

# Home 1B
## home-1b-rev1
- 1 epoch
- 2048 train ctx
- batch size 8
- learning rate 1e-5
- weight decay 0.1
- gradient clipping 1.0
- dataset changes:
  - updated chatml format
  - json function calling
  - included the alpaca split
+ it works OK with low temperatures
+ seems to handle the alpaca dataset not so well

### Home-1b-v1-GGUF
- eval results: 0.767816091954023

## home-1b-rev5/6 parameters
- 1 epoch
- 2048 train ctx
- batch size 8
- learning rate 1e-5
- weight decay 0.1
- gradient clipping 1.0
- save model every 200 or 400 steps

### home-1b-rev5
- dataset size: medium
- evaluation results:
  - 200: 0.553448275862069
  - 400: 0.7482758620689656 (+.19)
  - 600: 0.8103448275862069 (+.06)
  - 800: 0.8316091954022988 (+.02)
  - 1000:  0.8396551724137931 (+.008)
  - 1200: 0.8488505747126437 (+.009)
  - Final (1467): 0.8494252873563218 (+.00005)

### home-1b-rev5_1
- dataset size: small
- evaluation results:
  - 200: 0.6057471264367816
  - 400: 0.7494252873563219 (+.143)
  - 600: 0.7683908045977011 (+.018)
  - 800: 0.7729885057471264 (+.0046)
  - Final (869): bad

### home-1b-rev5_2
- dataset size: large
- evaluation results:
  - 200: --
  - 400: --
  - 600: 0.8425287356321839
  - 800: 0.8666666666666667
  - 1000: 0.8770114942528736
  - 1200: 0.8844827586206897
  - 1400: 0.8879310344827587
  - 1600: 0.8844827586206897
  - Final (1848): 0.8833333333333333

### home-1b-rev6
- dataset size: large (fixed templates + function calling arguments; brightness is broken)
- evaluation results: 0.8254149971379507

### home-1b-rev6_1
- dataset size: xl (fixed templates + function calling arguments; 0-255 brightness is broken)
- evaluation results: 
  - 400: 0.7240984544934173
  - 800: 0.8311390955924441
  - 1200: 0.8471665712650257
  - 1600: 0.8597595878649112
  - 2000: 0.8551803091013166
  - Final (2322): 0.8586147681740126

### home-1b-rev6_2 = Home-1B-v2-GGUF
- dataset size: large (change brightness back to percentages; increase color references by ~2x)
- evaluation results: 
  - 400: 0.7856064418721691
  - 800: 0.864116759
  - 1200: 0.882234524
  - 1600: 0.885254152
  - 2000: 0.8852541519879215
  - Final (2048): 

# Home 3B
- 1 epoch
- 2048 train ctx
- batch size 8
- learning rate 1e-5
- weight decay 0.1
- gradient clipping 1.0
- save model every 200 or 400 steps

Missing a lot of earlier 3B training results (not sure where they are)

### Home-3b-v2-GGUF (broken training run)
- evaluation result: 0.6908045977011494

### home-3b-v3-rev1
- dataset size: large
- evaluation results: 0.9091954022988505

### home-3b-v3-rev2 = Home-3B-v2-GGUF (republished)
- dataset size: xl + alpaca
- evaluation results: 0.8731756416708606

### Home-3B-v2-GGUF:ha_only
- dataset size: large
- evaluation results: FAILED (again.....)


## Potential Other Datasets to Use

### SFT
Alpaca: https://huggingface.co/datasets/yahma/alpaca-cleaned
Alpaca (Translated): https://huggingface.co/datasets/saillab/taco-datasets
WizardLM 200k: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k
WizardLM 70k: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k
Huggingface Ultrachat 200k: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
OpenOrca Slim Deduped (363k): https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup

### DPO
Intel Orca DPO Pairs: https://huggingface.co/datasets/Intel/orca_dpo_pairs
Huggingface Ultrachat: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
