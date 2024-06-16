## Qwen/Qwen2-0.5B-Instruct

# tinyhome-qwen-rev1
- full fine tune
- epochs: 1
- 2048 train ctx
- batch size 32
- learning rate 2e-5
- weight decay 0.1
- gradient clipping 1.0
- dataset size: small
+ evaluation results: NEEDS RE-TEST b/c OF BAD EVAL SCRIPT

# tinyhome-qwen-rev2
- full fine tune
- epochs: 1
- 2048 train ctx
- batch size 32
- learning rate 2e-5
- weight decay 0.1
- gradient clipping 1.0
- dataset size: medium
+ evaluation results: NEEDS RE-TEST b/c OF BAD EVAL SCRIPT

# tinyhome-qwen-rev3
- full fine tune
- epochs: 1
- 2048 train ctx
- batch size 64
- learning rate 2e-5
- weight decay 0.1
- gradient clipping 1.0
- dataset size: small 4 language mix
+ evaluation results:
  - english: 0.9842022116903634
  - german: 0.8992834394904459
  - french: 0.9307445956765412
  - spanish: 0.9406099518459069