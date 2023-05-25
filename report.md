# 深度學習 作業一
A1085125 洪祐鈞 2023/02/24

## Disclaimer

- The code in this homework is amalgamation of LLM prompt (ChatGPT and discord Clyde), GitHub Copilot autocompletion, and random code from the forum. I would provide the prompt of the LLM to llm_prompt.md, also keep the comment for the code generated from copilot.

## Environments

- O.S: ManjaroLinux 22.1.2 Talos
- Miniconda 23.1.1
- CPU: intel i7-1260p
- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Python 3.11.3
- For python module version, please refer to requirements.txt

## Quick Start

Referring requirements.txt to set up your environment, preferably use a conda first so it would not messed up your local environment.

Running: ```python ex_1.py``` for first experiment, there are 5 experiments for respective experiments.
After running any experiments, 

## General Strategy of the homework

- The computational resources is limited, instead of choosing a bulky model which has highest accuracy. What I care about is the efficiency of the model, which means I can get high accuracy in short amount of time. This way, I can have more feedback 
- Running on local computer, which I can modularize my code and having better developter experience. ~~Also, copilot.~~
- Keeping training/testing log, and visualize the result. For better knowing what exactly happened during training and testing.

## Mistake I've made

- Validation and train Set contains duplicate image, which cause the validation accuracy goes insanely high. Making it cannot present the true metrics
- Not reading the documentation or overly rely on the prompt that AI gives me.
- Using WSL as my first development, which turns out running on linux natively would be signigicantly faster. It seems like even on WSL, it still use CUDA that embedds in windows. And it makes anything related to data transform significantly slower.
- Procrastination. I could've try more thing, or doing more ablation study.

## Things I can improve or try

- Using Kaggle and utilizing the distributed training for getting faster result.

## Final Thoughts

