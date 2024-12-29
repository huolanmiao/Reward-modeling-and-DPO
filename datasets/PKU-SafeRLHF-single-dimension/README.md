---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- safe
- safety
- ai-safety
- llm
- lm
- human-feedback
- rlhf
- safe-rlhf
size_categories:
- 100K<n<1M
---

# Dataset Card for PKU-SafeRLHF-single-dimension

<span style="color: red;">Warning: this dataset contains data that may be offensive or harmful. The data are intended for research purposes, especially research that can make models less harmful. The views expressed in the data do not reflect the views of PKU-Alignment Team or any of its members. </span>

## Dataset Summary
By annotating Q-A-B pairs in [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) with single dimension, this dataset provide 81.1K high quality preference dataset. Specifically, each entry in this dataset includes two responses to a question, along with safety meta-labels and preferences for both responses.

In this work, we performed SFT on Llama2-7B and Llama3-8B with Alpaca 52K dataset, resulting in Alpaca2-7B and Alpaca3-8B. This dataset contains responses from Alpaca-7B, Alpaca2-7B, and Alpaca3-8B in the corresponding folders under /data.

The data collection pipeline for this dataset is depicted in the following image:

![Data Collection Pipeline](data-collection-pipeline.png)

## Usage

To load our dataset, use the `load_dataset()` function as follows:

```python
from datasets import load_dataset

dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-single-dimension")
```

To load a specified subset of our dataset, add the `data_dir` parameter. For example:

```python
from datasets import load_dataset

dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-single-dimension", data_dir='data/Alpaca-7B')
```
