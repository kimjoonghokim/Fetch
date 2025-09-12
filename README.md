# New README (under construction)

## 🚀 Setup (NEW):

These instructions are written assuming you are cloning this repo into a 4 GPU Runpod

### 📌 Step 1. Install packages and setup service of policy, verifier, and embedding model
After cloning the repo into your Runpod, simply run the following command to run the setup script.
```
bash setup.sh
```
The setup script installs the required packages as well as launches the policy, verifier, and embedding servers on their respective GPUs. The model paths for these servers and which GPU it is running on are defined in the `server_config.env` file. If you would like to use a different model, please change the model path in `server_config.env` before running the above setup script.

### 📌 Step 2. Run tree search algorithms
Within the `search` directory there are seperate directories for each tree search algorithm (`beamsearch`, `bfs`, `mcts`). Navigate to the directory of the search algorithm you would like to run and simply run the corresponding Python file.  
  
  For example for beamsearch:
```
python beamsearch.py
```
  For bfs:
```
python bfs.py
```
  For mcts:
```
python run_mcts.py
```

By default, the algorithms are run on a toy dataset of 16 questions, but you can change/specify the path in the `experiments_config.env` file within this `search` directory. Simply change the `PATH_TO_DATASET` variable to reflect the dataset you would like to use. We have also included various other datasets to use in the `dataset` directory


# (OLD README) Fetch — Efficient Tree Search for LLM Reasoning

Code for the paper [Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls](https://arxiv.org/abs/2502.11183)

---
## (OLD)🚀 Setup

Follow the steps below to run our scripts:

### 📌 Step 1. Setup service of policy, verifier, and embedding model

#### 📚 Policy

We employ [vllm](https://docs.vllm.ai/en/latest/) for the policy. To start the policy service, run the following command:
```
python3 -m vllm.entrypoints.openai.api_server --model xmu-nlp/Llama-3-8b-gsm8k --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096
```

#### 🔍 Verifier

1. Update your model path in `verifier/server.py`.
2. Run the script: `bash run.sh ./ 0` inside the `verifier` directory. Replace `0` with the GPU number you want to use

#### 📦 Embedding Model

If you're using state merging, follow these steps:
1. Update the path in `cluster/server_cluster.py`.
2. Run the script: `bash run_app.sh ./ 0` inside the `cluster` directory. Replace `0` with the GPU number you want to use

---

### 📌 Step 2. Run tree search algorithms

We provide three tree search algorithms: **BFS (Best-First Search)**, **Beam Search**, and **MCTS (Monte Carlo Tree Search)**.

1. Specify the input, output file paths, and other parameters in scripts such as `beamsearch.py`.

2. Simply execute the corresponding Python script. For instance, to run Beam Search: `python3 beamsearch.py`

---

## 🎯 Tips

- **Using Your Own Models**: If you prefer to use your own policy or verifier, please pay attention to the prompts fed to the models and the separator (e.g., "\n") to split steps.
- **Embedding Model Preparation**: Preparing an embedding model can improve performance. We provide scripts to train these models. You can also use pretrained SimCSE or other embedding models available on [Hugging Face Hub](https://huggingface.co/models).
- **Pre-trained Models**: If you don't want to train models, our trained models are available on HuggingFace Hub:
  - Policy: [`xmu-nlp/Llama-3-8b-gsm8k`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k)
  - Verifier: [`xmu-nlp/Llama-3-8b-gsm8k-value-A`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k-value-A) and [`xmu-nlp/Llama-3-8b-gsm8k-value-B`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k-value-B)
  - Emb: [`xmu-nlp/simcse-large-gsm8k`](https://huggingface.co/xmu-nlp/simcse-large-gsm8k)

---

## 📝 Citation

If you find our work useful, please cite our paper:

```
@misc{wang2025dontlosttreesstreamlining,
      title={Don't GetLost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls}, 
      author={Ante Wang and Linfeng Song and Ye Tian and Dian Yu and Haitao Mi and Xiangyu Duan and Zhaopeng Tu and Jinsong Su and Dong Yu},
      year={2025},
      eprint={2502.11183},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11183}, 
}
```
