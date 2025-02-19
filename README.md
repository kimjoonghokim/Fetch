# Fetch -- efficient tree search for LLM reasoning

Code for paper "Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls"

## Setup

Follow the steps below to run our scripts:

### Step 1. Setup service of policy, verifier, and embedding model

#### Policy

For policy, we employ vllm. First, start the policy service by running the following command:

```
python3 -m vllm.entrypoints.openai.api_server --model /path/to/policy/model --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096
```

Replace `/path/to/policy/model` with the actual path to your policy model.

#### Verifier

For verifier, first update your model path in `verifier/server.py`. Then run the script `bash run.sh ./ 0` under `verifier`.

#### Embedding Model

If state merging is used, please update the path in `cluster/server_cluster.py`, then run the script `bash run_app.sh ./ 0` under `cluster`.

### Step 2. Run tree search algorithms

We provide bfs, beamsearch, and mcts. Update the scripts (e.g., input and output file path) then just run (e.g., `python3 beamsearch.py`).

## Tips

- If you want to use your own policy or verifier, please notice the prompt fed to the models and the separator (e.g., "\n") to split steps.
- The embedding model can be prepared for better performance. We have provided the scripts to train these models. Using pretrained SimCSE or other embedding models also work. You can find many available models on huggingface hub.
- If you do not want to prepare these models, our trained models are available on huggingface hub:
  - Policy: xmu-nlp/Llama-3-8b-gsm8k
  - Verifier: xmu-nlp/Llama-3-8b-gsm8k-value-A and xmu-nlp/Llama-3-8b-gsm8k-value-B
  - Emb: xmu-nlp/simcse-large-gsm8k

## Citation

If you find our work useful, please cite our paper:
```
@misc{wang2025dontlosttreesstreamlining,
      title={Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls}, 
      author={Ante Wang and Linfeng Song and Ye Tian and Dian Yu and Haitao Mi and Xiangyu Duan and Zhaopeng Tu and Jinsong Su and Dong Yu},
      year={2025},
      eprint={2502.11183},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11183}, 
}
```