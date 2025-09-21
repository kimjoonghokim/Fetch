# this script is written in the early stage of this project. At that time, we use a slightly low-efficient implementation of multi-threading. Besides, we do not use the embedding model as a server.

import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import sys
from call_joint_service import call
from search_tree_merge import *
import json
import pickle
import numpy as np
import jsonlines
from tqdm import tqdm
import time
from dotenv import load_dotenv

LIMIT=50
BUDGET=10
TEMPERATURE=0.8

# task dependent
def assert_end(text):
    return True if text.strip().split("\n")[-1].startswith("The answer is") else False

def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = 0
    if state.content is not None and (len(state.content) == 0 or len(state.content) > 1024): # too short or too long
        state.value = 0


SEQ_STOP_TOKENS = []
STEP_STOP_TOKENS = ["\n"]

CONTINUE = False
load_dotenv(dotenv_path='../experiments_config.env')
data_fpath_var = os.getenv("PATH_TO_DATASET")
data_fpath = os.getenv(data_fpath_var) if data_fpath_var else None # path to the test set
if data_fpath:
    dataset_type = os.path.basename(data_fpath).split('.')[0]
    dataset_name = os.path.basename(os.path.dirname(data_fpath))
else:
    dataset_name = "unknown"
    dataset_type = "unknown"
output_fpath = f"{dataset_type}_{dataset_name}_bfs_merge_b{BUDGET}_t{TEMPERATURE}.pkl"
load_dotenv(dotenv_path='../../server_config.env')
policy_fpath = os.getenv("POLICY_MODEL_PATH") # path to the policy model

if __name__ == '__main__':
    
    if CONTINUE:    
        problems = pickle.load(open(output_fpath, "rb"))
        start = max([problem.return_timestep() for problem in problems])
    else:
        dataset = []
        with open(data_fpath, "r") as f:
            for line in f.readlines():
                dataset.append(json.loads(line))

        problems = []
        for instance in dataset:
            question = instance["question"]
            answer = instance["answer"]
            problem = Tree(question, answer, additional_info={"strategy": "max", "d": 0.15})
            problem.init_root_node(0)
            problems.append(problem)
        start = 0

    start_time = time.time()

    for i in range(start, LIMIT):
        clusters = []
        questions = []
        anchors = []
        paths = []
        finished = 0
        for problem in problems:
            cluster, states = problem.select_best_cluster()
            if states is not None:
                clusters += [cluster] * BUDGET
                anchors += (states * BUDGET)[:BUDGET]
                questions += [problem.question] * BUDGET
                paths += ([state.print_path() for state in states] * BUDGET)[:BUDGET]
            else:
                finished += 1

        print(f"iteration {i}")
        print(f"finished {finished} / {len(problems)}")

        if len(questions) == 0:
            break

        next_steps, next_values, usages, verifier_usages = call(questions, paths, [TEMPERATURE] * len(questions), [STEP_STOP_TOKENS] * len(questions))
        
        # Aggregate token usage
        for anchor_state, usage, verifier_usage in zip(anchors, usages, verifier_usages):
            tree = anchor_state.tree
            tree.prompt_tokens += usage.get("prompt_tokens", 0)
            tree.completion_tokens += usage.get("completion_tokens", 0)
            tree.total_tokens += usage.get("total_tokens", 0)
            tree.verifier_prompt_tokens += verifier_usage.get("prompt_tokens", 0)
            tree.verifier_completion_tokens += verifier_usage.get("completion_tokens", 0)
            tree.verifier_total_tokens += verifier_usage.get("total_tokens", 0)
            # Note: embedding tokens are tracked in merge_nodes() method
        
        for cluster, state, next_step, next_value in zip(clusters, anchors, next_steps, next_values):
            child = state.tree.add_node(next_step, next_value, state, i + 1, assert_end(next_step))
            cluster.cache.append(child)
            fix_value(child)

        # merge similar states
        for cluster in tqdm(clusters, desc="merging"):
            cluster.merge_nodes()
            
# if unfinish, select 1 node and extend to the end
questions = []
anchors = []
paths = []
finished = 0
for problem in problems:
    state = problem.select_best_node()
    if state is not None:
        anchors += [state]
        questions += [problem.question]
        paths += [state.print_path()]
    else:
        finished += 1

if len(questions) != 0:
    print(f"iteration final")
    print(f"finished {finished} / {len(problems)}")

    next_steps, next_values, usages, verifier_usages = call(questions, paths, [0] * len(questions), [SEQ_STOP_TOKENS] * len(questions))
    
    # Aggregate token usage
    for anchor_state, usage, verifier_usage in zip(anchors, usages, verifier_usages):
        tree = anchor_state.tree
        tree.prompt_tokens += usage.get("prompt_tokens", 0)
        tree.completion_tokens += usage.get("completion_tokens", 0)
        tree.total_tokens += usage.get("total_tokens", 0)
        tree.verifier_prompt_tokens += verifier_usage.get("prompt_tokens", 0)
        tree.verifier_completion_tokens += verifier_usage.get("completion_tokens", 0)
        tree.verifier_total_tokens += verifier_usage.get("total_tokens", 0)
        # Note: embedding tokens are tracked in merge_nodes() method
    
    for state, next_step, next_value in zip(anchors, next_steps, next_values):
        child = state.tree.add_node(next_step, next_value, state, LIMIT + 1, assert_end(next_step))
        fix_value(child)

# Calculate runtime and token totals
total_runtime = time.time() - start_time
total_prompt_tokens = sum([p.prompt_tokens for p in problems])
total_completion_tokens = sum([p.completion_tokens for p in problems])
total_tokens = sum([p.total_tokens for p in problems])
total_verifier_prompt_tokens = sum([p.verifier_prompt_tokens for p in problems])
total_verifier_completion_tokens = sum([p.verifier_completion_tokens for p in problems])
total_verifier_tokens = sum([p.verifier_total_tokens for p in problems])
total_embedding_prompt_tokens = sum([p.embedding_prompt_tokens for p in problems])
total_embedding_completion_tokens = sum([p.embedding_completion_tokens for p in problems])
total_embedding_tokens = sum([p.embedding_total_tokens for p in problems])
total_all_tokens = total_tokens + total_verifier_tokens + total_embedding_tokens

final_data = {
    'problems': problems,
    'metrics': {
        'total_runtime': total_runtime,
        'policy_server': {
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens
        },
        'verifier_server': {
            'total_prompt_tokens': total_verifier_prompt_tokens,
            'total_completion_tokens': total_verifier_completion_tokens,
            'total_tokens': total_verifier_tokens
        },
        'embedding_server': {
            'total_prompt_tokens': total_embedding_prompt_tokens,
            'total_completion_tokens': total_embedding_completion_tokens,
            'total_tokens': total_embedding_tokens
        },
        'combined': {
            'total_tokens': total_all_tokens,
            'verifier_percentage': (total_verifier_tokens / total_all_tokens * 100) if total_all_tokens > 0 else 0,
            'embedding_percentage': (total_embedding_tokens / total_all_tokens * 100) if total_all_tokens > 0 else 0
        }
    }
}

with open(output_fpath, "wb") as f:
    pickle.dump(final_data, f)

print("\n=== BFS Merge Complete ===")
print(f"Total runtime: {total_runtime:.2f} seconds")
print(f"\nPolicy Server Tokens:")
print(f"  Total: {total_tokens}")
print(f"  Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}")
print(f"\nVerifier Server Tokens:")
print(f"  Total: {total_verifier_tokens}")
print(f"  Prompt: {total_verifier_prompt_tokens}, Completion: {total_verifier_completion_tokens}")
print(f"\nEmbedding Server Tokens:")
print(f"  Total: {total_embedding_tokens}")
print(f"  Prompt: {total_embedding_prompt_tokens}, Completion: {total_embedding_completion_tokens}")
print(f"\nCombined Total: {total_all_tokens}")
print(f"Verifier contribution: {total_verifier_tokens / total_all_tokens * 100:.1f}%")
print(f"Embedding contribution: {total_embedding_tokens / total_all_tokens * 100:.1f}%")
print(f"Results saved to {output_fpath}")

