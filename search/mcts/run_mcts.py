import os
import sys
from gsm_config import *
from mcts_tree import *
import multiprocessing
import json
import pickle
import numpy as np
import math
import copy
from dotenv import load_dotenv
from tqdm import tqdm
import time


CONTINUE = False

if __name__ == '__main__':
    
    if CONTINUE:    
        output_fpath = "path/to/prev/unfinish/file"
        problems = pickle.load(open(output_fpath, "rb"))
    else:
        config = GSMConfig()
        
        load_dotenv(dotenv_path='../experiments_config.env')
        data_fpath_var = os.getenv("PATH_TO_DATASET")
        data_fpath = os.getenv(data_fpath_var) if data_fpath_var else None # path to the test set
        if data_fpath:
            dataset_type = os.path.basename(data_fpath).split('.')[0]
            dataset_name = os.path.basename(os.path.dirname(data_fpath))
        else:
            dataset_name = "unknown"
            dataset_type = "unknown"
        
        output_fpath = f"{dataset_type}_{dataset_name}_mcts_rb{config.root_budget}_nb{config.node_budget}_c{config.c}_r{config.n_rollouts}_md{config.max_depth}_mt{config.min_terminals}_d{config.d}.pkl"

        dataset = []
        with open(data_fpath, "r") as f:
            for line in f.readlines():
                dataset.append(json.loads(line))

        problems = []
        for instance in dataset:
            question = instance["question"]
            answer = instance["answer"]
            problem = MCTSTree(question, answer, config)
            problems.append(problem)

    def run_exp(problem):
        problem.run_mcts()
        return problem

    start_time = time.time()

    pool = multiprocessing.Pool(100)
    problems = list(tqdm(pool.imap_unordered(run_exp, problems), total=len(problems)))    
    pool.close()

    total_runtime = time.time() - start_time

    total_prompt_tokens = sum([p.prompt_tokens for p in problems])
    total_completion_tokens = sum([p.completion_tokens for p in problems])
    total_tokens = sum([p.total_tokens for p in problems])

    final_data = {
        'problems': problems,
        'metrics': {
            'total_runtime': total_runtime,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens
        }
    }
    
    with open(output_fpath, "wb") as f:
        pickle.dump(final_data, f)

    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"  (Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens})")
    print(f"Results saved to {output_fpath}")

