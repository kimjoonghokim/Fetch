import os
import json
import pickle
import numpy as np
import jsonlines
import requests
from tqdm import tqdm
import time

LIMIT=50
BUDGET=5
BEAM=5
TEMPERATURE=0.8
data_fpath = "../../dataset/toy.jsonl" # path to the test set
output_fpath = f"test_gsm8k_beamsearch_b{BUDGET}_t{TEMPERATURE}.pkl" # path to the output file
policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k" # path to the policy model

# task dependent
def assert_end(text):
    return True if "The answer is" in text and text.endswith(tokenizer.eos_token) else False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256
def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = -1
    if state.content is not None and (len(state.content) == 0 or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP): # too short or too long
        state.value = -1
    if state.content.endswith(tokenizer.eos_token) and not assert_end(state.content):
        state.value = -1
    return state

def call_policy(question, path):
    url = "http://127.0.0.1:8000/v1/completions"
    model = policy_fpath
    query = f"Question: {question}\nAnswer:{path}"
    pload ={"prompt": query, "model": model, "temperature": TEMPERATURE, "max_tokens": 512, 
            "stop": ["\n"], "include_stop_str_in_output": True, "skip_special_tokens": False}
    response =requests.post(url, json=pload)
    response_json = response.json()
    choice = response_json["choices"][0]
    usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return choice["text"], usage

def call_value(question, path):
    url = "http://127.0.0.1:8002/predict"
    query = f"Question: {question}\nAnswer:{path}"
    if query.endswith(tokenizer.eos_token):
        query = query[:-len(tokenizer.eos_token)] # this value is not trained like this
    pload ={"texts": [query]}
    response =requests.post(url, json=pload)
    return (min(max(response.json()["values"][0], -1.), 1.) + 1.) / 2

#### Search Tree ####
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "".join(self.return_path())

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        self.all_nodes.append(self.root)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.runtime_seconds = 0.0

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, is_leaf=False):
        node = Node(content, value, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def get_beam_to_expand(self, beam_size=5):
        curr_timestep = self.return_timestep()
        latest_nodes = [node for node in self.all_nodes if node.is_leaf or node.timestep == curr_timestep]
        beam = sorted(latest_nodes, key=lambda x: x.value, reverse=True)[:beam_size]
        return [node for node in beam if not node.is_leaf]
########

dataset = []
with open(data_fpath, "r") as f:
    for line in f.readlines():
        dataset.append(json.loads(line))

problems = []
for instance in dataset:
    question = instance["question"]
    answer = instance["answer"]
    problem = Tree(question, answer)
    problems.append(problem)

import multiprocessing

def worker(tree):
    start_time = time.time()
    question = tree.question
    for _ in range(LIMIT):
        actions = tree.get_beam_to_expand(BEAM)
        if actions:
            for action in actions:
                for _ in range(BUDGET):
                    # expand this state
                    # get next step content
                    path = action.print_path()
                    next_step, usage = call_policy(question, path)
                    tree.prompt_tokens += usage.get("prompt_tokens", 0)
                    tree.completion_tokens += usage.get("completion_tokens", 0)
                    tree.total_tokens += usage.get("total_tokens", 0)
                    # get next step value
                    next_value = call_value(question, path + next_step)
                    state = tree.add_node(next_step, next_value, action, assert_end(next_step))
                    fix_value(state)
                    # print((next_step, next_value))
        else:
            break
    tree.runtime_seconds = time.time() - start_time
    return tree

pool = multiprocessing.Pool(80)
problems = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems)))    
pool.close()

pickle.dump(problems, open(output_fpath, "wb"))