import os
import json
import pickle
import numpy as np
import jsonlines
import requests
from tqdm import tqdm
from dotenv import load_dotenv

LIMIT=50
BUDGET=5
BEAM=5
TEMPERATURE=0.8
DISTANCE=0.15
data_fpath = "gsm8k/test.jsonl" # path to the test set
output_fpath = f"test_gsm8k_beamsearch_merge_b{BUDGET}_t{TEMPERATURE}.pkl" # path to the output file
load_dotenv(dotenv_path='../../server_config.env')
policy_fpath = os.getenv("POLICY_MODEL_PATH") # path to the policy model


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
    return json.loads(response.content)["choices"][0]["text"]

def call_value(question, path):
    url = "http://127.0.0.1:8002/predict"
    query = f"Question: {question}\nAnswer:{path}"
    if query.endswith(tokenizer.eos_token):
        query = query[:-len(tokenizer.eos_token)] # this value is not trained like this
    pload ={"texts": [query]}
    response =requests.post(url, json=pload)
    return (min(max(response.json()["values"][0], -1.), 1.) + 1.) / 2

def clean_text(text):
    if text.endswith(tokenizer.eos_token):
        text = text[:-len(tokenizer.eos_token)]
    return text.strip()

def call_esm(texts):
    url = "http://127.0.0.1:8003/predict"
    texts = [clean_text(text) for text in texts]
    pload ={"texts": texts, "d": DISTANCE}
    response =requests.post(url, json=pload)
    return response.json()["labels"]

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

class VirtualNode:
    def __init__(self, nodes, parent=None):
        self.nodes = sorted(nodes, key=lambda x: x.value, reverse=True)
        self.tree = self.nodes[0].tree
        self.value = self.nodes[0].value
        self.visited = False
        self.children = []
        self.cache = []
        self.parent = parent
        self.is_leaf = self.nodes[0].is_leaf
        self.timestep = self.nodes[0].timestep
    
    def get_depth(self):
        return self.nodes[0].get_depth()

    def merge_nodes(self):
        if len(self.cache) == 0:
            return
        groups = [[node for node in self.cache if node.is_leaf], [node for node in self.cache if not node.is_leaf]]
        clusters = {}
        for gid, group in enumerate(groups):
            if len(group) > 0:
                labels = call_esm([node.content for node in group])
                for node, label in zip(group, labels):
                    key = (gid, label)
                    if key not in clusters:
                        clusters[key] = []
                    clusters[key].append(node)
        # for k, v in clusters.items():
        #     print(k, len(clusters))
        #     print([node.content for node in v])
        for nodes in clusters.values():
            virtual_node = VirtualNode(nodes, self)
            self.children.append(virtual_node)
            self.tree.virtual_nodes.append(virtual_node)
            self.cache = [] # clean cache


class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        self.virtual_nodes = [VirtualNode([self.root])]
        self.all_nodes.append(self.root)

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, timestep, is_leaf=False):
        node = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def get_node_to_expand(self):
        unexp_nodes = [node for node in self.all_nodes if node.value > -1 and len(node.children) == 0]
        if unexp_nodes:
            best_node = max(unexp_nodes, key=lambda x: x.value)
            if not best_node.is_leaf:
                return best_node
        return None
    
    def get_cluster_to_expand(self):
        best_cluster = None
        for virtual_node in self.virtual_nodes:
            if not virtual_node.visited:
                if best_cluster is None or virtual_node.value > best_cluster.value:
                    best_cluster = virtual_node
        if best_cluster is None: # all have been visited or have not started
            return None, None
        best_cluster.visited = True
        return best_cluster, [node for node in best_cluster.nodes if not node.is_leaf ]
    
    def get_beam_to_expand(self, beam_size):
        curr_timestep = self.return_timestep()
        latest_nodes = [node for node in self.virtual_nodes if node.is_leaf or node.timestep == curr_timestep]
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
    question = tree.question
    for i in range(LIMIT):
        actions = tree.get_beam_to_expand(BEAM)
        if actions:
            for _action in actions:
                for j in range(BUDGET):
                    # expand this state
                    # get next step content
                    action = _action.nodes[j % len(_action.nodes)]
                    path = action.print_path()
                    next_step = call_policy(question, path)
                    # get next step value
                    next_value = call_value(question, path + next_step)
                    state = tree.add_node(next_step, next_value, action, i + 1, assert_end(next_step))
                    fix_value(state)
                    # print((next_step, next_value))
                    if state.value > -1:
                        _action.cache.append(state)
                _action.merge_nodes()
        else:
            break
    return tree

pool = multiprocessing.Pool(80)
problems = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems)))    
pool.close()

pickle.dump(problems, open(output_fpath, "wb"))