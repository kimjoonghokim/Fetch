import os
import json
import pickle
import numpy as np
import jsonlines
import requests
from tqdm import tqdm
import time
from dotenv import load_dotenv
import subprocess
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load environment variables
load_dotenv(dotenv_path='../experiments_config.env')
data_fpath_var = os.getenv("PATH_TO_DATASET")
data_fpath = os.getenv(data_fpath_var) if data_fpath_var else None
if data_fpath:
    dataset_type = os.path.basename(data_fpath).split('.')[0]
    dataset_name = os.path.basename(os.path.dirname(data_fpath))
else:
    dataset_name = "unknown"
    dataset_type = "unknown"

load_dotenv(dotenv_path='../../server_config.env')
policy_fpath = os.getenv("POLICY_MODEL_PATH")

# SSDP Parameters from README
B = int(os.getenv("SSDP_B", 15))  # Batch Expansion
N = int(os.getenv("SSDP_N", 5))   # Dynamic Window Width
ALPHA = float(os.getenv("SSDP_ALPHA", 0.8)) # Relative-to-leader threshold (DEACTIVATED)
BETA = float(os.getenv("SSDP_BETA", 0.1)) # Depth-scaled minimum base
GAMMA = float(os.getenv("SSDP_GAMMA", 0.05)) # Depth-scaled minimum increment
DELTA = float(os.getenv("SSDP_DELTA", 0.01)) # Terminal convergence threshold
MAX_TERMINAL_NODES = int(os.getenv("SSDP_MAX_TERMINAL_NODES", 50))
L_CONSECUTIVE_COLLAPSE = int(os.getenv("SSDP_L_CONSECUTIVE_COLLAPSE", 3))
TEMPERATURE = float(os.getenv("SSDP_TEMPERATURE", 0.6))
DISTANCE = float(os.getenv("SSDP_DISTANCE", 0.1))
INITIAL_DIVERSITY_REWARD = float(os.getenv("SSDP_INITIAL_DIVERSITY_REWARD", 0.2))
DIVERSITY_DECAY_FACTOR = float(os.getenv("SSDP_DIVERSITY_DECAY_FACTOR", 0.9))
SIMILARITY_BONUS_SLOPE = float(os.getenv("SSDP_SIMILARITY_BONUS_SLOPE", 0.1))
CLUSTER_GLOBALLY = bool(os.getenv("SSDP_CLUSTER_GLOBALLY", True))

output_fpath = f"{dataset_type}_{dataset_name}_ssdp_b{B}_n{N}_t{TEMPERATURE}.pkl"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256

def assert_end(text):
    return True if "The answer is" in text and text.endswith(tokenizer.eos_token) else False

def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.confidence = -1
    if state.content is not None and (len(state.content) == 0 or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP): # too short or too long
        state.confidence = -1
    if state.content.endswith(tokenizer.eos_token) and not assert_end(state.content):
        state.confidence = -1
    return state

def call_policy(question, path):
    url = "http://127.0.0.1:8000/v1/completions"
    model = policy_fpath
    query = f"Question: {question}\nAnswer:{path}"
    pload ={"prompt": query, "model": model, "temperature": TEMPERATURE, "max_tokens": 512, 
            "stop": ["\n"], "include_stop_str_in_output": True, "skip_special_tokens": False, "logprobs": 1}
    response =requests.post(url, json=pload)
    response_json = response.json()
    choice = response_json["choices"][0]
    usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return choice["text"], choice.get("logprobs"), usage

def clean_text(text):
    if text.endswith(tokenizer.eos_token):
        text = text[:-len(tokenizer.eos_token)]
    return text.strip()

def call_esm(texts):
    url = "http://127.0.0.1:8003/predict"
    texts = [clean_text(text) for text in texts]
    pload ={"texts": texts, "d": DISTANCE}
    response =requests.post(url, json=pload)
    response_json = response.json()
    usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return response_json["labels"], response_json["embeddings"], usage

class Node:
    def __init__(self, content, confidence, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.confidence = confidence # Base confidence from policy
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf
        self.embedding = None
        self.similarity_bonus = 0
        self.diversity_reward = 0
        self.parent_score = parent.score if parent else 0
        self.score = 0
        self.cluster_id = None
        self.is_representative = False

    def get_depth(self):
        return self.timestep

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "".join(self.return_path())

    def update_score(self):
        self.score = self.confidence + self.similarity_bonus + self.diversity_reward + self.parent_score

class Cluster:
    def __init__(self, nodes):
        self.nodes = sorted(nodes, key=lambda x: x.confidence, reverse=True)
        self.representative = self.nodes[0]
        
        depth = self.representative.get_depth()
        dynamic_factor = depth * SIMILARITY_BONUS_SLOPE
        
        self.similarity_bonus = dynamic_factor * sum(n.confidence for n in self.nodes[1:])
        self.representative.similarity_bonus = self.similarity_bonus
        self.representative.update_score()

    @property
    def score(self):
        return self.representative.score

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.root = Node(None, 1.0, None, 0, self)
        self.root.update_score()
        self.all_nodes = [self.root]
        self.terminal_nodes = []
        self.window = [self.root]
        self.pruning_history = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.embedding_prompt_tokens = 0
        self.embedding_completion_tokens = 0
        self.embedding_total_tokens = 0
        self.runtime_seconds = 0.0

    def add_node(self, content, confidence, parent, timestep, is_leaf=False):
        node = Node(content, confidence, parent, timestep, self, is_leaf)
        if parent:
            parent.children.append(node)
        self.all_nodes.append(node)
        if is_leaf:
            self.terminal_nodes.append(node)
        return node

    def ssdp_step(self):
        if not self.window:
            return False

        # 1. Generate all candidate nodes
        candidates = []
        window_scores = [n.score for n in self.window]
        total_window_score = sum(window_scores)
        if total_window_score > 0:
            allocations = [int(B * score / total_window_score) for score in window_scores]
            remainder = B - sum(allocations)
            for i in range(remainder):
                allocations[i % len(allocations)] += 1
        else:
            allocations = [B // len(self.window)] * len(self.window)
            remainder = B % len(self.window)
            for i in range(remainder):
                allocations[i] += 1

        for node, num_to_gen in zip(self.window, allocations):
            if node.is_leaf:
                continue
            for _ in range(num_to_gen):
                path = node.print_path()
                next_step, logprobs, usage = call_policy(self.question, path)
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
                
                if logprobs and logprobs.get('token_logprobs'):
                    confidence = np.mean([p for p in logprobs.get('token_logprobs') if p is not None])
                    confidence = np.exp(confidence)
                else:
                    confidence = 0.5
                
                new_node = self.add_node(next_step, confidence, node, node.timestep + 1, assert_end(next_step))
                fix_value(new_node)
                if new_node.confidence > -1:
                    candidates.append(new_node)

        if not candidates:
            return False

        # 2. Get embeddings for all candidates
        texts = [c.content for c in candidates]
        global_labels, embeddings, usage = call_esm(texts)
        self.embedding_prompt_tokens += usage.get("prompt_tokens", 0)
        self.embedding_completion_tokens += usage.get("completion_tokens", 0)
        self.embedding_total_tokens += usage.get("total_tokens", 0)
        for node, emb in zip(candidates, embeddings):
            node.embedding = emb

        # 3. Cluster nodes based on strategy
        clusters = []
        if CLUSTER_GLOBALLY:
            clusters_map = defaultdict(list)
            for node, label in zip(candidates, global_labels):
                clusters_map[label].append(node)
            clusters = [Cluster(nodes) for nodes in clusters_map.values()]
        else: # Sibling-only clustering
            parent_to_children = defaultdict(list)
            for node in candidates:
                parent_to_children[node.parent].append(node)
            
            for parent_node, siblings in parent_to_children.items():
                if len(siblings) < 2:
                    if siblings:
                        clusters.append(Cluster(siblings))
                    continue
                
                sibling_embeddings = np.array([s.embedding for s in siblings])
                similarity_matrix = cosine_similarity(sibling_embeddings)
                clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=DISTANCE).fit(1 - similarity_matrix)
                
                sibling_clusters_map = defaultdict(list)
                for node, label in zip(siblings, clustering.labels_):
                    sibling_clusters_map[label].append(node)
                
                for nodes_in_cluster in sibling_clusters_map.values():
                    clusters.append(Cluster(nodes_in_cluster))

        # Tag nodes with cluster info for visualization
        for i, cluster in enumerate(clusters):
            if not cluster.nodes:
                continue
            cluster_id = f"T{cluster.nodes[0].timestep}_C{i}"
            for node in cluster.nodes:
                node.cluster_id = cluster_id
            cluster.representative.is_representative = True

        # 4. Score representatives and apply diversity reward
        if not clusters:
            self.window = []
            return True
            
        representatives = [c.representative for c in clusters]
        representatives.sort(key=lambda x: x.score, reverse=True)
        leader = representatives[0]
        
        for rep in representatives[1:]:
            leader_emb = np.array(leader.embedding).reshape(1, -1)
            cluster_emb = np.array(rep.embedding).reshape(1, -1)
            if cosine_similarity(leader_emb, cluster_emb)[0][0] < (1 - DISTANCE):
                depth = rep.get_depth()
                decaying_reward = INITIAL_DIVERSITY_REWARD * (DIVERSITY_DECAY_FACTOR ** depth)
                rep.diversity_reward += decaying_reward
            rep.update_score()

        # 5. Keep top N representatives
        representatives.sort(key=lambda x: x.score, reverse=True)
        top_representatives = representatives[:N]

        # 6. Pruning (with corrected rule)
        leader_score = leader.score
        event_timestep = leader.timestep
        min_score_d = BETA + GAMMA * event_timestep
        pruning_threshold = min_score_d # ALPHA rule removed

        self.pruning_history.append({
            'timestep': event_timestep,
            'leader_score': leader_score,
            'relative_threshold': ALPHA * leader_score, # Still record for analysis
            'depth_threshold': min_score_d,
            'final_threshold': pruning_threshold
        })

        self.window = [rep for rep in top_representatives if rep.score >= pruning_threshold]

        # 7. Check stopping criteria
        if self.check_stopping_criteria():
            return False
            
        return True

    def check_stopping_criteria(self):
        if len(self.terminal_nodes) >= MAX_TERMINAL_NODES:
            return True
        if len(self.window) == 0:
            return True
        return False

def worker(tree):
    start_time = time.time()
    for i in range(20):
        if not tree.ssdp_step():
            break
    tree.runtime_seconds = time.time() - start_time
    return tree

if __name__ == "__main__":
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
    start_time = time.time()

    pool = multiprocessing.Pool(80)
    problems = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems)))
    pool.close()

    total_runtime = time.time() - start_time
    
    total_prompt_tokens = sum([p.prompt_tokens for p in problems])
    total_completion_tokens = sum([p.completion_tokens for p in problems])
    total_tokens = sum([p.total_tokens for p in problems])
    total_embedding_prompt_tokens = sum([p.embedding_prompt_tokens for p in problems])
    total_embedding_completion_tokens = sum([p.embedding_completion_tokens for p in problems])
    total_embedding_tokens = sum([p.embedding_total_tokens for p in problems])
    total_all_tokens = total_tokens + total_embedding_tokens

    final_data = {
        'problems': problems,
        'metrics': {
            'total_runtime': total_runtime,
            'policy_server': {
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'embedding_server': {
                'total_prompt_tokens': total_embedding_prompt_tokens,
                'total_completion_tokens': total_embedding_completion_tokens,
                'total_tokens': total_embedding_tokens
            },
            'combined': {
                'total_tokens': total_all_tokens,
            }
        }
    }

    with open(output_fpath, "wb") as f:
        pickle.dump(final_data, f)

    print(f"=== SSDP Search Complete ===")
    print(f"Results saved to {output_fpath}")

    print(f"\nRunning evaluation script on {output_fpath}...")
    subprocess.run(["python", "eval_search.py", output_fpath])