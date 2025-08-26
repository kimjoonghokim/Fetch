import os
import json
import pickle
import numpy as np
import requests
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed

# SSDP Configuration
LIMIT = 50                    # Maximum search iterations
MAX_PARALLEL_PATHS = 8        # Maximum number of parallel paths to explore
MIN_EXPANSION_BUDGET = 3      # Minimum expansions per node
MAX_EXPANSION_BUDGET = 5      # Maximum expansions per node
TEMPERATURE = 0.8             # Model temperature
OVERALL_SCORE_THRESHOLD = 0.3 # Minimum overall score to keep a path
SIMILARITY_THRESHOLD = 0.85   # Similarity threshold for merging nodes
PRUNE_FREQUENCY = 3           # Prune every N iterations
MAX_DEPTH = 10               # Maximum reasoning depth

# File paths
data_fpath = "../../dataset/toy.jsonl"
output_fpath = f"test_gsm8k_ssdp_p{MAX_PARALLEL_PATHS}_t{OVERALL_SCORE_THRESHOLD}.pkl"
policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k"

# Task dependent functions
def assert_end(text):
    """Check if the reasoning has reached a conclusion."""
    return True if text and text.strip().split("\n")[-1].startswith("The answer is") else False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256

def call_policy(question, path):
    """Call the policy model to generate next step and get logprobs."""
    url = "http://127.0.0.1:8000/v1/completions"
    model = policy_fpath
    query = f"Question: {question}\nAnswer:{path}"
    pload = {
        "prompt": query, 
        "model": model, 
        "temperature": TEMPERATURE, 
        "max_tokens": 512,
        "stop": ["\n"], 
        "logprobs": 5,  # Request logprobs for scoring
        "include_stop_str_in_output": True, 
        "skip_special_tokens": False
    }
    response = requests.post(url, json=pload)
    response_json = response.json()
    choice = response_json["choices"][0]
    usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    
    return choice, usage

#### Helper for Scoring ####

def _calculate_average_confidence(logprobs: List[float]) -> float:
    """Calculate average log probability confidence."""
    if not logprobs:
        return 0.0
    avg_logprob = np.mean(logprobs)
    return float(np.exp(avg_logprob))

#### SSDP Search Tree ####

class SSDPNode:
    """
    Enhanced node for SSDP algorithm with confidence score calculated from generation metadata.
    """
    
    def __init__(self, choice, parent, timestep, tree, is_leaf=False):
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf
        
        # Attributes from the generation choice
        self.content = None
        self.logprobs_data = None
        
        # SSDP-specific attributes
        self.overall_score = 0.0
        self.confidence_score = 0.0
        self.detailed_scores = None
        self.semantic_embedding = None
        self.merged_nodes = []  # Track nodes that were merged into this one
        self.pruned = False

        if choice:
            self.content = choice["text"]
            self.logprobs_data = choice.get("logprobs", {})
            self._calculate_scores_from_logprobs()

    def _calculate_scores_from_logprobs(self):
        """Calculate scores directly from the logprobs of the generation call."""
        if not self.logprobs_data:
            return

        token_logprobs = self.logprobs_data.get("token_logprobs", [])
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        
        if not valid_logprobs:
            return

        # The overall score is the average confidence
        self.confidence_score = _calculate_average_confidence(valid_logprobs)
        self.overall_score = self.confidence_score
        
        # Store detailed info for potential analysis
        self.detailed_scores = {
            'confidence': self.confidence_score,
            'details': {
                'avg_confidence': self.confidence_score,
                'token_logprobs': token_logprobs
            }
        }

    def get_primary_score(self):
        """Get the primary score for ranking (overall score)."""
        return self.overall_score
    
    def get_depth(self):
        return len(self.return_path()) + 1
    
    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content] if self.content is not None else []
        return self.parent.return_path() + ([self.content] if self.content is not None else [])
    
    def print_path(self):
        return "".join(self.return_path())
    
    def get_semantic_embedding(self, vectorizer):
        """Get semantic embedding for similarity comparison."""
        if self.semantic_embedding is None and self.content:
            text = self.content.strip()
            if text:
                try:
                    self.semantic_embedding = vectorizer.transform([text]).toarray()[0]
                except:
                    self.semantic_embedding = np.zeros(vectorizer.get_feature_names_out().shape[0])
        return self.semantic_embedding
    
    def is_similar_to(self, other_node, vectorizer, threshold=SIMILARITY_THRESHOLD):
        """Check if this node is semantically similar to another node."""
        if self.content is None or other_node.content is None:
            return False
        
        emb1 = self.get_semantic_embedding(vectorizer)
        emb2 = other_node.get_semantic_embedding(vectorizer)
        
        if emb1 is None or emb2 is None:
            return False
        
        similarity = cosine_similarity([emb1], [emb2])[0, 0]
        return similarity >= threshold
    
    def merge_with(self, other_node):
        """Merge another node into this one."""
        if other_node.get_primary_score() > self.get_primary_score():
            self.overall_score = other_node.overall_score
            self.confidence_score = other_node.confidence_score
            self.detailed_scores = other_node.detailed_scores
        
        self.merged_nodes.append(other_node)
        self.merged_nodes.extend(other_node.merged_nodes)
        other_node.pruned = True
    
    def get_score_breakdown(self):
        """Get detailed score breakdown for analysis."""
        if self.detailed_scores:
            return {
                'overall_score': self.overall_score,
                'confidence_score': self.confidence_score,
                'component_scores': self.detailed_scores
            }
        return {'overall_score': self.overall_score, 'confidence_score': self.confidence_score}

class SSDPTree:
    """
    SSDP Tree with semantic similarity-based dynamic pruning.
    """
    
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.pruned_nodes = []
        self.root = SSDPNode(None, None, 0, self)
        self.all_nodes.append(self.root)
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.vectorizer_fitted = False
        
        # Statistics & Metrics
        self.total_expansions = 0
        self.total_merges = 0
        self.total_prunes = 0
        self.runtime_seconds = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
    
    def fit_vectorizer(self):
        if not self.vectorizer_fitted:
            texts = [node.content for node in self.all_nodes if node.content is not None]
            if texts:
                try:
                    self.vectorizer.fit(texts)
                    self.vectorizer_fitted = True
                except:
                    pass
    
    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes if not node.pruned])
    
    def add_node(self, choice, parent, is_leaf=False):
        """Add a new node to the tree."""
        node = SSDPNode(choice, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        self.total_expansions += 1
        return node
    
    def get_active_nodes_at_level(self, timestep=None):
        if timestep is None:
            timestep = self.return_timestep()
        return [node for node in self.all_nodes if node.timestep == timestep and not node.pruned and not node.is_leaf]
    
    def get_nodes_to_expand(self, max_paths=MAX_PARALLEL_PATHS):
        current_level_nodes = self.get_active_nodes_at_level()
        if not current_level_nodes:
            prev_timestep = self.return_timestep() - 1
            if prev_timestep >= 0:
                current_level_nodes = self.get_active_nodes_at_level(prev_timestep)
        
        scored_nodes = [(node, node.get_primary_score()) for node in current_level_nodes]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, score in scored_nodes[:max_paths]]
    
    def merge_similar_nodes(self, nodes=None):
        if nodes is None:
            nodes = self.get_active_nodes_at_level()
        if not nodes or not self.vectorizer_fitted:
            return
        
        merged_pairs = []
        for i, node1 in enumerate(nodes):
            if node1.pruned: continue
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if node2.pruned: continue
                if node1.is_similar_to(node2, self.vectorizer):
                    if node2.get_primary_score() > node1.get_primary_score():
                        node2.merge_with(node1)
                    else:
                        node1.merge_with(node2)
                    self.total_merges += 1
    
    def prune_low_scoring_nodes(self, threshold=OVERALL_SCORE_THRESHOLD):
        for node in self.all_nodes:
            if not node.pruned and node.content is not None and node.get_primary_score() < threshold:
                node.pruned = True
                self.pruned_nodes.append(node)
                self.total_prunes += 1
    
    def adaptive_expansion_budget(self, node):
        score = node.get_primary_score()
        if score >= 0.8: return MAX_EXPANSION_BUDGET
        elif score >= 0.6: return MAX_EXPANSION_BUDGET - 1
        elif score >= 0.4: return MIN_EXPANSION_BUDGET + 1
        else: return MIN_EXPANSION_BUDGET
    
    def get_best_terminal_node(self):
        terminal_nodes = [node for node in self.all_nodes if node.is_leaf and not node.pruned]
        if not terminal_nodes: return None
        return max(terminal_nodes, key=lambda x: x.get_primary_score())
    
    def print_statistics(self):
        active_nodes = len([n for n in self.all_nodes if not n.pruned])
        terminal_nodes = len([n for n in self.all_nodes if n.is_leaf and not n.pruned])
        print(f"SSDP Statistics:")
        print(f"  Total nodes: {len(self.all_nodes)}, Active: {active_nodes}, Terminal: {terminal_nodes}")
        print(f"  Total expansions: {self.total_expansions}, Merges: {self.total_merges}, Prunes: {self.total_prunes}")
        if terminal_nodes > 0:
            best_terminal = self.get_best_terminal_node()
            print(f"  Best terminal score: {best_terminal.get_primary_score():.3f}")

def fix_value(node):
    if node.parent is not None and node.content is not None:
        if node.parent.content == node.content:
            node.overall_score = 0.0
    if node.content is not None and (len(node.content) == 0 or len(tokenizer.tokenize(node.content)) > MAX_LEN_PER_STEP):
        node.overall_score = 0.0
    if node.content and node.content.endswith(tokenizer.eos_token) and not assert_end(node.content):
        node.overall_score = 0.0
    return node

def ssdp_worker(tree):
    question = tree.question
    iteration = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATHS) as executor:
        while iteration < LIMIT:
            nodes_to_expand = tree.get_nodes_to_expand(MAX_PARALLEL_PATHS)
            if not nodes_to_expand:
                break
            
            current_depth = max([node.get_depth() for node in nodes_to_expand]) if nodes_to_expand else 0
            if current_depth >= MAX_DEPTH:
                break
            
            future_to_node = {}
            for node in nodes_to_expand:
                if node.pruned or node.is_leaf: continue
                budget = tree.adaptive_expansion_budget(node)
                for _ in range(budget):
                    path = node.print_path()
                    future = executor.submit(call_policy, question, path)
                    future_to_node[future] = node

            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    choice, usage = future.result()
                    
                    # Update tree-specific token counts
                    tree.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    tree.total_completion_tokens += usage.get("completion_tokens", 0)
                    tree.total_tokens += usage.get("total_tokens", 0)

                    is_terminal = assert_end(choice["text"])
                    new_node = tree.add_node(choice, node, is_terminal)
                    fix_value(new_node)
                except Exception as e:
                    print(f"Error expanding node: {e}")
                    continue

            if iteration == 0:
                tree.fit_vectorizer()
            if iteration % 2 == 0 and tree.vectorizer_fitted:
                tree.merge_similar_nodes()
            if iteration % PRUNE_FREQUENCY == 0:
                tree.prune_low_scoring_nodes(OVERALL_SCORE_THRESHOLD)
            
            iteration += 1
            best_terminal = tree.get_best_terminal_node()
            if best_terminal and best_terminal.get_primary_score() >= 0.8:
                break
            
tree.runtime_seconds = time.time() - start_time
    return tree

def prepare_problem(instance):
    """Prepare a problem for the worker by creating an SSDPTree."""
    question = instance["question"]
    answer = instance["answer"]
    tree = SSDPTree(question, answer)
    return tree

#### Main Execution ####

def main():
    """Main execution function."""
    print("Loading dataset...")
    
    # Count the number of lines for tqdm
    with open(data_fpath, "r") as f:
        num_lines = sum(1 for line in f)

    print(f"Found {num_lines} problems.")

    manager = Manager()
    shared_data = manager.dict({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    processed_problems = []
    
    print("Starting SSDP search with multiprocessing...")
    
    with Pool(processes=os.cpu_count()) as pool:
        with open(data_fpath, "r") as f:
            # Create a generator for problems
            problem_generator = (json.loads(line) for line in f)
            
            # Prepare arguments for the worker
            worker_args = (prepare_problem(instance) for instance in problem_generator)

            try:
                for result in tqdm(pool.imap(ssdp_worker, worker_args), total=num_lines, desc="SSDP Search"):
                    processed_problems.append(result)
                    shared_data["prompt_tokens"] += result.total_prompt_tokens
                    shared_data["completion_tokens"] += result.total_completion_tokens
                    shared_data["total_tokens"] += result.total_tokens
            except Exception as e:
                print(f"An error occurred during multiprocessing: {e}")
            finally:
                pool.close()
                pool.join()


    print(f"Saving results to {output_fpath}")
    with open(output_fpath, "wb") as f:
        pickle.dump(processed_problems, f)

    # Print summary statistics
    if processed_problems:
        print("\n=== SSDP Search Complete ===")
        total_nodes = sum(len(p.all_nodes) for p in processed_problems)
        total_expansions = sum(p.total_expansions for p in processed_problems)
        total_merges = sum(p.total_merges for p in processed_problems)
        total_prunes = sum(p.total_prunes for p in processed_problems)
        total_runtime = sum(p.runtime_seconds for p in processed_problems if hasattr(p, 'runtime_seconds'))

        print(f"Total nodes created: {total_nodes}")
        print(f"Total expansions: {total_expansions}")
        print(f"Total merges: {total_merges}")
        print(f"Total prunes: {total_prunes}")
        if len(processed_problems) > 0:
            print(f"Average nodes per problem: {total_nodes / len(processed_problems):.1f}")
        print(f"\n--- Metrics ---")
        print(f"Total runtime: {total_runtime:.2f} seconds")
        if len(processed_problems) > 0:
            valid_runtimes = [p.runtime_seconds for p in processed_problems if hasattr(p, 'runtime_seconds')]
            if valid_runtimes:
                print(f"Average runtime per problem: {sum(valid_runtimes) / len(valid_runtimes):.2f} seconds")
        print(f"Total prompt tokens: {shared_data['prompt_tokens']}")
        print(f"Total completion tokens: {shared_data['completion_tokens']}")
        print(f"Total tokens: {shared_data['total_tokens']}")


if __name__ == "__main__":
    main()
