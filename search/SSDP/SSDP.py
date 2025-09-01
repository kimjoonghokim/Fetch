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
from collections import Counter
from config import *

# File paths
output_fpath = f"test_gsm8k_ssdp_v3.pkl"

# Task dependent functions
def assert_end(text):
    """Check if the reasoning has reached a conclusion."""
    return True if text and text.strip().split("\n")[-1].startswith("The answer is") else False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)

def call_policy(question, path):
    """Call the policy model to generate next step and get logprobs."""
    url = POLICY_URL
    model = POLICY_MODEL
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
    Simplified node for the new SSDP algorithm.
    """
    
    def __init__(self, choice, parent, timestep, tree, is_leaf=False):
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf
        
        self.content = None
        self.logprobs_data = None
        
        self.overall_score = 0.0
        self.confidence_score = 0.0
        self.semantic_embedding = None
        self.merged_nodes = []
        self.pruned = False

        if choice:
            self.content = choice["text"]
            self.logprobs_data = choice.get("logprobs", {})
            self._calculate_scores()

    def _calculate_scores(self):
        """Calculate scores based only on the confidence score."""
        if not self.logprobs_data:
            return

        token_logprobs = self.logprobs_data.get("token_logprobs", [])
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if valid_logprobs:
            self.confidence_score = _calculate_average_confidence(valid_logprobs)
            self.overall_score = self.confidence_score

    def get_primary_score(self):
        return self.overall_score
    
    def get_depth(self):
        return self.timestep
    
    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content] if self.content is not None else []
        return self.parent.return_path() + ([self.content] if self.content is not None else [])
    
    def print_path(self):
        return "".join(self.return_path())
    
    def get_semantic_embedding(self, vectorizer):
        if self.semantic_embedding is None and self.content:
            text = self.content.strip()
            if text:
                try:
                    self.semantic_embedding = vectorizer.transform([text]).toarray()[0]
                except:
                    self.semantic_embedding = np.zeros(vectorizer.get_feature_names_out().shape[0])
        return self.semantic_embedding
    
    def is_similar_to(self, other_node, vectorizer, threshold=SIMILARITY_THRESHOLD):
        if self.content is None or other_node.content is None:
            return False
        
        emb1 = self.get_semantic_embedding(vectorizer)
        emb2 = other_node.get_semantic_embedding(vectorizer)
        
        if emb1 is None or emb2 is None:
            return False
        
        similarity = cosine_similarity([emb1], [emb2])[0, 0]
        return similarity >= threshold

class SSDPTree:
    """
    The new simplified SSDP Tree.
    """
    
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.root = SSDPNode(None, None, 0, self)
        self.all_nodes.append(self.root)
        
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english', ngram_range=TFIDF_NGRAM_RANGE)
        self.vectorizer_fitted = False
        
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
            texts.append(self.question)
            if texts:
                try:
                    self.vectorizer.fit(texts)
                    self.vectorizer_fitted = True
                except:
                    pass
    
    def add_node(self, choice, parent, is_leaf=False):
        node = SSDPNode(choice, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        self.total_expansions += 1
        return node
    
    def get_nodes_to_expand(self, max_parallel_paths):
        active_nodes = [n for n in self.all_nodes if not n.pruned and not n.is_leaf]
        active_nodes.sort(key=lambda x: x.get_primary_score(), reverse=True)
        return active_nodes[:max_parallel_paths]
    
    def merge_similar_nodes(self):
        # ... (merging logic remains the same)
        pass

    def prune_nodes(self):
        active_nodes = [n for n in self.all_nodes if not n.pruned and not n.is_leaf]
        if not active_nodes:
            return

        active_nodes.sort(key=lambda x: x.get_primary_score())
        
        num_to_prune = int(len(active_nodes) * PRUNE_RATIO)
        
        for i in range(num_to_prune):
            active_nodes[i].pruned = True
            self.total_prunes += 1
        
        print(f"Pruned {num_to_prune} nodes.")

    def get_best_terminal_node(self):
        terminal_nodes = [node for node in self.all_nodes if node.is_leaf and not node.pruned]
        if not terminal_nodes: return None
        return max(terminal_nodes, key=lambda x: x.get_primary_score())

def ssdp_worker(args):
    tree, max_parallel_paths = args
    question = tree.question
    iteration = 0
    
    start_time = time.time()
    tree.fit_vectorizer()
    
    print(f"\n--- Starting SSDP for question: {question[:50]}... ---")
    
    with ThreadPoolExecutor(max_workers=max_parallel_paths) as executor:
        while iteration < LIMIT:
            print(f"\n--- Iteration {iteration} ---")
            nodes_to_expand = tree.get_nodes_to_expand(max_parallel_paths)
            
            if not nodes_to_expand:
                print("No more nodes to expand. Stopping.")
                break
            
            print(f"Expanding {len(nodes_to_expand)} nodes...")
            for i, node in enumerate(nodes_to_expand):
                print(f"  Node {i}: score={node.get_primary_score():.3f}, depth={node.get_depth()}")

            future_to_node = {}
            for node in nodes_to_expand:
                path = node.print_path()
                future = executor.submit(call_policy, question, path)
                future_to_node[future] = node

            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    choice, usage = future.result()
                    
                    tree.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    tree.total_completion_tokens += usage.get("completion_tokens", 0)
                    tree.total_tokens += usage.get("total_tokens", 0)

                    is_terminal = assert_end(choice["text"])
                    new_node = tree.add_node(choice, node, is_terminal)
                    print(f"  New node created: score={new_node.get_primary_score():.3f}, depth={new_node.get_depth()}")
                except Exception as e:
                    print(f"Error expanding node: {e}")
                    continue

            if iteration % MERGE_FREQUENCY == 0:
                print("Merging similar nodes...")
                tree.merge_similar_nodes()
            
            print("Pruning low-scoring nodes...")
            tree.prune_nodes()

            iteration += 1
            
    tree.runtime_seconds = time.time() - start_time
    return tree

def prepare_problem(instance, max_parallel_paths):
    question = instance["question"]
    answer = instance["answer"]
    tree = SSDPTree(question, answer)
    return (tree, max_parallel_paths)

#### Main Execution ####
def main():
    """Main execution function."""
    print("Loading dataset...")
    
    with open(DATA_PATH, "r") as f:
        num_lines = sum(1 for line in f)

    print(f"Found {num_lines} problems.")

    processed_problems = []
    
    print("Starting SSDP search with multiprocessing...")
    
    with Pool(processes=os.cpu_count()) as pool:
        with open(DATA_PATH, "r") as f:
            problem_generator = (json.loads(line) for line in f)
            worker_args = (prepare_problem(instance, MAX_PARALLEL_PATHS) for instance in problem_generator)

            try:
                for result in tqdm(pool.imap(ssdp_worker, worker_args), total=num_lines, desc="SSDP Search"):
                    processed_problems.append(result)
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
        total_runtime = sum(p.runtime_seconds for p in processed_problems)
        total_prompt_tokens = sum(p.total_prompt_tokens for p in processed_problems)
        total_completion_tokens = sum(p.total_completion_tokens for p in processed_problems)
        total_tokens = sum(p.total_tokens for p in processed_problems)

        print(f"Total nodes created: {total_nodes}")
        print(f"Total expansions: {total_expansions}")
        print(f"Total merges: {total_merges}")
        print(f"Total prunes: {total_prunes}")
        if len(processed_problems) > 0:
            print(f"Average nodes per problem: {total_nodes / len(processed_problems):.1f}")
        print(f"\n--- Metrics ---")
        print(f"Total runtime: {total_runtime:.2f} seconds")
        if len(processed_problems) > 0:
            print(f"Average runtime per problem: {total_runtime / len(processed_problems):.2f} seconds")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
