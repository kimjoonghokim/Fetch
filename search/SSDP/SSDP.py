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
output_fpath = f"test_gsm8k_ssdp_v2.pkl"

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
    Enhanced node for the new SSDP algorithm.
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
        self.semantic_embedding = None
        self.merged_nodes = []
        self.pruned = False
        self.status = "exploit"  # New: explore/exploit status

        if parent:
            self.status = parent.status

        if choice:
            self.content = choice["text"]
            self.logprobs_data = choice.get("logprobs", {})
            self._calculate_scores()

    def _calculate_scores(self):
        """Calculate scores based on confidence, parent's score, and other factors."""
        if not self.logprobs_data:
            return

        # Confidence Score
        token_logprobs = self.logprobs_data.get("token_logprobs", [])
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if valid_logprobs:
            self.confidence_score = _calculate_average_confidence(valid_logprobs)
        
        # Parent's Score Inheritance
        parent_score = self.parent.overall_score if self.parent else 0.0
        
        # Combine scores (tunable weights can be added to config.py)
        self.overall_score = (0.7 * self.confidence_score) + (0.3 * parent_score)

    def update_score_with_merging(self, similar_nodes_count):
        """Update the score based on the number of similar nodes (voting)."""
        # The more similar nodes, the higher the score
        self.overall_score += (similar_nodes_count * 0.05) # Tunable factor

    def get_primary_score(self):
        """Get the primary score for ranking (overall score)."""
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

class SSDPTree:
    """
    The new SSDP Tree with all the new features.
    """
    
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.root = SSDPNode(None, None, 0, self)
        self.all_nodes.append(self.root)
        
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english', ngram_range=TFIDF_NGRAM_RANGE)
        self.vectorizer_fitted = False
        
        # Statistics & Metrics
        self.total_expansions = 0
        self.total_merges = 0
        self.total_prunes = 0
        self.runtime_seconds = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        # Early stopping
        self.best_score_so_far = 0.0
        self.patience_counter = 0

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
        """Add a new node to the tree."""
        node = SSDPNode(choice, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        self.total_expansions += 1
        return node
    
    def get_nodes_to_expand(self):
        exploit_nodes = [n for n in self.all_nodes if not n.pruned and not n.is_leaf and n.status == "exploit"]
        explore_nodes = [n for n in self.all_nodes if not n.pruned and not n.is_leaf and n.status == "explore"]
        
        exploit_nodes.sort(key=lambda x: x.get_primary_score(), reverse=True)
        explore_nodes.sort(key=lambda x: x.get_primary_score(), reverse=True)

        # Prioritize exploit nodes, then explore nodes
        return exploit_nodes[:MAX_PARALLEL_PATHS] + explore_nodes[:MAX_PARALLEL_PATHS]
    
    def merge_similar_nodes(self):
        nodes_at_current_level = [n for n in self.all_nodes if not n.pruned and not n.is_leaf]
        if not nodes_at_current_level or not self.vectorizer_fitted:
            return

        # Group nodes by level
        nodes_by_level = {}
        for node in nodes_at_current_level:
            level = node.get_depth()
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        for level, nodes in nodes_by_level.items():
            merged_in_level = 0
            for i, node1 in enumerate(nodes):
                if node1.pruned: continue
                similar_nodes = []
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    if node2.pruned: continue
                    if node1.is_similar_to(node2, self.vectorizer):
                        similar_nodes.append(node2)
                
                if similar_nodes:
                    # Merge similar nodes into the one with the highest score
                    all_nodes_to_merge = [node1] + similar_nodes
                    best_node = max(all_nodes_to_merge, key=lambda x: x.get_primary_score())
                    for node_to_merge in all_nodes_to_merge:
                        if node_to_merge != best_node:
                            best_node.merged_nodes.append(node_to_merge)
                            node_to_merge.pruned = True
                            merged_in_level += 1
                    best_node.update_score_with_merging(len(similar_nodes))
            self.total_merges += merged_in_level

    def prune_nodes(self, iteration):
        pruned_count = 0
        for node in self.all_nodes:
            if not node.pruned:
                # Depth-aware and budget-aware pruning threshold
                depth_penalty = node.get_depth() * DEPTH_AWARE_PRUNING_FACTOR
                budget_penalty = (iteration / LIMIT) * BUDGET_AWARE_PRUNING_FACTOR
                dynamic_threshold = OVERALL_SCORE_THRESHOLD + depth_penalty + budget_penalty

                if node.get_primary_score() < dynamic_threshold:
                    node.pruned = True
                    pruned_count += 1
        self.total_prunes += pruned_count

    def check_early_stopping(self):
        best_terminal = self.get_best_terminal_node()
        if best_terminal:
            if best_terminal.get_primary_score() > self.best_score_so_far + EARLY_STOPPING_THRESHOLD:
                self.best_score_so_far = best_terminal.get_primary_score()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        return self.patience_counter >= EARLY_STOPPING_PATIENCE

    def get_best_terminal_node(self):
        terminal_nodes = [node for node in self.all_nodes if node.is_leaf and not node.pruned]
        if not terminal_nodes: return None
        return max(terminal_nodes, key=lambda x: x.get_primary_score())

def fix_value(node):
    if node.parent is not None and node.content is not None:
        if node.parent.content == node.content:
            node.overall_score = 0.0
    if node.content is not None and (len(node.content) == 0 or len(tokenizer.tokenize(node.content)) > MAX_LEN_PER_STEP):
        node.overall_score = 0.0
    if node.content and node.content.endswith(tokenizer.eos_token) and not assert_end(node.content):
        node.overall_score = 0.0
    
    path_text = node.print_path()
    if len(path_text) > MAX_PATH_LENGTH:
        node.overall_score = 0.0

    if is_repetitive(path_text, REPETITION_PENALTY):
        node.overall_score = 0.0
        
    if is_dissimilar_to_question(node.content, node.tree.question, node.tree.vectorizer, MIN_QUESTION_SIMILARITY):
        node.overall_score = 0.0

    # Update explore/exploit status
    if node.status == "exploit" and node.get_primary_score() < EXPLORE_EXPLOIT_THRESHOLD:
        node.status = "explore"
        
    return node

def ssdp_worker(tree):
    question = tree.question
    iteration = 0
    
    start_time = time.time()
    tree.fit_vectorizer()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATHS) as executor:
        while iteration < LIMIT:
            nodes_to_expand = tree.get_nodes_to_expand()
            if not nodes_to_expand:
                break
            
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
                    fix_value(new_node)
                except Exception as e:
                    print(f"Error expanding node: {e}")
                    continue

            if iteration % MERGE_FREQUENCY == 0:
                tree.merge_similar_nodes()
            if iteration % PRUNE_FREQUENCY == 0:
                tree.prune_nodes(iteration)
            
            if tree.check_early_stopping():
                print("Early stopping due to no improvement.")
                break

            iteration += 1
            
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
    
    with open(DATA_PATH, "r") as f:
        num_lines = sum(1 for line in f)

    print(f"Found {num_lines} problems.")

    processed_problems = []
    
    print("Starting SSDP search with multiprocessing...")
    
    with Pool(processes=os.cpu_count()) as pool:
        with open(DATA_PATH, "r") as f:
            problem_generator = (json.loads(line) for line in f)
            worker_args = (prepare_problem(instance) for instance in problem_generator)

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
