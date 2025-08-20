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

# Add the scorer directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scorer'))
from scoring import get_overall_answer_score, AnswerScorer

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
    return True if "The answer is" in text and text.endswith(tokenizer.eos_token) else False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256

def call_policy(question, path):
    """Call the policy model to generate next step."""
    url = "http://127.0.0.1:8000/v1/completions"
    model = policy_fpath
    query = f"Question: {question}\nAnswer:{path}"
    pload = {
        "prompt": query, 
        "model": model, 
        "temperature": TEMPERATURE, 
        "max_tokens": 512,
        "stop": ["\n"], 
        "include_stop_str_in_output": True, 
        "skip_special_tokens": False
    }
    response = requests.post(url, json=pload)
    return json.loads(response.content)["choices"][0]["text"]

#### SSDP Search Tree ####

class SSDPNode:
    """
    Enhanced node for SSDP algorithm with comprehensive scoring from scoring.py.
    """
    
    def __init__(self, content, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf
        
        # SSDP-specific attributes
        self.overall_score = 0.0
        self.confidence_score = 0.0
        self.detailed_scores = None
        self.semantic_embedding = None
        self.merged_nodes = []  # Track nodes that were merged into this one
        self.pruned = False
        
        # Calculate scores
        self._calculate_scores()
    
    def _calculate_scores(self):
        """Calculate scores using the scoring.py system."""
        if self.content is not None:
            question = self.tree.question
            path = self.print_path()
            
            try:
                # Get overall score (our main score)
                self.overall_score = get_overall_answer_score(question, path)
                
                # Get detailed breakdown for analysis
                scorer = AnswerScorer()
                detailed_result = scorer.get_overall_score(question, path)
                self.detailed_scores = detailed_result
                
                # Extract confidence component specifically
                confidence_component = detailed_result['component_scores'].get('confidence', {})
                self.confidence_score = confidence_component.get('score', 0.0)
                
            except Exception as e:
                print(f"Error getting scores: {e}")
                self.overall_score = 0.0
                self.confidence_score = 0.0
                self.detailed_scores = None
    
    def get_primary_score(self):
        """Get the primary score for ranking (overall score)."""
        return self.overall_score
    
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
        # Keep the better score
        if other_node.get_primary_score() > self.get_primary_score():
            self.overall_score = other_node.overall_score
            self.confidence_score = other_node.confidence_score
            self.detailed_scores = other_node.detailed_scores
        
        # Track merged nodes
        self.merged_nodes.append(other_node)
        self.merged_nodes.extend(other_node.merged_nodes)
        
        # Mark other node as pruned
        other_node.pruned = True
    
    def get_score_breakdown(self):
        """Get detailed score breakdown for analysis."""
        if self.detailed_scores:
            return {
                'overall_score': self.overall_score,
                'confidence_score': self.confidence_score,
                'component_scores': self.detailed_scores['component_scores'],
                'weights_used': self.detailed_scores['weights_used']
            }
        return {'overall_score': self.overall_score, 'confidence_score': self.confidence_score}

class SSDPTree:
    """
    SSDP Tree with semantic similarity-based dynamic pruning using scoring.py.
    """
    
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.pruned_nodes = []
        self.root = SSDPNode(None, None, 0, self)
        self.all_nodes.append(self.root)
        
        # Semantic similarity components
        self.vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        self.vectorizer_fitted = False
        
        # Statistics
        self.total_expansions = 0
        self.total_merges = 0
        self.total_prunes = 0
    
    def fit_vectorizer(self):
        """Fit the vectorizer on existing node contents."""
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
    
    def add_node(self, content, parent, is_leaf=False):
        """Add a new node to the tree."""
        node = SSDPNode(content, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        self.total_expansions += 1
        return node
    
    def get_active_nodes_at_level(self, timestep=None):
        """Get active (non-pruned) nodes at a specific timestep."""
        if timestep is None:
            timestep = self.return_timestep()
        
        return [node for node in self.all_nodes 
                if node.timestep == timestep and not node.pruned and not node.is_leaf]
    
    def get_nodes_to_expand(self, max_paths=MAX_PARALLEL_PATHS):
        """Get the best nodes to expand based on overall scores."""
        current_level_nodes = self.get_active_nodes_at_level()
        
        if not current_level_nodes:
            # Look at previous level if current is empty
            prev_timestep = self.return_timestep() - 1
            if prev_timestep >= 0:
                current_level_nodes = self.get_active_nodes_at_level(prev_timestep)
        
        # Sort by overall score and take top candidates
        scored_nodes = [(node, node.get_primary_score()) for node in current_level_nodes]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, score in scored_nodes[:max_paths]]
    
    def merge_similar_nodes(self, nodes=None):
        """Merge semantically similar nodes at the same level."""
        if nodes is None:
            nodes = self.get_active_nodes_at_level()
        
        if not nodes or not self.vectorizer_fitted:
            return
        
        merged_pairs = []
        
        for i, node1 in enumerate(nodes):
            if node1.pruned:
                continue
                
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if node2.pruned:
                    continue
                
                if node1.is_similar_to(node2, self.vectorizer):
                    # Merge node2 into node1 (keep the better one as primary)
                    if node2.get_primary_score() > node1.get_primary_score():
                        node2.merge_with(node1)
                        merged_pairs.append((node2, node1))
                    else:
                        node1.merge_with(node2)
                        merged_pairs.append((node1, node2))
                    
                    self.total_merges += 1
        
        return merged_pairs
    
    def prune_low_scoring_nodes(self, threshold=OVERALL_SCORE_THRESHOLD):
        """Prune nodes below overall score threshold."""
        nodes_to_prune = []
        
        for node in self.all_nodes:
            if (not node.pruned and 
                node.content is not None and 
                node.get_primary_score() < threshold):
                
                node.pruned = True
                nodes_to_prune.append(node)
                self.pruned_nodes.append(node)
                self.total_prunes += 1
        
        return nodes_to_prune
    
    def adaptive_expansion_budget(self, node):
        """Determine expansion budget based on node quality."""
        score = node.get_primary_score()
        
        if score >= 0.8:
            return MAX_EXPANSION_BUDGET
        elif score >= 0.6:
            return MAX_EXPANSION_BUDGET - 1
        elif score >= 0.4:
            return MIN_EXPANSION_BUDGET + 1
        else:
            return MIN_EXPANSION_BUDGET
    
    def get_best_terminal_node(self):
        """Get the best terminal (leaf) node."""
        terminal_nodes = [node for node in self.all_nodes 
                         if node.is_leaf and not node.pruned]
        
        if not terminal_nodes:
            return None
        
        return max(terminal_nodes, key=lambda x: x.get_primary_score())
    
    def print_statistics(self):
        """Print search statistics."""
        active_nodes = len([n for n in self.all_nodes if not n.pruned])
        terminal_nodes = len([n for n in self.all_nodes if n.is_leaf and not n.pruned])
        
        print(f"SSDP Statistics:")
        print(f"  Total nodes: {len(self.all_nodes)}")
        print(f"  Active nodes: {active_nodes}")
        print(f"  Terminal nodes: {terminal_nodes}")
        print(f"  Total expansions: {self.total_expansions}")
        print(f"  Total merges: {self.total_merges}")
        print(f"  Total prunes: {self.total_prunes}")
        
        if terminal_nodes > 0:
            best_terminal = self.get_best_terminal_node()
            print(f"  Best terminal score: {best_terminal.get_primary_score():.3f}")

def fix_value(node):
    """Apply validation and fixing rules to node values."""
    if node.parent is not None:
        if node.parent.content == node.content:
            node.overall_score = 0.0
            node.confidence_score = 0.0
    
    if (node.content is not None and 
        (len(node.content) == 0 or 
         len(tokenizer.tokenize(node.content)) > MAX_LEN_PER_STEP)):
        node.overall_score = 0.0
        node.confidence_score = 0.0
    
    if (node.content and 
        node.content.endswith(tokenizer.eos_token) and 
        not assert_end(node.content)):
        node.overall_score = 0.0
        node.confidence_score = 0.0
    
    return node

#### Main SSDP Algorithm ####

def ssdp_worker(tree):
    """
    SSDP worker function implementing the main search algorithm.
    """
    question = tree.question
    iteration = 0
    
    print(f"Starting SSDP for question: {question[:50]}...")
    
    while iteration < LIMIT:
        # Get nodes to expand
        nodes_to_expand = tree.get_nodes_to_expand(MAX_PARALLEL_PATHS)
        
        if not nodes_to_expand:
            print(f"No nodes to expand at iteration {iteration}")
            break
        
        # Check if we've reached maximum depth
        current_depth = max([node.get_depth() for node in nodes_to_expand])
        if current_depth >= MAX_DEPTH:
            print(f"Reached maximum depth {MAX_DEPTH}")
            break
        
        # Expand each selected node
        for node in nodes_to_expand:
            if node.pruned or node.is_leaf:
                continue
            
            # Determine expansion budget for this node
            budget = tree.adaptive_expansion_budget(node)
            
            # Expand node multiple times
            for _ in range(budget):
                try:
                    # Generate next step
                    path = node.print_path()
                    next_step = call_policy(question, path)
                    
                    # Create new node
                    is_terminal = assert_end(next_step)
                    new_node = tree.add_node(next_step, node, is_terminal)
                    
                    # Apply validation rules
                    fix_value(new_node)
                    
                except Exception as e:
                    print(f"Error expanding node: {e}")
                    continue
        
        # Fit vectorizer after first expansion
        if iteration == 0:
            tree.fit_vectorizer()
        
        # Merge similar nodes every few iterations
        if iteration % 2 == 0 and tree.vectorizer_fitted:
            tree.merge_similar_nodes()
        
        # Prune low-scoring nodes periodically
        if iteration % PRUNE_FREQUENCY == 0:
            tree.prune_low_scoring_nodes(OVERALL_SCORE_THRESHOLD)
        
        iteration += 1
        
        # Early stopping if we have good terminal nodes
        best_terminal = tree.get_best_terminal_node()
        if best_terminal and best_terminal.get_primary_score() >= 0.8:
            print(f"Found high-quality solution at iteration {iteration}")
            break
    
    # Final statistics
    tree.print_statistics()
    
    return tree

#### Main Execution ####

def main():
    """Main execution function."""
    print("Loading dataset...")
    dataset = []
    with open(data_fpath, "r") as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    
    print(f"Creating {len(dataset)} SSDP trees...")
    problems = []
    for instance in dataset:
        question = instance["question"]
        answer = instance["answer"]
        problem = SSDPTree(question, answer)
        problems.append(problem)
    
    print("Starting SSDP search...")
    
    # Process problems (can be parallelized if needed)
    processed_problems = []
    for problem in tqdm(problems, desc="SSDP Search"):
        try:
            result = ssdp_worker(problem)
            processed_problems.append(result)
        except Exception as e:
            print(f"Error processing problem: {e}")
            processed_problems.append(problem)  # Keep original
    
    print(f"Saving results to {output_fpath}")
    pickle.dump(processed_problems, open(output_fpath, "wb"))
    
    # Print summary statistics
    print("\n=== SSDP Search Complete ===")
    total_nodes = sum(len(p.all_nodes) for p in processed_problems)
    total_expansions = sum(p.total_expansions for p in processed_problems)
    total_merges = sum(p.total_merges for p in processed_problems)
    total_prunes = sum(p.total_prunes for p in processed_problems)
    
    print(f"Total nodes created: {total_nodes}")
    print(f"Total expansions: {total_expansions}")
    print(f"Total merges: {total_merges}")
    print(f"Total prunes: {total_prunes}")
    print(f"Average nodes per problem: {total_nodes / len(processed_problems):.1f}")

if __name__ == "__main__":
    main() 