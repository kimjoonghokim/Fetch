import os
import json
import pickle
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Task dependent functions
def assert_end(text, tokenizer):
    """Check if the reasoning has reached a conclusion."""
    return "The answer is" in text

def call_policy(question, path, model, tokenizer, temperature=TEMPERATURE):
    """Call the policy model to generate next step."""
    # Use the passed model instead of HTTP calls
    query = f"Question: {question}\nAnswer:{path}"
    
    # Tokenize input with explicit attention mask
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Ensure attention mask is properly set
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with model
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error in model generation: {e}")
            return ""
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new part
    if query in generated_text:
        new_text = generated_text[len(query):]
    else:
        new_text = generated_text
    
    return new_text.strip()

#### SSDP Search Tree ####

class SSDPNode:
    """
    Enhanced node for SSDP algorithm with comprehensive scoring.
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
        """Calculate scores using simplified scoring system."""
        if self.content is not None:
            # Simplified scoring based on content length and keywords
            content_length = len(self.content) if self.content else 0
            
            # Basic heuristic scoring
            if "The answer is" in self.content:
                self.overall_score = 0.8  # High score for answers
            elif content_length > 50:
                self.overall_score = 0.6  # Medium score for longer content
            elif content_length > 10:
                self.overall_score = 0.4  # Lower score for short content
            else:
                self.overall_score = 0.2  # Very low score for minimal content
                
            # Add some randomness to avoid ties
            import random
            self.overall_score += random.uniform(-0.1, 0.1)
            self.overall_score = max(0.0, min(1.0, self.overall_score))
            
            self.confidence_score = self.overall_score
            self.detailed_scores = {
                'overall_score': self.overall_score,
                'component_scores': {'confidence': {'score': self.confidence_score}},
                'weights_used': {'confidence': 1.0}
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
    SSDP Tree with semantic similarity-based dynamic pruning.
    """
    
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.pruned_nodes = []
        # Start with root having None content (like the old working code)
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
        
        # Filter out root node (timestep 0) to prevent it from being expanded
        return [node for node in self.all_nodes 
                if node.timestep == timestep and not node.pruned and not node.is_leaf and node.timestep > 0]
    
    def get_nodes_to_expand(self, max_paths=MAX_PARALLEL_PATHS):
        """Get the best nodes to expand based on overall scores."""
        current_level_nodes = self.get_active_nodes_at_level()
        
        if not current_level_nodes:
            # Look at previous level if current is empty
            prev_timestep = self.return_timestep() - 1
            if prev_timestep >= 0:
                current_level_nodes = self.get_active_nodes_at_level(prev_timestep)
            
            # If still no nodes, check if we can expand from root (timestep 0)
            if not current_level_nodes and self.root.content is not None:
                current_level_nodes = [self.root]
        
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

def fix_value(node, tokenizer):
    """Apply validation and fixing rules to node values."""
    if node.parent is not None:
        if node.parent.content == node.content:
            node.overall_score = 0.0
            node.confidence_score = 0.0
    
    if (node.content is not None and 
        (len(node.content) == 0 or len(node.content) > 1024)):
        node.overall_score = 0.0
        node.confidence_score = 0.0
    
    if (node.content and 
        not assert_end(node.content, tokenizer)):
        node.overall_score = 0.0
        node.confidence_score = 0.0
    
    return node

#### Main SSDP Algorithm ####

def ssdp_worker(tree, model, tokenizer):
    """
    SSDP worker function implementing the main search algorithm.
    """
    question = tree.question
    iteration = 0
    
    print(f"Starting SSDP for question: {question[:50]}...")
    
    # Handle initial state - generate first step for root if it has no content
    if tree.root.content is None:
        print("Generating initial reasoning step...")
        try:
            initial_step = call_policy(question, "", model, tokenizer)
            if initial_step and initial_step.strip():
                tree.root.content = initial_step
                print(f"Initial step: {initial_step[:100]}...")
            else:
                print("Failed to generate initial step, using placeholder")
                tree.root.content = "Let me think about this step by step."
        except Exception as e:
            print(f"Error generating initial step: {e}")
            tree.root.content = "Let me think about this step by step."
    
    while iteration < LIMIT:
        # Get nodes to expand
        nodes_to_expand = tree.get_nodes_to_expand(MAX_PARALLEL_PATHS)
        
        print(f"Iteration {iteration}: Found {len(nodes_to_expand)} nodes to expand")
        
        if not nodes_to_expand:
            print(f"No nodes to expand at iteration {iteration}")
            print("Available nodes:", [(n.content[:30] if n.content else "None", n.timestep, n.is_leaf, n.pruned) for n in tree.all_nodes if not n.pruned])
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
                    next_step = call_policy(question, path, model, tokenizer)
                    
                    # Create new node
                    is_terminal = assert_end(next_step, tokenizer)
                    new_node = tree.add_node(next_step, node, is_terminal)
                    
                    # Apply validation rules
                    fix_value(new_node, tokenizer)
                    
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

def run_ssdp_experiment(question, expected_answer, model, tokenizer):
    """Run SSDP search algorithm."""
    # Create SSDP tree
    tree = SSDPTree(question, expected_answer)
    
    # Run SSDP worker
    result_tree = ssdp_worker(tree, model, tokenizer)
    
    # Extract best answer
    best_terminal = result_tree.get_best_terminal_node()
    if best_terminal:
        final_answer = best_terminal.content
        answer_found = True
    else:
        final_answer = ""
        answer_found = False
    
    return {
        'final_answer': final_answer,
        'answer_found': answer_found,
        'search_steps': len(result_tree.all_nodes),
        'tokens_used': sum(len(tokenizer.tokenize(node.content)) for node in result_tree.all_nodes if node.content),
        'raw_tree': result_tree
    } 