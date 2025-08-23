import os
import json
import pickle
import numpy as np
import jsonlines
import requests
import torch
from tqdm import tqdm

LIMIT=50
BUDGET=5
BEAM=5
TEMPERATURE=0.8

# Remove hardcoded paths - these will be passed as parameters
# data_fpath = "../../dataset/toy.jsonl"  # REMOVED
# output_fpath = f"test_gsm8k_beamsearch_b{BUDGET}_t{TEMPERATURE}.pkl"  # REMOVED
# policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k"  # REMOVED

# task dependent
def assert_end(text, tokenizer):
    return True if "The answer is" in text and text.endswith(tokenizer.eos_token) else False

# Remove hardcoded tokenizer loading
# from transformers import AutoTokenizer  # REMOVED
# tokenizer = AutoTokenizer.from_pretrained(policy_fpath)  # REMOVED
MAX_LEN_PER_STEP = 256

def fix_value(state, tokenizer):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = -1
    if state.content is not None and (len(state.content) == 0 or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP): # too short or too long
        state.value = -1
    if state.content.endswith(tokenizer.eos_token) and not assert_end(state.content, tokenizer):
        state.value = -1
    return state

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
    
    # Generate with model - handle attention mask properly
    with torch.no_grad():
        try:
            # Try with attention mask first
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
            # Fallback without attention mask if there are issues
            print(f"Warning: Using fallback generation method: {e}")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new part
    if query in generated_text:
        new_text = generated_text[len(query):]
    else:
        new_text = generated_text
    
    return new_text.strip()

def call_value(question, path, verifier_model, verifier_tokenizer):
    """Call the verifier model to get value score."""
    # This would need to be updated to use the verifier model
    # For now, return a placeholder
    return 0.5

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

def run_beamsearch_experiment(question, expected_answer, model, tokenizer, verifier_model=None, verifier_tokenizer=None):
    """Run beam search experiment and return results."""
    # Create tree
    tree = Tree(question, expected_answer)
    
    # Run beam search iterations
    for iteration in range(LIMIT):
        # Get beam to expand
        beam = tree.get_beam_to_expand(BEAM)
        
        if not beam:
            break
            
        # Expand each node in beam
        for node in beam:
            # Generate next step
            path = node.print_path()
            next_step = call_policy(question, path, model, tokenizer)
            
            # Create new node
            is_terminal = assert_end(next_step, tokenizer)
            new_node = tree.add_node(next_step, 0, node, is_terminal)  # Value will be set by verifier
            
            # Apply validation rules
            fix_value(new_node, tokenizer)
            
            # If we have a verifier, get the value
            if verifier_model and verifier_tokenizer:
                value = call_value(question, next_step, verifier_model, verifier_tokenizer)
                new_node.value = value
    
    # Get best answer
    best_terminal = None
    best_value = -float('inf')
    
    for node in tree.all_nodes:
        if node.is_leaf and node.value > best_value:
            best_terminal = node
            best_value = node.value
    
    if best_terminal:
        final_answer = best_terminal.content
        answer_found = True
    else:
        final_answer = ""
        answer_found = False
    
    return {
        'final_answer': final_answer,
        'answer_found': answer_found,
        'search_steps': len(tree.all_nodes),
        'tokens_used': sum(len(tokenizer.tokenize(node.content)) for node in tree.all_nodes if node.content),
        'raw_tree': tree
    }

# Remove the old main execution code that used hardcoded paths
# if __name__ == '__main__':
#     # ... old code removed ...