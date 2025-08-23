# this script is written in the early stage of this project. At that time, we use a slightly low-efficient implementation of multi-threading. Besides, we do not use the embedding model as a server.

import os
import sys
# from call_joint_service import call  # REMOVED - we'll use model-based approach
from .search_tree import *
import json
import pickle
import numpy as np
import jsonlines
import torch

LIMIT=50
BUDGET=10
TEMPERATURE=0.8

print("BUDGET:", BUDGET)

# task dependent
def assert_end(text):
    return True if text.strip().split("\n")[-1].startswith("The answer is") else False

def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = 0
    if state.content is not None and (len(state.content) == 0 or len(state.content) > 1024): # too short or too long
        state.value = 0

SEQ_STOP_TOKENS = []
STEP_STOP_TOKENS = ["\n"]

CONTINUE = False

# Remove hardcoded paths
# data_fpath = "xmu-nlp/Llama-3-8b-gsm8k"  # REMOVED
# output_fpath = f"test_gsm8k_bfs_b{BUDGET}_t{TEMPERATURE}.pkl"  # REMOVED
# policy_fpath = "path/to/llama/ckpt"  # REMOVED

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

def run_bfs_experiment(question, expected_answer, model, tokenizer, verifier_model=None, verifier_tokenizer=None):
    """Run BFS experiment and return results."""
    # Create BFS tree
    tree = Tree(question, expected_answer, additional_info={})
    tree.init_root_node(0)
    
    # Run BFS iterations
    for iteration in range(LIMIT):
        # Select best node to expand
        state = tree.select_best_node()
        if state is None:
            break
            
        # Expand node multiple times
        for _ in range(BUDGET):
            # Generate next step using model
            path = state.print_path()
            next_step = call_policy(question, path, model, tokenizer)
            
            # Create child node
            is_terminal = assert_end(next_step)
            child = tree.add_node(next_step, 0, state, iteration + 1, is_terminal)
            
            # Apply validation rules
            fix_value(child)
            
            # If we have a verifier, get the value
            if verifier_model and verifier_tokenizer:
                value = call_value(question, next_step, verifier_model, verifier_tokenizer)
                child.value = value
    
    # Get best answer
    best_node = tree.select_best_node()
    if best_node:
        final_answer = best_node.print_path()
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