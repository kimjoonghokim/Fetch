import os
import sys
from gsm_config import *
from mcts_tree import *
import multiprocessing
import json
import pickle
import numpy as np
import math
import copy
import torch

CONTINUE = False

# Remove hardcoded paths
# data_fpath = "gsm8k/test.json"  # REMOVED
# output_fpath = f"test_mcts_rb{config.root_budget}_nb{config.node_budget}_c{config.c}_r{config.n_rollouts}_md{config.max_depth}_mt{config.min_terminals}_d{config.d}.pkl"  # REMOVED

def call_policy(question, path, model, tokenizer, temperature=0.8):
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

def run_mcts_experiment(question, expected_answer, model, tokenizer, config=None, verifier_model=None, verifier_tokenizer=None):
    """Run MCTS experiment and return results."""
    # Create MCTS tree
    if config is None:
        config = GSMConfig()  # Use default config
    
    tree = MCTSTree(question, expected_answer, config)
    
    # Run MCTS
    tree.run_mcts()
    
    # Get best answer
    best_terminal = tree.get_best_terminal()
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