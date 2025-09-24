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

LIMIT=50
BUDGET=5
BEAM=5
TEMPERATURE=0.8
load_dotenv(dotenv_path='../experiments_config.env')
data_fpath_var = os.getenv("PATH_TO_DATASET")
data_fpath = os.getenv(data_fpath_var) if data_fpath_var else None # path to the test set
if data_fpath:
    dataset_type = os.path.basename(data_fpath).split('.')[0]
    dataset_name = os.path.basename(os.path.dirname(data_fpath))
else:
    dataset_name = "unknown"
    dataset_type = "unknown"
output_fpath = f"{dataset_type}_{dataset_name}_beamsearch_b{BUDGET}_t{TEMPERATURE}.pkl"
load_dotenv(dotenv_path='../../server_config.env')
policy_fpath = os.getenv("POLICY_MODEL_PATH") # path to the policy model

# task dependent
def assert_end(text):
    return True if "The answer is" in text else False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)

# Model family detection for tokenizer-specific handling
def get_model_family(model_path):
    """Detect model family from the model path"""
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower:
        return "qwen"
    elif "llama" in model_path_lower:
        return "llama"
    elif "gemma" in model_path_lower:
        return "gemma"
    else:
        return "unknown"

model_family = get_model_family(policy_fpath)
print(f"Detected model family: {model_family} for model: {policy_fpath}")

# Model-specific token IDs and settings
def get_model_specific_config(model_family, tokenizer):
    """Get model-specific configuration"""
    if model_family == "qwen":
        # Qwen-specific token IDs - these may need adjustment based on the specific Qwen model
        # Common Qwen patterns: Answer: token followed by newline
        split_token_ids = tokenizer.encode("Answer:", add_special_tokens=False)
        if len(split_token_ids) > 0:
            # Add newline token ID if not already included
            newline_id = tokenizer.encode("\n", add_special_tokens=False)
            if len(newline_id) > 0:
                split_token_ids.extend(newline_id)
        return {
            "split_token_ids": split_token_ids,
            "end_pattern": "ĊĊ",  # Qwen newline pattern
            "eos_handling": "standard"
        }
    elif model_family == "llama":
        # Llama-specific configuration
        return {
            "split_token_ids": [16533, 25],  # Original Llama tokens
            "end_pattern": "ĊĊ",  # Llama newline pattern
            "eos_handling": "standard"
        }
    else:
        # Default/fallback configuration
        return {
            "split_token_ids": tokenizer.encode("Answer:", add_special_tokens=False),
            "end_pattern": "ĊĊ",
            "eos_handling": "standard"
        }

model_config = get_model_specific_config(model_family, tokenizer)
print(f"Using model config: {model_config}")

MAX_LEN_PER_STEP = 256
def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = -1
    if state.content is not None and (len(state.content) == 0 or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP): # too short or too long
        state.value = -1
    # Model-specific EOS token handling
    if model_config["eos_handling"] == "standard":
        if state.content and state.content.endswith(tokenizer.eos_token) and not assert_end(state.content):
            state.value = -1
    return state

def call_policy(question, path):
    url = "http://127.0.0.1:8000/v1/completions"
    model = policy_fpath
    query = f"Question: {question}\nAnswer:{path}"
    pload ={"prompt": query, "model": model, "temperature": TEMPERATURE, "max_tokens": 512, 
            "stop": ["\n"], "include_stop_str_in_output": True, "skip_special_tokens": False}
    
    try:
        response = requests.post(url, json=pload)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        
        # Debug: print the response structure (commented out for cleaner output)
        # print(f"API Response keys: {list(response_json.keys())}")
        
        # Check if response contains expected fields
        if "choices" not in response_json:
            print(f"Unexpected API response format: {response_json}")
            # Try to handle different response formats
            if "text" in response_json:
                return response_json["text"], response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            elif "output" in response_json:
                return response_json["output"], response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            else:
                raise KeyError(f"API response missing 'choices' field: {response_json}")
        
        choice = response_json["choices"][0]
        usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return choice["text"], usage
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to policy server at {url}")
        print("Make sure the policy server is running on port 8000")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from policy server: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise
    except KeyError as e:
        print(f"KeyError in API response: {e}")
        print(f"Full response: {response_json if 'response_json' in locals() else 'No response JSON'}")
        raise
    except Exception as e:
        print(f"Unexpected error in call_policy: {e}")
        raise

def call_value(question, path):
    url = "http://127.0.0.1:8002/predict"
    query = f"Question: {question}\nAnswer:{path}"
    # Model-specific EOS token handling
    if model_config["eos_handling"] == "standard":
        if query.endswith(tokenizer.eos_token):
            query = query[:-len(tokenizer.eos_token)] # this value is not trained like this
    pload ={"texts": [query]}
    
    try:
        response = requests.post(url, json=pload)
        response.raise_for_status()
        response_json = response.json()
        
        # Debug: print the response structure (commented out for cleaner output)
        # print(f"Verifier API Response keys: {list(response_json.keys())}")
        
        if "values" not in response_json:
            print(f"Unexpected verifier API response format: {response_json}")
            raise KeyError(f"Verifier API response missing 'values' field: {response_json}")
        
        value = (min(max(response_json["values"][0], -1.), 1.) + 1.) / 2
        usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return value, usage
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to verifier server at {url}")
        print("Make sure the verifier server is running on port 8002")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from verifier server: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise
    except KeyError as e:
        print(f"KeyError in verifier API response: {e}")
        print(f"Full response: {response_json if 'response_json' in locals() else 'No response JSON'}")
        raise
    except Exception as e:
        print(f"Unexpected error in call_value: {e}")
        raise

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
        # Policy server token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # Verifier server token tracking
        self.verifier_prompt_tokens = 0
        self.verifier_completion_tokens = 0
        self.verifier_total_tokens = 0
        self.runtime_seconds = 0.0

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
    start_time = time.time()
    question = tree.question
    for _ in range(LIMIT):
        actions = tree.get_beam_to_expand(BEAM)
        if actions:
            for action in actions:
                for _ in range(BUDGET):
                    # expand this state
                    # get next step content
                    path = action.print_path()
                    next_step, usage = call_policy(question, path)
                    tree.prompt_tokens += usage.get("prompt_tokens", 0)
                    tree.completion_tokens += usage.get("completion_tokens", 0)
                    tree.total_tokens += usage.get("total_tokens", 0)
                    # get next step value
                    next_value, verifier_usage = call_value(question, path + next_step)
                    tree.verifier_prompt_tokens += verifier_usage.get("prompt_tokens", 0)
                    tree.verifier_completion_tokens += verifier_usage.get("completion_tokens", 0)
                    tree.verifier_total_tokens += verifier_usage.get("total_tokens", 0)
                    state = tree.add_node(next_step, next_value, action, assert_end(next_step))
                    fix_value(state)
                    # print((next_step, next_value))
        else:
            break
    tree.runtime_seconds = time.time() - start_time
    return tree

start_time = time.time()

print(f"\n🚀 Starting beamsearch with {len(problems)} problems...")
print(f"📊 Using {model_family} model: {policy_fpath}")
print(f"🔧 Beam size: {BEAM}, Budget: {BUDGET}, Temperature: {TEMPERATURE}")
print("=" * 60)

pool = multiprocessing.Pool(80)
problems = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems), desc="Processing problems", unit="problem"))    
pool.close()

print("=" * 60)
print("✅ Beamsearch completed!")

total_runtime = time.time() - start_time

# Policy server token totals
total_prompt_tokens = sum([p.prompt_tokens for p in problems])
total_completion_tokens = sum([p.completion_tokens for p in problems])
total_tokens = sum([p.total_tokens for p in problems])

# Verifier server token totals
total_verifier_prompt_tokens = sum([p.verifier_prompt_tokens for p in problems])
total_verifier_completion_tokens = sum([p.verifier_completion_tokens for p in problems])
total_verifier_tokens = sum([p.verifier_total_tokens for p in problems])

# Combined totals
total_all_tokens = total_tokens + total_verifier_tokens

final_data = {
    'problems': problems,
    'metrics': {
        'total_runtime': total_runtime,
        'policy_server': {
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens
        },
        'verifier_server': {
            'total_prompt_tokens': total_verifier_prompt_tokens,
            'total_completion_tokens': total_verifier_completion_tokens,
            'total_tokens': total_verifier_tokens
        },
        'combined': {
            'total_tokens': total_all_tokens,
            'verifier_percentage': (total_verifier_tokens / total_all_tokens * 100) if total_all_tokens > 0 else 0
        }
    }
}

with open(output_fpath, "wb") as f:
    pickle.dump(final_data, f)

print("=== Beam Search Complete ===")
print(f"Total runtime: {total_runtime:.2f} seconds")
print(f"\nPolicy Server Tokens:")
print(f"  Total: {total_tokens}")
print(f"  Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}")
print(f"\nVerifier Server Tokens:")
print(f"  Total: {total_verifier_tokens}")
print(f"  Prompt: {total_verifier_prompt_tokens}, Completion: {total_verifier_completion_tokens}")
print(f"\nCombined Total: {total_all_tokens}")
print(f"Verifier contribution: {total_verifier_tokens / total_all_tokens * 100:.1f}%")
print(f"Results saved to {output_fpath}")

print(f"\nRunning evaluation script on {output_fpath}...")
subprocess.run(["python", "eval_search.py", output_fpath])