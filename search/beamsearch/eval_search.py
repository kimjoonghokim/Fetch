import pickle
import re
import sys
import numpy as np

class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

class VirtualNode:
    def __init__(self, nodes, parent=None):
        self.nodes = nodes
        self.tree = None
        self.value = 0
        self.visited = False
        self.children = []
        self.cache = []
        self.parent = parent
        self.is_leaf = False
        self.timestep = 0

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
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

def extract_gold_answer(completion):
    """Extracts the gold answer from a string."""
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        try:
            float(match_str)
            return match_str
        except ValueError:
            return "[INVALID]"
    return "[INVALID]"

def extract_pred_answer(text):
    """Extracts the predicted answer from the model's output."""
    if text is None:
        return "[INVALID]"
    text = text.replace("<|end_of_text|>", "").strip()
    PATTERN = re.compile(r"The answer is (.*)")
    match = PATTERN.search(text)
    if match:
        # Handle potential dollar signs and commas in the number
        return match.group(1).replace(",", "").replace("$", "").strip()
    # Fallback for numbers at the end of the string
    numbers = re.findall(r"([+-]?\d*\.?\d+)", text)
    if numbers:
        return numbers[-1]
    return "[INVALID]"

def are_answers_equal(pred, gold):
    """Compares two answers for equality, handling floating point comparisons."""
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return False

def main(results_file_path):
    """Main function to load results and print evaluation metrics."""
    print(f"--- Loading results from: {results_file_path} ---")
    try:
        with open(results_file_path, "rb") as f:
            data = pickle.load(f)
            problems = data['problems']
            metrics = data['metrics']
    except (IOError, pickle.UnpicklingError, KeyError) as e:
        print(f"Error: Could not load or parse the results file. Make sure it's a valid pickle file. Details: {e}")
        return

    total_problems = len(problems)
    if total_problems == 0:
        print("No problems found in the results file.")
        return

    correct_solutions = 0
    unfinished_problems = 0
    runtimes = [p.runtime_seconds for p in problems]

    for problem in problems:
        gold_answer = extract_gold_answer(problem.answer)
        
        best_node = None
        if hasattr(problem, 'virtual_nodes'):
            # It's from beamsearch_merge.py
            leaf_nodes = [node for node in problem.virtual_nodes if node.is_leaf]
            if leaf_nodes:
                best_virtual_node = max(leaf_nodes, key=lambda x: x.value)
                if best_virtual_node.nodes:
                    best_node = best_virtual_node.nodes[0]
        else:
            # It's from beamsearch.py
            leaf_nodes = [node for node in problem.all_nodes if node.is_leaf]
            if leaf_nodes:
                best_node = max(leaf_nodes, key=lambda x: x.value)

        if best_node:
            predicted_answer = extract_pred_answer(best_node.content)
            if are_answers_equal(predicted_answer, gold_answer):
                correct_solutions += 1
        else:
            unfinished_problems += 1

    # --- Print Human-Readable Results ---
    print("\n" + "="*50)
    print("         Beam Search Evaluation         ")
    print("="*50 + "\n")

    # Accuracy Metrics
    accuracy = (correct_solutions / total_problems) * 100 if total_problems > 0 else 0
    completion_rate = ((total_problems - unfinished_problems) / total_problems) * 100 if total_problems > 0 else 0
    
    print("--- Performance ---")
    print(f"Total Problems:      {total_problems}")
    print(f"Correct Solutions:   {correct_solutions}")
    print(f"Unfinished Problems: {unfinished_problems}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print(f"Completion Rate:     {completion_rate:.2f}%")
    print("-" * 50)

    # Runtime Metrics
    total_runtime = metrics.get('total_runtime', 0)
    avg_runtime = np.mean(runtimes) if runtimes else 0
    min_runtime = min(runtimes) if runtimes else 0
    max_runtime = max(runtimes) if runtimes else 0
    median_runtime = np.median(runtimes) if runtimes else 0
    
    print("--- Runtime ---")
    print(f"Total Runtime:       {total_runtime:.2f} seconds")
    print(f"Average per Problem: {avg_runtime:.2f} seconds")
    print(f"Min per Problem:     {min_runtime:.2f} seconds")
    print(f"Max per Problem:     {max_runtime:.2f} seconds")
    print(f"Median per Problem:  {median_runtime:.2f} seconds")
    print("-" * 50)

    # Token Usage Metrics
    if 'policy_server' in metrics and 'verifier_server' in metrics:
        policy_metrics = metrics['policy_server']
        verifier_metrics = metrics['verifier_server']
        embedding_metrics = metrics.get('embedding_server', {'total_tokens': 0, 'total_prompt_tokens': 0, 'total_completion_tokens': 0})
        combined_metrics = metrics.get('combined', {})
        
        policy_tokens = policy_metrics.get('total_tokens', 0)
        policy_prompt = policy_metrics.get('total_prompt_tokens', 0)
        policy_completion = policy_metrics.get('total_completion_tokens', 0)
        
        verifier_tokens = verifier_metrics.get('total_tokens', 0)
        verifier_prompt = verifier_metrics.get('total_prompt_tokens', 0)
        verifier_completion = verifier_metrics.get('total_completion_tokens', 0)

        embedding_tokens = embedding_metrics.get('total_tokens', 0)
        embedding_prompt = embedding_metrics.get('total_prompt_tokens', 0)
        embedding_completion = embedding_metrics.get('total_completion_tokens', 0)
        
        total_tokens = combined_metrics.get('total_tokens', policy_tokens + verifier_tokens + embedding_tokens)
        verifier_percentage = combined_metrics.get('verifier_percentage', 0)
        embedding_percentage = combined_metrics.get('embedding_percentage', 0)

        print("--- Token Usage ---")
        print(f"Total Tokens Used:   {total_tokens}")
        print(f"")
        print(f"Policy Server:")
        print(f"  - Total Tokens:    {policy_tokens}")
        print(f"  - Prompt Tokens:   {policy_prompt}")
        print(f"  - Completion Tokens: {policy_completion}")
        print(f"")
        print(f"Verifier Server:")
        print(f"  - Total Tokens:    {verifier_tokens}")
        print(f"  - Prompt Tokens:   {verifier_prompt}")
        print(f"  - Completion Tokens: {verifier_completion}")
        print(f"")
        print(f"Embedding Server:")
        print(f"  - Total Tokens:    {embedding_tokens}")
        print(f"  - Prompt Tokens:   {embedding_prompt}")
        print(f"  - Completion Tokens: {embedding_completion}")
        print(f"")
        print(f"Verifier Contribution: {verifier_percentage:.1f}%")
        if embedding_tokens > 0:
            print(f"Embedding Contribution: {embedding_percentage:.1f}%")
        print(f"Average per Problem: {total_tokens / total_problems:.2f} tokens")
        
    else:
        # Fallback for old format
        total_tokens = metrics.get('total_tokens', 0)
        prompt_tokens = metrics.get('total_prompt_tokens', 0)
        completion_tokens = metrics.get('total_completion_tokens', 0)
        avg_tokens = total_tokens / total_problems if total_problems > 0 else 0

        print("--- Token Usage (Legacy Format) ---")
        print(f"Total Tokens Used:   {total_tokens}")
        print(f"  - Prompt Tokens:   {prompt_tokens}")
        print(f"  - Completion Tokens: {completion_tokens}")
        print(f"Average per Problem: {avg_tokens:.2f} tokens")
    
    print("="*50)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python eval_search.py <path_to_results.pkl>")
        sys.exit(1)
    main(sys.argv[1])
