import pickle
import re
import sys

# The search_tree module must be in the path for pickle to unpickle the Tree and Node objects.
# We assume it's in the same directory.
from search_tree import Tree, Node

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
        print(f"Error: Could not load or parse the results file. Make sure it's a valid pickle file in the new format. Details: {e}")
        return

    total_problems = len(problems)
    if total_problems == 0:
        print("No problems found in the results file.")
        return

    correct_solutions = 0
    unfinished_problems = 0

    # Detect if this is a merge file or regular BFS
    is_merge = hasattr(problems[0], 'virtual_nodes') if problems else False

    for problem in problems:
        gold_answer = extract_gold_answer(problem.answer)
        
        best_node = None
        if is_merge:
            # It's from bfs_merge.py - check virtual_nodes
            leaf_nodes = [node for node in problem.virtual_nodes if node.is_leaf]
            if leaf_nodes:
                best_virtual_node = max(leaf_nodes, key=lambda x: x.value)
                if best_virtual_node.nodes:
                    best_node = best_virtual_node.nodes[0]
        else:
            # It's from bfs.py - check all_nodes
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
    print("\n" + "="*30)
    if is_merge:
        print("    BFS Merge Evaluation     ")
    else:
        print("    BFS Search Evaluation     ")
    print("="*30 + "\n")

    # Accuracy Metrics
    accuracy = (correct_solutions / total_problems) * 100 if total_problems > 0 else 0
    completion_rate = ((total_problems - unfinished_problems) / total_problems) * 100 if total_problems > 0 else 0
    
    print("--- Performance ---")
    print(f"Total Problems:      {total_problems}")
    print(f"Correct Solutions:   {correct_solutions}")
    print(f"Unfinished Problems: {unfinished_problems}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print(f"Completion Rate:     {completion_rate:.2f}%")
    print("-" * 30)

    # Runtime Metrics
    total_runtime = metrics.get('total_runtime', 0)
    avg_runtime = total_runtime / total_problems if total_problems > 0 else 0
    
    print("--- Runtime ---")
    print(f"Total Runtime:       {total_runtime:.2f} seconds")
    print(f"Average per Problem: {avg_runtime:.2f} seconds")
    print("-" * 30)

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
        if is_merge:
            print(f"")
            print(f"Embedding Server:")
            print(f"  - Total Tokens:    {embedding_tokens}")
            print(f"  - Prompt Tokens:   {embedding_prompt}")
            print(f"  - Completion Tokens: {embedding_completion}")
        print(f"")
        print(f"Verifier Contribution: {verifier_percentage:.1f}%")
        if is_merge and embedding_tokens > 0:
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
    
    print("="*30)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python eval_search.py <path_to_results.pkl>")
        sys.exit(1)
    main(sys.argv[1])