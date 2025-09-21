import pickle
import sys
import re
import numpy as np

def extract_answer(text):
    # Robustly extract the answer, handling different formats
    # Look for "The answer is", then try to find a number
    match = re.search(r'The answer is (.*?)(?:\n|$)', text)
    if match:
        # Extract the first number found after "The answer is"
        num_match = re.search(r'-?\d+\.?\d*|-\.\d+', match.group(1))
        if num_match:
            return num_match.group(0)
    # Fallback: find the last number in the string
    all_numbers = re.findall(r'-?\d+\.?\d*|-\.\d+', text)
    if all_numbers:
        return all_numbers[-1]
    return None

def is_correct(generated_answer, true_answer):
    if generated_answer is None or true_answer is None:
        return False
    try:
        # Compare as floating point numbers
        return abs(float(generated_answer) - float(true_answer)) < 1e-3
    except (ValueError, TypeError):
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_search.py <path_to_pickle_file>")
        sys.exit(1)

    pickle_fpath = sys.argv[1]

    with open(pickle_fpath, 'rb') as f:
        data = pickle.load(f)

    problems = data['problems']
    metrics = data['metrics']
    correct_count = 0
    total_count = len(problems)
    runtimes = [p.runtime_seconds for p in problems]

    for tree in problems:
        true_answer_text = extract_answer(tree.answer)
        
        best_node = None
        if tree.terminal_nodes:
            best_node = max(tree.terminal_nodes, key=lambda node: node.score)
        
        if best_node:
            generated_answer_text = extract_answer(best_node.print_path())
            if is_correct(generated_answer_text, true_answer_text):
                correct_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n=== Evaluation Results ===")
    print(f"Total problems: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    print(f"\n=== Performance Metrics ===")
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
    
    print(f"\nPolicy Server Tokens:")
    policy_metrics = metrics['policy_server']
    print(f"  Total: {policy_metrics['total_tokens']}")
    print(f"  Prompt: {policy_metrics['total_prompt_tokens']}, Completion: {policy_metrics['total_completion_tokens']}")

    print(f"\nEmbedding Server Tokens:")
    embedding_metrics = metrics['embedding_server']
    print(f"  Total: {embedding_metrics['total_tokens']}")
    print(f"  Prompt: {embedding_metrics['total_prompt_tokens']}, Completion: {embedding_metrics['total_completion_tokens']}")

    print(f"\nCombined Total Tokens: {metrics['combined']['total_tokens']}")
