import pickle
import re
import sys
import argparse

# The search_tree module must be in the path for pickle to unpickle the Tree and Node objects.
from search_tree import Tree, Node

def extract_gold_answer(completion):
    """Extracts the gold answer from a string which has a '####' marker."""
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
    """Extracts the predicted answer from the model's full solution text."""
    if text is None:
        return "[INVALID]"
    
    # Primary pattern for "The answer is X"
    PATTERN = re.compile(r"The answer is (.*)")
    match = PATTERN.search(text)
    if match:
        # Handle potential dollar signs and commas in the number
        return match.group(1).replace(",", "").replace("$", "").strip()
        
    # Fallback for numbers at the end of the string if the main pattern fails
    numbers = re.findall(r"([+-]?\d*\.?\d+)", text)
    if numbers:
        return numbers[-1]
        
    return "[INVALID]"

def are_answers_equal(pred, gold):
    """Compares two answers for equality, handling floating point comparisons."""
    if pred == "[INVALID]" or gold == "[INVALID]":
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return False

def main():
    """Main function to load results and print evaluation metrics."""
    parser = argparse.ArgumentParser(description="Evaluate BFS search results from a .pkl file.")
    parser.add_argument("input_path", type=str, help="Path to the results .pkl file.")
    args = parser.parse_args()

    print(f"--- Loading results from: {args.input_path} ---")
    try:
        with open(args.input_path, "rb") as f:
            data = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        print(f"Error: Could not load or parse the results file. Details: {e}")
        sys.exit(1)

    # Handle both new dictionary format and old list format for backward compatibility
    if isinstance(data, dict):
        problems = data.get('problems', [])
        metrics = data.get('metrics', {})
    elif isinstance(data, list):
        problems = data
        metrics = {} # No metrics in old format
    else:
        print("Error: Unrecognized data format in pickle file.")
        sys.exit(1)

    total_problems = len(problems)
    if total_problems == 0:
        print("No problems found in the results file.")
        return

    correct_solutions = 0
    
    for problem in problems:
        gold_answer = extract_gold_answer(problem.answer)
        
        # This is the compatible way to get the best solution from the BFS Tree
        best_node = problem.get_best_solution()
        
        predicted_answer = "[INVALID]"
        if best_node:
            # The full solution path is needed to find "The answer is..."
            solution_text = best_node.print_path() 
            predicted_answer = extract_pred_answer(solution_text)

        if are_answers_equal(predicted_answer, gold_answer):
            correct_solutions += 1

    # --- Print Human-Readable Results ---
    print("\n" + "="*30)
    print("    BFS Search Evaluation     ")
    print("="*30 + "\n")

    accuracy = (correct_solutions / total_problems) * 100 if total_problems > 0 else 0
    
    print("--- Performance ---")
    print(f"Total Problems:      {total_problems}")
    print(f"Correct Solutions:   {correct_solutions}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print("-"*30)

    if metrics:
        total_runtime = metrics.get('total_runtime', 0)
        avg_runtime = total_runtime / total_problems if total_problems > 0 else 0
        
        print("--- Runtime ---")
        print(f"Total Runtime:       {total_runtime:.2f} seconds")
        print(f"Average per Problem: {avg_runtime:.2f} seconds")
        print("-"*30)

        total_tokens = metrics.get('total_tokens', 0)
        prompt_tokens = metrics.get('total_prompt_tokens', 0)
        completion_tokens = metrics.get('total_completion_tokens', 0)
        avg_tokens = total_tokens / total_problems if total_problems > 0 else 0

        print("--- Token Usage ---")
        print(f"Total Tokens Used:   {total_tokens}")
        print(f"  - Prompt Tokens:   {prompt_tokens}")
        print(f"  - Completion Tokens: {completion_tokens}")
        print(f"Average per Problem: {avg_tokens:.2f} tokens")
        print("="*30)
    else:
        print("No runtime or token metrics found in file.")


if __name__ == '__main__':
    main()