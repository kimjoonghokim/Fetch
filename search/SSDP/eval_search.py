import pickle
import sys
import re
import numpy as np
import os

# Class definitions from ssdp.py
class Node:
    def __init__(self, content, confidence, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.confidence = confidence
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf
        self.embedding = None
        self.similarity_bonus = 0
        self.diversity_reward = 0
        self.parent_score = parent.score if parent else 0
        self.score = 0

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

    def update_score(self):
        self.score = self.confidence + self.similarity_bonus + self.diversity_reward + self.parent_score

class Cluster:
    def __init__(self, nodes):
        self.nodes = sorted(nodes, key=lambda x: x.confidence, reverse=True)
        self.representative = self.nodes[0]
        self.similarity_bonus = sum(n.confidence for n in self.nodes[1:])
        self.representative.similarity_bonus = self.similarity_bonus
        self.representative.update_score()

    @property
    def score(self):
        return self.representative.score

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.root = Node(None, 1.0, None, 0, self)
        self.root.update_score()
        self.all_nodes = [self.root]
        self.terminal_nodes = []
        self.beam = [self.root]
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.embedding_prompt_tokens = 0
        self.embedding_completion_tokens = 0
        self.embedding_total_tokens = 0
        self.runtime_seconds = 0.0

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def extract_answer(text):
    if text is None:
        return None
    match = re.search(r'The answer is (.*?)(?:\n|$)', text)
    if match:
        num_match = re.search(r'-?\d+\.?\d*|-\.\d+', match.group(1))
        if num_match:
            return num_match.group(0)
    all_numbers = re.findall(r'-?\d+\.?\d*|-\.\d+', text)
    if all_numbers:
        return all_numbers[-1]
    return None

def is_correct(generated_answer, true_answer):
    if generated_answer is None or true_answer is None:
        return False
    try:
        return abs(float(generated_answer) - float(true_answer)) < 1e-3
    except (ValueError, TypeError):
        return False

def main(results_file_path):
    log_file_path = os.path.splitext(results_file_path)[0] + '.log'
    original_stdout = sys.stdout
    
    with open(log_file_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)

        print(f"--- Loading results from: {results_file_path} ---")
        try:
            with open(results_file_path, "rb") as f:
                data = pickle.load(f)
                problems = data['problems']
                metrics = data['metrics']
        except (IOError, pickle.UnpicklingError, KeyError) as e:
            print(f"Error: Could not load or parse the results file. Details: {e}")
            sys.stdout = original_stdout
            return

        total_problems = len(problems)
        if total_problems == 0:
            print("No problems found in the results file.")
            sys.stdout = original_stdout
            return

        correct_solutions = 0
        unfinished_problems = 0
        runtimes = [p.runtime_seconds for p in problems]

        for problem in problems:
            true_answer_text = extract_answer(problem.answer)
            
            best_node = None
            if problem.terminal_nodes:
                best_node = max(problem.terminal_nodes, key=lambda node: node.score)
            
            if best_node:
                generated_answer_text = extract_answer(best_node.print_path())
                if is_correct(generated_answer_text, true_answer_text):
                    correct_solutions += 1
            else:
                unfinished_problems += 1

        print("\n" + "="*50 + "\n           SSDP Evaluation         \n" + "="*50 + "\n")

        accuracy = (correct_solutions / total_problems) * 100 if total_problems > 0 else 0
        completion_rate = ((total_problems - unfinished_problems) / total_problems) * 100 if total_problems > 0 else 0
        
        print("--- Performance ---")
        print(f"Total Problems:      {total_problems}")
        print(f"Correct Solutions:   {correct_solutions}")
        print(f"Unfinished Problems: {unfinished_problems}")
        print(f"Accuracy:            {accuracy:.2f}%")
        print(f"Completion Rate:     {completion_rate:.2f}%")
        print("-"*50)

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
        print("-"*50)

        policy_metrics = metrics.get('policy_server', {})
        embedding_metrics = metrics.get('embedding_server', {})
        combined_metrics = metrics.get('combined', {})

        policy_tokens = policy_metrics.get('total_tokens', 0)
        policy_prompt = policy_metrics.get('total_prompt_tokens', 0)
        policy_completion = policy_metrics.get('total_completion_tokens', 0)

        embedding_tokens = embedding_metrics.get('total_tokens', 0)
        embedding_prompt = embedding_metrics.get('total_prompt_tokens', 0)
        embedding_completion = embedding_metrics.get('total_completion_tokens', 0)

        total_tokens = combined_metrics.get('total_tokens', policy_tokens + embedding_tokens)
        embedding_percentage = (embedding_tokens / total_tokens * 100) if total_tokens > 0 else 0

        print("--- Token Usage ---")
        print(f"Total Tokens Used:   {total_tokens}")
        print(f"")
        print(f"Policy Server:")
        print(f"  - Total Tokens:    {policy_tokens}")
        print(f"  - Prompt Tokens:   {policy_prompt}")
        print(f"  - Completion Tokens: {policy_completion}")
        print(f"")
        print(f"Embedding Server:")
        print(f"  - Total Tokens:    {embedding_tokens}")
        print(f"  - Prompt Tokens:   {embedding_prompt}")
        print(f"  - Completion Tokens: {embedding_completion}")
        print(f"")
        print(f"Embedding Contribution: {embedding_percentage:.1f}%")
        print(f"Average per Problem: {total_tokens / total_problems:.2f} tokens")
        print("="*50)

    sys.stdout = original_stdout
    print(f"Evaluation results saved to {log_file_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python eval_search.py <path_to_results.pkl>")
        sys.exit(1)
    main(sys.argv[1])
