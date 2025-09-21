import pickle
import sys
import re
import numpy as np

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

def extract_answer(text):
    # Robustly extract the answer, handling different formats
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