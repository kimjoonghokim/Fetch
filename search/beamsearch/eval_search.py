import pickle
import re
from transformers import AutoTokenizer

data_fpath = "path/to/output"
policy_fpath = "path/to/policy"

class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        self.all_nodes.append(self.root)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.runtime_seconds = 0.0

with open(data_fpath, "rb") as f:
    problems = pickle.load(f)

def extract_gold_answer(completion):
    ANS_RE = re.compile(r"#### (-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return "[INVALID]"
        return match_str
    else:
        return "[INVALID]"

def extract_pred_answer(text):
    PATTERN = re.compile(r"The answer is (.*)<|endoftext|>")
    match = PATTERN.search(text)
    if match is not None:
        return match.group(1).replace(",", "")
    else:
        return "[INVALID]"

def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except:
        return False

tokenizer = AutoTokenizer.from_pretrained(policy_fpath)

corr, total = 0, 0
not_finished = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0
total_runtime = 0.0

for problem in problems:
    ref = extract_gold_answer(problem.answer) if "####" in problem.answer else problem.answer
    leaf_nodes = [node for node in problem.all_nodes if node.is_leaf]
    if leaf_nodes:
        best_node = max(leaf_nodes, key=lambda x: x.value)
        hyp = extract_pred_answer(best_node.content)
        if eq(hyp, ref):
            corr += 1
    else:
        not_finished += 1
    total += 1
    total_prompt_tokens += problem.prompt_tokens
    total_completion_tokens += problem.completion_tokens
    total_tokens += problem.total_tokens
    total_runtime += problem.runtime_seconds

print(f"Accuracy: {corr}/{total}={corr/total}")
print(f"Not Finished: {not_finished}/{total}={not_finished/total}")
print(f"\n--- Runtime ---")
print(f"Total runtime: {total_runtime:.2f} seconds")
if total > 0:
    print(f"Average runtime per problem: {total_runtime / total:.2f} seconds")
print(f"\n--- Token Usage ---")
print(f"Total prompt tokens: {total_prompt_tokens}")
print(f"Total completion tokens: {total_completion_tokens}")
print(f"Total tokens: {total_tokens}")