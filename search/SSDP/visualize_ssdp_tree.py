import sys
import pickle
import graphviz
import os
import re
import textwrap

# Class definitions from ssdp.py, needed to load the pickle file
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
        self.window = [self.root]
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.embedding_prompt_tokens = 0
        self.embedding_completion_tokens = 0
        self.embedding_total_tokens = 0
        self.runtime_seconds = 0.0

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

def add_nodes_to_graph(graph, node, parent_id=None):
    """Recursively adds nodes and edges to the graphviz graph."""
    node_id = str(id(node))
    
    if parent_id is None: # Root node
        content = textwrap.fill(node.tree.question, width=50)
        label = f"QUESTION:\n{content}"
        color = 'orange'
    else:
        content = node.content.strip() if node.content else ""
        wrapped_content = textwrap.fill(content, width=50)
        label = f"Content: {wrapped_content}\nScore: {node.score:.2f}\nConfidence: {node.confidence:.2f}"
        color = 'lightblue'
        if node.is_leaf:
            color = 'lightgreen'
        if not node.children and not node.is_leaf:
            color = 'lightgrey'

    graph.node(node_id, label=label, shape='box', style='filled', fillcolor=color)
    
    if parent_id:
        graph.edge(parent_id, node_id)
        
    for child in node.children:
        add_nodes_to_graph(graph, child, node_id)

def main(pickle_fpath, question_index):
    """Main function to load data and generate the visualization."""
    print(f"Loading data from {pickle_fpath}...")
    try:
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pickle_fpath}")
        return
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return

    problems = data.get('problems', [])
    if not problems or question_index >= len(problems):
        print(f"Error: Question index {question_index} is out of bounds.")
        return

    tree_to_visualize = problems[question_index]
    true_answer = extract_answer(tree_to_visualize.answer)
    
    graph_title = f"SSDP Search Tree for Question {question_index}\nCorrect Answer: {true_answer}"
    dot = graphviz.Digraph(comment=f'SSDP Search Tree for Question {question_index}')
    dot.attr(rankdir='TB', size='50,50', dpi='350', fontsize='20', fontcolor='black', label=graph_title)

    print("Building graph...")
    add_nodes_to_graph(dot, tree_to_visualize.root)
    
    output_filename = f'ssdp_tree_q{question_index}'
    print(f"Rendering graph to {output_filename}.png ...")
    
    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"Successfully created {output_filename}.png")
    except graphviz.backend.ExecutableNotFound:
        print("\n--- Graphviz Error ---")
        print("Graphviz executable not found. Please make sure Graphviz is installed and in your system's PATH.")
        print("On Ubuntu/Debian, you can install it with: sudo apt-get install graphviz")
        print(f"The DOT source file was saved as '{output_filename}'. You can render it manually.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during rendering ---")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python visualize_ssdp_tree.py <path_to_pickle_file> [question_index]")
        sys.exit(1)

    pickle_fpath = sys.argv[1]
    question_index = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    
    main(pickle_fpath, question_index)