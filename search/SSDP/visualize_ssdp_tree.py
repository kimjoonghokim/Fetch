import sys
import pickle
import graphviz
import os
import re
import textwrap
from collections import defaultdict

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
        self.cluster_id = None
        self.is_representative = False

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
        self.pruning_history = []
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

def is_correct(generated_answer, true_answer):
    if generated_answer is None or true_answer is None:
        return False
    try:
        return abs(float(generated_answer) - float(true_answer)) < 1e-3
    except (ValueError, TypeError):
        return False

def build_graph(dot, tree):
    """Builds the graph visualization from the tree data, level by level."""
    nodes_by_id = {id(node): node for node in tree.all_nodes}
    nodes_by_timestep = defaultdict(list)
    for node in tree.all_nodes:
        nodes_by_timestep[node.timestep].append(node)

    pruning_history_map = {h['timestep']: h for h in getattr(tree, 'pruning_history', [])}

    max_timestep = max(nodes_by_timestep.keys()) if nodes_by_timestep else 0

    for i in range(max_timestep + 1):
        # Create a subgraph for the current level to enforce rank
        with dot.subgraph(name=f'rank_{i}') as s:
            s.attr(rank='same')
            
            # Add invisible node for threshold label
            threshold_info = pruning_history_map.get(i)
            if threshold_info:
                label = (
                    f"Threshold: {threshold_info['final_threshold']:.2f}\n"
                    f"(Rel: {threshold_info['relative_threshold']:.2f}, "
                    f"Depth: {threshold_info['depth_threshold']:.2f})")
                s.node(f'level_{i}_info', label=label, shape='plaintext', fontsize='10')

            nodes_in_level = nodes_by_timestep[i]
            clusters = defaultdict(list)
            for node in nodes_in_level:
                if node.cluster_id:
                    clusters[node.cluster_id].append(node)
            
            if not clusters: # Handle nodes without clusters (like root)
                for node in nodes_in_level:
                    add_node_to_graph(s, node)
            else:
                for cluster_id, nodes_in_cluster in clusters.items():
                    with s.subgraph(name=f'cluster_{cluster_id}') as c:
                        c.attr(label=f'Cluster\n{cluster_id}', style='rounded', color='black')
                        for node in nodes_in_cluster:
                            add_node_to_graph(c, node)

    # Add all edges between nodes
    for node_id, node in nodes_by_id.items():
        if node.parent:
            parent_id = str(id(node.parent))
            dot.edge(parent_id, str(node_id))

def add_node_to_graph(graph, node):
    """Adds a single node to the graph with appropriate styling."""
    node_id = str(id(node))
    
    if node.parent is None: # Root node
        content = textwrap.fill(node.tree.question, width=50)
        label = f"QUESTION:\n{content}"
        color = 'orange'
    else:
        content = textwrap.fill(node.content.strip(), width=50)
        label = f"Content: {content}\nScore: {node.score:.2f}\nConf: {node.confidence:.2f}"
        
        penwidth = '1'
        fillcolor = 'white' # Default for non-representative

        if node.is_representative:
            penwidth = '3'
            if node.children or node.is_leaf:
                fillcolor = 'lightblue' # Expanded or terminal
            else:
                fillcolor = 'lightgrey' # Pruned representative
        
        if node.is_leaf:
            fillcolor = 'lightgreen'

    graph.node(node_id, label=label, shape='box', style='filled', fillcolor=fillcolor, penwidth=penwidth)

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

    # Determine outcome
    best_node = None
    outcome_str = "Outcome: No terminal nodes found"
    if tree_to_visualize.terminal_nodes:
        best_node = max(tree_to_visualize.terminal_nodes, key=lambda node: node.score)
        predicted_answer = extract_answer(best_node.print_path())
        if is_correct(predicted_answer, true_answer):
            outcome_str = f"Outcome: CORRECT (Predicted: {predicted_answer})"
        else:
            outcome_str = f"Outcome: INCORRECT (Predicted: {predicted_answer})"

    graph_title = (
                   f"SSDP Search Tree for Question {question_index}\n"
                   f"Correct Answer: {true_answer}\n"
                   f"{outcome_str}")

    dot = graphviz.Digraph(comment=f'SSDP Search Tree for Question {question_index}')
    dot.attr(rankdir='TB', size='50,50', dpi='150', fontsize='12', fontcolor='black', label=graph_title, labelloc='t')

    print("Building graph...")
    build_graph(dot, tree_to_visualize)
    
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
