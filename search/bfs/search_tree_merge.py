import os
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv(dotenv_path='../../server_config.env')
model_fpath = os.getenv("EMBEDDING_MODEL_PATH", "path/to/merge/model")
tokenizer = None
model = None

def call_esm(texts, distance=0.15):
    import requests
    url = "http://127.0.0.1:8003/predict"
    pload ={"texts": texts, "d": distance}
    response =requests.post(url, json=pload)
    response_json = response.json()
    usage = response_json.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return response_json["labels"], usage

def compute_emb(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        # embeddings = model(**inputs, output_hidden_states=True, return_dict=False)[0][:,0]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()

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
        return "\n".join(self.return_path())


class VirtualNode:
    def __init__(self, nodes, strategy="max", d=0.15):
        self.nodes = sorted(nodes, key=lambda x: x.value, reverse=True)
        self.tree = self.nodes[0].tree
        if strategy == "avg":
            self.value = sum([node.value for node in nodes]) / len(nodes) # avg
        elif strategy == "max":
            self.value = self.nodes[0].value # max
        elif strategy == "min":
            self.value = self.nodes[-1].value # min
        else:
            raise Exception("UNKNOWN STRATEGY")
        self.visited = False
        self.children = []
        self.cache = []
        self.strategy = strategy
        self.d = d
        self.is_leaf = self.nodes[0].is_leaf if self.nodes else False
    
    def merge_nodes(self):
        if self.cache:
            if len(self.cache) > 1:
                contents = [child.content for child in self.cache]
                labels, usage = call_esm(contents, self.d)
                self.tree.embedding_prompt_tokens += usage.get("prompt_tokens", 0)
                self.tree.embedding_completion_tokens += usage.get("completion_tokens", 0)
                self.tree.embedding_total_tokens += usage.get("total_tokens", 0)

                clusters = {}
                for child, label in zip(self.cache, labels):
                    key = label
                    if key not in clusters:
                        clusters[key] = []
                    clusters[key].append(child)
            else:
                clusters = {0: [self.cache[0]]}
            for nodes in clusters.values():
                virtual_node = VirtualNode(nodes, self.strategy, self.d)
                self.children.append(virtual_node)
                self.tree.virtual_nodes.append(virtual_node)
            self.cache = [] # clean cache

class Tree:
    def __init__(self, question, answer, additional_info):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.virtual_nodes = []
        self.root = None # wait init
        self.additional_info = additional_info
        # Policy server token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # Verifier server token tracking
        self.verifier_prompt_tokens = 0
        self.verifier_completion_tokens = 0
        self.verifier_total_tokens = 0
        # Embedding server token tracking
        self.embedding_prompt_tokens = 0
        self.embedding_completion_tokens = 0
        self.embedding_total_tokens = 0
        # Runtime tracking
        self.runtime_seconds = 0.0

    def init_root_node(self, value):
        self.root = Node(None, value, None, 0, self)
        self.all_nodes.append(self.root)
        self.virtual_nodes.append(VirtualNode([self.root], 
        self.additional_info["strategy"], self.additional_info["d"]))

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, timestep, is_leaf=False):
        node = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def select_next_node(self, timestep):
        # latest timestep and highest value
        available_nodes = [node for node in self.all_nodes if node.timestep == timestep]
        if len(available_nodes) == 0:
            return None # end previously
            # raise Exception("No node with timestep {}".format(timestep))
        selected_node = max(available_nodes, key=lambda x: x.value)
        if selected_node.is_leaf: # search done
            return None
        return selected_node

    def select_best_node(self, visit_cnt=1):
        available_nodes = [node for node in self.all_nodes if len(node.children) < visit_cnt] # have not been explored
        if len(available_nodes) == 0:
            return None # end previously
            # raise Exception("No available node")
        best_node = max(available_nodes, key=lambda x: x.value)
        if best_node.is_leaf: # search done
            return None
        return best_node

    def select_best_cluster(self):
        best_cluster = None
        for virtual_node in self.virtual_nodes:
            if not virtual_node.visited:
                if best_cluster is None or virtual_node.value > best_cluster.value:
                    best_cluster = virtual_node
        if best_cluster is None: # all have been visited or have not started
            return None, None
        returned_nodes = [node for node in best_cluster.nodes if not node.is_leaf]
        if len(returned_nodes) == 0:
            return None, None
        best_cluster.visited = True
        return best_cluster, returned_nodes

    def return_best_path(self, use_greedy=True):
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]
        if leaf_nodes:
            state = max(leaf_nodes, key=lambda x: x.value)
            return state.print_path(), state.value
        return None, None