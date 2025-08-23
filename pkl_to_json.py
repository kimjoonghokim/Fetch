
import pickle
import json
import argparse
import sys
import os

# This is needed so that the script can find the `search` module
sys.path.append(os.getcwd())

from search.SSDP.SSDP import SSDPTree, SSDPNode

def node_to_dict(node):
    if not node:
        return None
    return {
        "content": node.content,
        "timestep": node.timestep,
        "is_leaf": node.is_leaf,
        "overall_score": node.overall_score,
        "confidence_score": node.confidence_score,
        "detailed_scores": node.detailed_scores,
        "pruned": node.pruned,
        "children": [node_to_dict(child) for child in node.children]
    }

def tree_to_dict(tree):
    return {
        "question": tree.question,
        "answer": tree.answer,
        "root": node_to_dict(tree.root),
        "total_expansions": tree.total_expansions,
        "total_merges": tree.total_merges,
        "total_prunes": tree.total_prunes,
    }


def convert_pkl_to_json(pkl_path, json_path):
    """
    Converts a pickle file to a JSON file.

    Args:
        pkl_path (str): The path to the input pickle file.
        json_path (str): The path to the output JSON file.
    """
    try:
        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        if isinstance(data, list):
            json_data = [tree_to_dict(tree) for tree in data]
        else:
            json_data = tree_to_dict(data)


        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Successfully converted '{pkl_path}' to '{json_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a .pkl file to a .json file.')
    parser.add_argument('pkl_file', help='The path to the input .pkl file')
    parser.add_argument('json_file', help='The path to the output .json file')
    args = parser.parse_args()

    convert_pkl_to_json(args.pkl_file, args.json_file)
