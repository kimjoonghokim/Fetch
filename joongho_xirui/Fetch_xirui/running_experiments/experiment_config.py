#!/usr/bin/env python3
"""
⚙️ Experiment Configuration for Tree-of-Thoughts Research
Modify these parameters to customize your experiments.
"""

# Dataset Configuration
DATASET_CONFIG = {
    "toy_dataset": "dataset/toy.jsonl",  # Small dataset for testing
    "gsm8k_test": "gsm8k/test.jsonl",   # Full GSM8K test set (if available)
    "max_problems": 5,                   # Number of problems to test (set to None for all)
}

# BFS Experiment Parameters
BFS_CONFIG = {
    "max_depths": [3, 5, 7, 10],        # Different search depths to test
    "max_budgets": [5, 10, 15, 20],     # Different computational budgets
    "temperatures": [0.5, 0.8, 1.0],    # Different temperature settings for policy
    "random_seed": 42,                   # For reproducibility
}

# MCTS Experiment Parameters
MCTS_CONFIG = {
    "n_rollouts": [3, 5, 8, 12],        # Different numbers of rollouts
    "max_times": [30, 60, 120],          # Different time limits (seconds)
    "c_values": [0.3, 0.6, 1.0, 1.5],  # Different UCB exploration constants
    "alpha": 0.5,                        # Weight for value vs. prior
    "min_terminals": 10,                 # Minimum terminal nodes before stopping
}

# Beam Search Experiment Parameters
BEAMSEARCH_CONFIG = {
    "beam_sizes": [2, 3, 5, 8],         # Different beam sizes
    "budgets": [5, 10, 15],              # Different computational budgets
    "temperatures": [0.5, 0.8, 1.0],    # Different temperature settings
    "max_depth": 10,                     # Maximum tree depth
}

# State Merging Experiment Parameters
MERGING_CONFIG = {
    "merge_thresholds": [0.05, 0.1, 0.15, 0.2],  # Different clustering thresholds
    "embedding_model": "xmu-nlp/simcse-large-gsm8k",  # SimCSE model for clustering
    "linkage_methods": ["average", "single", "complete"],  # Different clustering methods
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_dir": "experiment_results",  # Directory to save results
    "save_format": "pickle",             # Format: "pickle" or "json"
    "generate_plots": True,              # Whether to generate visualization plots
    "save_detailed_logs": True,          # Whether to save detailed execution logs
}

# Policy Server Configuration (if using real policy model)
POLICY_CONFIG = {
    "url": "127.0.0.1",
    "port": 8000,
    "model_name": "xmu-nlp/Llama-3-8b-gsm8k",
    "max_tokens": 512,
    "timeout": 30,
}

# Clustering Server Configuration (if using real clustering service)
CLUSTERING_CONFIG = {
    "url": "127.0.0.1",
    "port": 8003,
    "timeout": 30,
}

# Experiment Schedules
EXPERIMENT_SCHEDULES = {
    "quick_test": {
        "description": "Quick test with minimal parameters",
        "bfs": {"max_depths": [3, 5], "max_budgets": [5, 10]},
        "mcts": {"n_rollouts": [3, 5], "max_times": [30, 60]},
        "beamsearch": {"beam_sizes": [2, 3], "budgets": [5, 10]},
        "merging": {"merge_thresholds": [0.1, 0.2]},
        "max_problems": 3,
    },
    
    "comprehensive": {
        "description": "Full parameter sweep for paper",
        "bfs": {"max_depths": [3, 5, 7, 10], "max_budgets": [5, 10, 15, 20]},
        "mcts": {"n_rollouts": [3, 5, 8, 12], "max_times": [30, 60, 120]},
        "beamsearch": {"beam_sizes": [2, 3, 5, 8], "budgets": [5, 10, 15]},
        "merging": {"merge_thresholds": [0.05, 0.1, 0.15, 0.2]},
        "max_problems": None,  # All problems
    },
    
    "ablation_study": {
        "description": "Ablation study focusing on state merging",
        "bfs": {"max_depths": [5], "max_budgets": [10]},
        "mcts": {"n_rollouts": [5], "max_times": [60]},
        "beamsearch": {"beam_sizes": [3], "budgets": [10]},
        "merging": {"merge_thresholds": [0.05, 0.1, 0.15, 0.2, 0.3]},
        "max_problems": 10,
    }
}

# Metrics to Track
METRICS = {
    "performance": ["accuracy", "solution_length", "time_to_solution"],
    "efficiency": ["nodes_expanded", "memory_usage", "computation_time"],
    "quality": ["value_scores", "path_diversity", "convergence_rate"],
    "merging": ["clusters_found", "merge_ratio", "similarity_scores"],
}

# Random Scoring Configuration
RANDOM_SCORING = {
    "enabled": True,                     # Use random scoring instead of verifier
    "seed": 42,                          # Random seed for reproducibility
    "distribution": "uniform",           # "uniform", "normal", "beta"
    "range": [0, 1],                     # Score range [min, max]
    "normalize": True,                   # Whether to normalize scores
} 