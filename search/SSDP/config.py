"""
Configuration file for the new SSDP (Semantic Similarity based Dynamic Pruning) algorithm.
"""

# Search Parameters
LIMIT = 50                      # Maximum search iterations
MAX_DEPTH = 10                  # Maximum reasoning depth

# Exploration and Exploitation
EXPLORE_EXPLOIT_THRESHOLD = 0.5 # Score threshold to switch from exploit to explore

# Scoring Thresholds
OVERALL_SCORE_THRESHOLD = 0.3   # Minimum overall score to keep a path
SIMILARITY_THRESHOLD = 0.85     # Similarity threshold for merging nodes
HIGH_QUALITY_THRESHOLD = 0.9    # Threshold for a high-quality solution

# Pruning and Merging
PRUNE_FREQUENCY = 1             # Prune every N iterations
MERGE_FREQUENCY = 1             # Merge similar nodes every N iterations

# Heuristic Pruning Parameters
MAX_PATH_LENGTH = 1024
REPETITION_PENALTY = 3
MIN_QUESTION_SIMILARITY = 0.1

# Depth-Aware and Budget-Aware Pruning
DEPTH_AWARE_PRUNING_FACTOR = 0.1 # How much to increase pruning threshold with depth
BUDGET_AWARE_PRUNING_FACTOR = 0.1 # How much to increase pruning threshold as budget is used

# Early Stopping
EARLY_STOPPING_PATIENCE = 5       # Number of iterations to wait for score improvement
EARLY_STOPPING_THRESHOLD = 0.01   # Minimum score improvement to reset patience

# Model Parameters
TEMPERATURE = 0.8               # Model temperature
MAX_LEN_PER_STEP = 256         # Maximum tokens per reasoning step

# File Paths
DATA_PATH = "../../dataset/toy.jsonl"
POLICY_MODEL = "xmu-nlp/Llama-3-8b-gsm8k"

# API Endpoints
POLICY_URL = "http://127.0.0.1:8000/v1/completions"

# Semantic Similarity Parameters
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
