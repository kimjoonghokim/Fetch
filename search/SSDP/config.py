"""
Configuration file for the new SSDP (Semantic Similarity based Dynamic Pruning) algorithm.
"""

# Search Parameters
LIMIT = 50                      # Maximum search iterations
MAX_PARALLEL_PATHS = 8          # Maximum number of parallel paths to explore
MAX_DEPTH = 10                  # Maximum reasoning depth

# Scoring Thresholds
SIMILARITY_THRESHOLD = 0.85     # Similarity threshold for merging nodes
HIGH_QUALITY_THRESHOLD = 0.9    # Threshold for a high-quality solution

# Pruning and Merging
PRUNE_RATIO = 0.5               # The ratio of nodes to prune in each iteration
MERGE_FREQUENCY = 1             # Merge similar nodes every N iterations

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
