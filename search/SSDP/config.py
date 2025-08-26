"""
Configuration file for SSDP (Semantic Similarity based Dynamic Pruning) algorithm.
Uses scoring.py instead of verifier model for all scoring.
"""

# Search Parameters
LIMIT = 50                      # Maximum search iterations
MAX_PARALLEL_PATHS = 8          # Maximum number of parallel paths to explore
MIN_EXPANSION_BUDGET = 3        # Minimum expansions per node
MAX_EXPANSION_BUDGET = 5        # Maximum expansions per node
MAX_DEPTH = 10                  # Maximum reasoning depth

# Scoring Thresholds (using scoring.py system)
OVERALL_SCORE_THRESHOLD = 0.3   # Minimum overall score to keep a path
SIMILARITY_THRESHOLD = 0.85     # Similarity threshold for merging nodes
HIGH_QUALITY_THRESHOLD = 0.8    # Threshold for early stopping

# Pruning and Merging
PRUNE_FREQUENCY = 3             # Prune every N iterations
MERGE_FREQUENCY = 2             # Merge similar nodes every N iterations

# Heuristic Pruning Parameters
MAX_PATH_LENGTH = 1024
REPETITION_PENALTY = 1.0
MIN_QUESTION_SIMILARITY = 0.1

# Model Parameters
TEMPERATURE = 0.8               # Model temperature
MAX_LEN_PER_STEP = 256         # Maximum tokens per reasoning step

# File Paths
DATA_PATH = "../../dataset/toy.jsonl"
POLICY_MODEL = "xmu-nlp/Llama-3-8b-gsm8k"

# API Endpoints (only policy model needed)
POLICY_URL = "http://127.0.0.1:8000/v1/completions"

# Semantic Similarity Parameters
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)

# Scoring System Integration
# SSDP uses the comprehensive scoring.py system:
# - Overall score (primary ranking metric)
# - Confidence score (for analysis)
# - Future scoring components as they become available

# Custom weights for scoring components (when available)
CUSTOM_SCORING_WEIGHTS = {
    'confidence': 0.7,
    'length_penalty': 0.2,
    'parent_child_quality': 0.1,  # Future
    'semantic_similarity': 0.0,   # Future
    'coherence': 0.0,            # Future
    'factual_consistency': 0.0   # Future
}