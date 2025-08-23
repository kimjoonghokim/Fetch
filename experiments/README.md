# Math Reasoning Experiments Configuration

This directory contains the configuration system for running experiments on different models, datasets, and search methods for math reasoning tasks.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   cd experiments
   pip install -r requirements.txt
   ```

2. **Test the configuration:**
   ```bash
   python test_config.py
   ```

3. **Run a quick test:**
   ```bash
   python run_experiments.py
   # Choose option 1 for quick test
   ```

4. **Run full experiments:**
   ```bash
   python run_experiments.py
   # Choose option 2 for full experiments
   ```

## 📋 Configuration Overview

### Models
- **Qwen2.5-1.5B**: Small, fast model for quick testing
- **Qwen2.5-7B**: Medium-sized model with good performance
- **LLaMA-3.1-8B**: Large model with strong reasoning capabilities
- **LLaMA-3.2-3B**: Compact model for efficient experiments

### Datasets
- **GSM8K**: Grade school math word problems (8.5K problems)
- **MATH-500**: 500 challenging math problems
- **AIME2025**: Advanced math competition problems

### Search Methods
- **Beam Search**: Uses verifier model for scoring
- **BFS**: Breadth-first search with verifier
- **MCTS**: Monte Carlo Tree Search with verifier
- **SSDP**: Semantic Similarity Dynamic Pruning with scoring.py

## ⚙️ Configuration Structure

### Main Configuration Classes

1. **`ExperimentConfig`**: General experiment settings
   - Output directory and format
   - Parallel execution settings
   - Logging configuration

2. **`ModelConfig`**: Language model parameters
   - Model ID and name
   - Generation parameters (temperature, max_tokens)
   - Stop tokens

3. **`DatasetConfig`**: Dataset settings
   - Dataset ID and split
   - Sample limits for testing

4. **`SearchMethodConfig`**: Search algorithm settings
   - Whether to use verifier or scoring system
   - Algorithm-specific parameters

5. **`VerifierConfig`**: Verifier model settings
   - Model ID and API endpoints
   - Used by beamsearch, bfs, and mcts

6. **`ScoringConfig`**: Scoring system parameters
   - Policy model and ESM service endpoints
   - Tunable hyperparameters
   - Scoring weights

### Tunable Hyperparameters

#### Scoring System (SSDP) - All Parameters Configurable
- **Semantic Similarity Parameters**:
  - `similarity_threshold`: 0.15 (distance threshold for semantic clustering, 0.0-1.0)
  - `early_boost_factor`: 0.3 (boost factor for early reasoning nodes, 0.0-1.0)
  
- **Confidence Scoring Parameters**:
  - `confidence_method`: "confidence_average" (confidence calculation method)
  
- **Vote Scoring Parameters**:
  - `vote_score_merge_boost`: 0.1 (boost per semantic merge)
  - `vote_score_max_boost`: 1.0 (maximum vote score boost)
  
- **Repeating Node Scoring Parameters**:
  - `repeating_node_depth_advantage`: True (whether to apply depth-based advantage)
  - `repeating_node_max_boost`: 1.0 (maximum repeating node boost)
  
- **Parent Quality Scoring Parameters**:
  - `parent_quality_multiplier_range`: (0.1, 1.0) (range for parent multiplier)
  - `parent_confidence_normalization`: True (whether to normalize parent confidence)
  
- **ESM Service Parameters**:
  - `esm_timeout`: 10 (timeout for ESM service calls)
  - `esm_fallback_clustering`: True (whether to use fallback clustering if ESM fails)
  
- **Scoring Weights** (fully configurable):
  - `confidence`: 0.4, `vote_score`: 0.25, `repeating_node`: 0.2, `parent_quality`: 0.15

#### Search Algorithms
- **Beam Search**: `beam_width`, `max_depth`, `max_iterations`
- **BFS**: `max_depth`, `max_iterations`, `max_queue_size`
- **MCTS**: `num_simulations`, `exploration_constant`, `max_depth`
- **SSDP**: `limit`, `max_parallel_paths`, `similarity_threshold`

## 🔧 Customizing Configuration

### Quick Configuration Modifiers

```python
from config import quick_test_config, full_experiment_config

# Quick test with minimal data
quick_test_config()

# Full experiments with all combinations
full_experiment_config()
```

### Single Model/Dataset Experiments

```python
from config import single_model_config, single_dataset_config

# Run experiments with only one model
single_model_config("qwen2.5-1.5b")

# Run experiments with only one dataset
single_dataset_config("gsm8k")
```

### Custom Scoring Weights

```python
from config import scoring_config

# Customize scoring weights
scoring_config.scoring_weights = {
    'confidence': 0.5,
    'vote_score': 0.3,
    'repeating_node': 0.2,
    'parent_quality': 0.0
}
```

### Adjust Search Parameters

```python
from config import ssdp_config, beamsearch_config

# SSDP parameters
ssdp_config.similarity_threshold = 0.2
ssdp_config.max_parallel_paths = 10

# Beam search parameters
beamsearch_config.beam_width = 10
beamsearch_config.max_depth = 15
```

## 📊 Results Collection

The system collects the following metrics for each experiment:

- **Tokens**: Number of tokens used
- **Latency**: Execution time in milliseconds
- **Accuracy**: 1 if answer found, 0 if not found
- **Additional**: Search steps, final answer, configuration used

### Output Formats

- **JSON**: Human-readable, easy to analyze
- **Pickle**: Binary format, faster to load

### File Naming

Results are saved with timestamps:
```
experiment_results_20241201_143022.json
experiment_results_20241201_143022.pkl
```

## 🏗️ Architecture

```
config.py                    # Main configuration file
├── ModelConfig             # Language model settings
├── DatasetConfig           # Dataset parameters
├── SearchMethodConfig      # Search algorithm settings
├── VerifierConfig          # Verifier model configuration
├── ScoringConfig           # Scoring system parameters
├── Algorithm-specific      # Individual algorithm configs
├── ExecutionConfig         # Experiment execution settings
├── Dataset loading         # Hugging Face dataset loading
└── Model loading           # Transformers model loading

run_experiments.py          # Experiment runner with parallel execution
├── Configuration validation
├── Dataset loading
├── Parallel experiment execution
├── Results collection
└── JSON file output

test_config.py              # Configuration testing script
├── Configuration validation
├── Dataset loading tests
├── Model configuration tests
└── Search method tests

requirements.txt             # Python dependencies
```

## 🔍 API Endpoints

### Required Services

1. **Policy Model**: `http://127.0.0.1:8000/v1/completions` (Hard-coded)
2. **Verifier Model**: `http://127.0.0.1:8002/predict` (Hard-coded)
3. **ESM Service**: `http://127.0.0.1:8003/predict` (Hard-coded)

### Configuration

All endpoints are hard-coded in the respective config classes:
- `scoring_config.policy_full_url`
- `verifier_config.full_url`
- `scoring_config.esm_full_url`

**Note**: These ports are hard-coded as requested. Modify the config.py file if you need different ports.

## 🚨 Troubleshooting

### Common Issues

1. **Configuration validation fails**
   - Check if all required models/datasets/search methods exist
   - Verify output directory permissions

2. **API connection errors**
   - Ensure all services are running on correct ports
   - Check network connectivity

3. **Memory issues**
   - Reduce `max_questions_per_dataset` for testing
   - Use smaller models for initial experiments

### Validation Commands

```python
from config import validate_config, get_experiment_summary

# Validate configuration
validate_config()

# Show current configuration
print(get_experiment_summary())
```

## 📝 Example Usage

### Basic Experiment

```python
from config import execution_config, MODELS, DATASETS, SEARCH_METHODS

# Configure for specific experiments
execution_config.models_to_run = ["qwen2.5-1.5b"]
execution_config.datasets_to_run = ["gsm8k"]
execution_config.search_methods_to_run = ["ssdp"]

# Run experiments
for model in execution_config.models_to_run:
    for dataset in execution_config.datasets_to_run:
        for method in execution_config.search_methods_to_run:
            print(f"Running: {model} + {dataset} + {method}")
            # Your experiment logic here
```

### Custom Scoring

```python
from config import scoring_config

# Adjust similarity threshold
scoring_config.similarity_threshold = 0.25

# Custom weights
scoring_config.scoring_weights = {
    'confidence': 0.6,
    'vote_score': 0.4
}
```

## 🔮 Future Enhancements

- **Parallel execution**: Run multiple experiments simultaneously
- **Hyperparameter tuning**: Automated parameter optimization
- **Result analysis**: Built-in visualization and analysis tools
- **Experiment tracking**: Integration with MLflow or similar tools
- **Distributed execution**: Run experiments across multiple machines

## 📚 References

- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [MATH-500 Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [AIME2025 Dataset](https://huggingface.co/datasets/opencompass/AIME2025)
- [Scoring System Documentation](../scorer/scoring.py)
- [Search Methods](../search/)
