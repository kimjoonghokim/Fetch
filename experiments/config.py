"""
Configuration file for running experiments on different models, datasets, and search methods.
This config file centralizes all experiment parameters for easy modification and execution.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum
import time
from datetime import datetime

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Main configuration class for experiments"""
    
    # Experiment metadata
    experiment_name: str = "math_reasoning_experiments"
    output_dir: str = "experiment_results"
    output_format: str = "json"  # JSON format for results
    experiment_description: str = "Comprehensive evaluation of math reasoning models using different search algorithms"
    
    # Execution settings
    run_parallel: bool = True  # Enable parallel execution by default
    max_workers: int = 4      # Number of parallel workers
    save_intermediate_results: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "experiment.log"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for language models"""
    name: str
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.8
    stop_tokens: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ["\n"]

# Available models for experiments
MODELS = {
    "qwen2.5-1.5b": ModelConfig(
        name="Qwen2.5-1.5B",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",  # Correct model ID - uses Instruct suffix
        max_tokens=512,
        temperature=0.8
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B", 
        model_id="Qwen/Qwen2.5-7B-Instruct",  # Correct model ID - uses Instruct suffix
        max_tokens=512,
        temperature=0.8
    ),
    "llama3.1-8b": ModelConfig(
        name="LLaMA-3.1-8B",
        model_id="meta-llama/Llama-3.1-8B-Instruct",  # Requires HF token
        max_tokens=512,
        temperature=0.8
    ),
    "llama3.2-3b": ModelConfig(
        name="LLaMA-3.2-3B",
        model_id="meta-llama/Llama-3.2-3B-Instruct",  # Requires HF token
        max_tokens=512,
        temperature=0.8
    )
}

# Alternative smaller models for testing (if the above don't work)
ALTERNATIVE_MODELS = {
    "tiny-llama": ModelConfig(
        name="TinyLlama",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Only 1.1B parameters
        max_tokens=512,
        temperature=0.8
    ),
    "phi-2": ModelConfig(
        name="Phi-2",
        model_id="microsoft/phi-2",  # Only 2.7B parameters
        max_tokens=512,
        temperature=0.8
    ),
    "gemma-2b": ModelConfig(
        name="Gemma-2B",
        model_id="google/gemma-2b-it",  # Only 2B parameters
        max_tokens=512,
        temperature=0.8
    )
}

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for datasets"""
    name: str
    dataset_id: str
    split: str = "test"
    config_name: Optional[str] = None  # Config name for datasets that need it
    max_samples: Optional[int] = None  # None means use all samples
    subset_size: Optional[int] = None  # For quick testing

# Available datasets for experiments
DATASETS = {
    "gsm8k": DatasetConfig(
        name="GSM8K",
        dataset_id="openai/gsm8k",  # Full dataset ID
        split="test",
        config_name="main",  # GSM8K requires config name
        max_samples=None
    ),
    "math500": DatasetConfig(
        name="MATH-500",
        dataset_id="HuggingFaceH4/MATH-500", 
        split="test",
        max_samples=None
    ),
    "aime": DatasetConfig(
        name="AIME2025",
        dataset_id="opencompass/AIME2025",
        split="test",
        config_name="AIME2025-I",  # AIME requires config name
        max_samples=None
    )
}

# ============================================================================
# SEARCH METHOD CONFIGURATIONS
# ============================================================================

@dataclass
class SearchMethodConfig:
    """Base configuration for search methods"""
    name: str
    use_verifier: bool = True  # Whether this method uses verifier model
    use_scorer: bool = False   # Whether this method uses scoring.py
    description: str = ""      # Description of the search method

# Available search methods
SEARCH_METHODS = {
    "beamsearch": SearchMethodConfig(
        name="Beam Search",
        use_verifier=True,
        use_scorer=False,
        description="Beam search algorithm that explores multiple promising paths simultaneously using a verifier model for scoring"
    ),
    "bfs": SearchMethodConfig(
        name="Breadth-First Search", 
        use_verifier=True,
        use_scorer=False,
        description="Breadth-first search that explores all nodes at the current depth before moving to the next level, using verifier model for scoring"
    ),
    "mcts": SearchMethodConfig(
        name="Monte Carlo Tree Search",
        use_verifier=True, 
        use_scorer=False,
        description="Monte Carlo Tree Search algorithm that balances exploration and exploitation using verifier model for node evaluation"
    ),
    "ssdp": SearchMethodConfig(
        name="Semantic Similarity Dynamic Pruning",
        use_verifier=False,
        use_scorer=True,
        description="Semantic Similarity based Dynamic Pruning that uses the scoring.py system for comprehensive answer evaluation"
    )
}

# ============================================================================
# VERIFIER CONFIGURATION
# ============================================================================

@dataclass
class VerifierConfig:
    """Configuration for verifier model (used by beamsearch, bfs, mcts)"""
    model_id: str = "xmu-nlp/Llama-3-8b-gsm8k-value-A"
    url: str = "http://127.0.0.1"
    port: int = 8002
    endpoint: str = "/predict"
    timeout: int = 30
    description: str = "Shared verifier model used by beamsearch, bfs, and mcts for scoring reasoning paths"
    
    @property
    def full_url(self) -> str:
        return f"{self.url}:{self.port}{self.endpoint}"

# ============================================================================
# SCORING SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class ScoringConfig:
    """Configuration for scoring system (used by SSDP)"""
    
    # Policy model configuration
    policy_url: str = "http://127.0.0.1"
    policy_port: int = 8000
    policy_endpoint: str = "/v1/completions"
    
    # ESM service configuration
    esm_url: str = "http://127.0.0.1"
    esm_port: int = 8003
    esm_endpoint: str = "/predict"
    
    # Scoring method weights
    scoring_weights: Dict[str, float] = None
    
    # ============================================================================
    # SCORING SYSTEM TUNABLE HYPERPARAMETERS
    # ============================================================================
    
    # Semantic similarity parameters
    similarity_threshold: float = 0.15  # Distance threshold for semantic clustering (0.0-1.0)
    early_boost_factor: float = 0.3    # Boost factor for early reasoning nodes (0.0-1.0)
    
    # Confidence scoring parameters
    confidence_method: str = "confidence_average"  # confidence_average, confidence_min, confidence_geometric, perplexity
    
    # Vote scoring parameters
    vote_score_merge_boost: float = 0.1  # Boost per semantic merge (default: 0.1)
    vote_score_max_boost: float = 1.0    # Maximum vote score boost (default: 1.0)
    
    # Repeating node scoring parameters
    repeating_node_depth_advantage: bool = True  # Whether to apply depth-based advantage
    repeating_node_max_boost: float = 1.0        # Maximum repeating node boost
    
    # Parent quality scoring parameters
    parent_quality_multiplier_range: tuple = (0.1, 1.0)  # Range for parent multiplier
    parent_confidence_normalization: bool = True          # Whether to normalize parent confidence
    
    # ESM service parameters
    esm_timeout: int = 10  # Timeout for ESM service calls
    esm_fallback_clustering: bool = True  # Whether to use fallback clustering if ESM fails
    
    def __post_init__(self):
        if self.scoring_weights is None:
            self.scoring_weights = {
                'confidence': 0.4,
                'vote_score': 0.25,
                'repeating_node': 0.2,
                'parent_quality': 0.15,
                'semantic_similarity': 0.0,
                'length_penalty': 0.0,
                'coherence': 0.0,
                'factual_consistency': 0.0
            }
    
    @property
    def policy_full_url(self) -> str:
        return f"{self.policy_url}:{self.policy_port}{self.policy_endpoint}"
    
    @property
    def esm_full_url(self) -> str:
        return f"{self.esm_url}:{self.esm_port}{self.esm_endpoint}"

# ============================================================================
# SEARCH ALGORITHM SPECIFIC CONFIGURATIONS
# ============================================================================

@dataclass
class BeamSearchConfig:
    """Configuration for beam search algorithm"""
    beam_width: int = 5
    max_depth: int = 10
    max_iterations: int = 100

@dataclass
class BFSConfig:
    """Configuration for breadth-first search algorithm"""
    max_depth: int = 10
    max_iterations: int = 100
    max_queue_size: int = 1000

@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search algorithm"""
    num_simulations: int = 100
    exploration_constant: float = 1.414
    max_depth: int = 10
    max_iterations: int = 100

@dataclass
class SSDPConfig:
    """Configuration for SSDP algorithm"""
    limit: int = 50
    max_parallel_paths: int = 8
    min_expansion_budget: int = 3
    max_expansion_budget: int = 5
    max_depth: int = 10
    overall_score_threshold: float = 0.3
    similarity_threshold: float = 0.85
    high_quality_threshold: float = 0.8
    prune_frequency: int = 3
    merge_frequency: int = 2
    max_len_per_step: int = 256

# ============================================================================
# EXPERIMENT EXECUTION CONFIGURATION
# ============================================================================

@dataclass
class ExecutionConfig:
    """Configuration for experiment execution"""
    
    # Which experiments to run
    models_to_run: List[str] = None
    datasets_to_run: List[str] = None
    search_methods_to_run: List[str] = None
    
    # Experiment limits
    max_questions_per_dataset: int = 3  # Limit for quick testing (small for debugging)
    timeout_per_question: int = 300  # seconds
    
    # Memory management
    load_models_sequentially: bool = True  # Load one model at a time to save memory
    unload_model_after_use: bool = True    # Unload model after all its experiments
    
    # Results collection
    collect_tokens: bool = True
    collect_latency: bool = True
    collect_accuracy: bool = True
    
    def __post_init__(self):
        if self.models_to_run is None:
            self.models_to_run = list(MODELS.keys())
        if self.datasets_to_run is None:
            self.datasets_to_run = list(DATASETS.keys())
        if self.search_methods_to_run is None:
            self.search_methods_to_run = list(SEARCH_METHODS.keys())

# ============================================================================
# DATASET AND MODEL LOADING FUNCTIONS
# ============================================================================

def load_dataset(dataset_name: str, max_samples: Optional[int] = None):
    """
    Load a dataset from Hugging Face datasets.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'math500', 'aime')
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Dataset object with questions and answers
    """
    try:
        from datasets import load_dataset
        
        print(f"🔄 Loading dataset: {dataset_name}")
        
        # Get dataset configuration
        dataset_config = DATASETS.get(dataset_name)
        if not dataset_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Build load_dataset arguments
        load_args = {
            "path": dataset_config.dataset_id,
            "split": dataset_config.split
        }
        
        # Add config name if specified
        if dataset_config.config_name:
            load_args["name"] = dataset_config.config_name
            print(f"  📋 Using config: {dataset_config.config_name}")
        
        # Load the dataset
        dataset = load_dataset(**load_args)
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        print(f"✅ Successfully loaded {dataset_name} dataset with {len(dataset)} samples")
        print(f"  📊 Sample fields: {list(dataset[0].keys())}")
        return dataset
        
    except ImportError:
        print("❌ Error: datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset {dataset_name}: {e}")
        print(f"  💡 Try installing the dataset manually:")
        if dataset_name == "gsm8k":
            print(f"     python -c \"from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='test')\"")
        elif dataset_name == "aime":
            print(f"     python -c \"from datasets import load_dataset; load_dataset('opencompass/AIME2025', split='test')\"")
        return None

def load_model(model_name: str):
    """
    Load a language model for text generation.
    
    Args:
        model_name: Name of the model (e.g., 'qwen2.5-1.5b', 'llama3.1-8b')
        
    Returns:
        Model and tokenizer objects
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_config = MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"🔄 Loading model: {model_config.name} ({model_config.model_id})")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
        
        # Use more aggressive quantization to save disk space
        try:
            from transformers import BitsAndBytesConfig
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                # Use flash attention if available
                attn_implementation="flash_attention_2" if hasattr(torch, 'flash_attention') else "eager"
            )
        except ImportError:
            # Fallback if bitsandbytes is not available
            print("⚠️  bitsandbytes not available, using standard loading")
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        print(f"✅ Model {model_config.name} loaded successfully")
        return model, tokenizer
        
    except ImportError:
        print("❌ Error: transformers library not found. Install with: pip install transformers torch")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        
        # Provide helpful error messages
        if "Can't load the model" in str(e):
            print(f"💡 This usually means:")
            print(f"   1. The model ID '{model_config.model_id}' doesn't exist")
            print(f"   2. You need authentication for this model")
            print(f"   3. The model is too large for your disk space")
            print(f"💡 Try using a smaller alternative model:")
            for alt_name, alt_config in ALTERNATIVE_MODELS.items():
                print(f"   - {alt_name}: {alt_config.name} ({alt_config.model_id})")
        
        return None, None

def get_dataset_sample(dataset, index: int):
    """
    Get a sample from the dataset with proper formatting.
    
    Args:
        dataset: Dataset object
        index: Sample index
        
    Returns:
        Dictionary with question and answer
    """
    try:
        sample = dataset[index]
        
        # Handle different dataset formats
        if "question" in sample and "answer" in sample:
            return {
                "question": sample["question"],
                "answer": sample["answer"],
                "index": index
            }
        elif "problem" in sample and "solution" in sample:
            return {
                "question": sample["problem"],
                "answer": sample["solution"],
                "index": index
            }
        else:
            # Try to find question/answer fields
            question_field = None
            answer_field = None
            
            for key in sample.keys():
                if "question" in key.lower() or "problem" in key.lower():
                    question_field = key
                elif "answer" in key.lower() or "solution" in key.lower():
                    answer_field = key
            
            if question_field and answer_field:
                return {
                    "question": sample[question_field],
                    "answer": sample[answer_field],
                    "index": index
                }
            else:
                print(f"⚠️  Warning: Could not identify question/answer fields in dataset sample")
                return {"question": str(sample), "answer": "", "index": index}
                
    except Exception as e:
        print(f"❌ Error getting dataset sample {index}: {e}")
        return None

# ============================================================================
# MAIN CONFIGURATION INSTANCE
# ============================================================================

# Create main configuration instance
experiment_config = ExperimentConfig()
verifier_config = VerifierConfig()
scoring_config = ScoringConfig()

# Search algorithm specific configs
beamsearch_config = BeamSearchConfig()
bfs_config = BFSConfig()
mcts_config = MCTSConfig()
ssdp_config = SSDPConfig()

# Execution configuration
execution_config = ExecutionConfig()

# ============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# ============================================================================

def validate_config():
    """Validate the configuration and check for any issues"""
    errors = []
    
    # Check if required models exist
    for model_name in execution_config.models_to_run:
        if model_name not in MODELS:
            errors.append(f"Model '{model_name}' not found in MODELS configuration")
    
    # Check if required datasets exist
    for dataset_name in execution_config.datasets_to_run:
        if dataset_name not in DATASETS:
            errors.append(f"Dataset '{dataset_name}' not found in DATASETS configuration")
    
    # Check if required search methods exist
    for method_name in execution_config.search_methods_to_run:
        if method_name not in SEARCH_METHODS:
            errors.append(f"Search method '{method_name}' not found in SEARCH_METHODS configuration")
    
    # Check if output directory exists or can be created
    if not os.path.exists(experiment_config.output_dir):
        try:
            os.makedirs(experiment_config.output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory '{experiment_config.output_dir}': {e}")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    print("✅ Configuration validation passed!")
    return True

def test_model_availability():
    """Test if models can be loaded (without actually loading them)"""
    print("\n🧪 TESTING MODEL AVAILABILITY")
    print("="*50)
    
    try:
        from huggingface_hub import model_info
        
        for model_name in execution_config.models_to_run:
            model_config = MODELS.get(model_name)
            if not model_config:
                continue
                
            try:
                print(f"\n🔍 Testing model: {model_name}")
                print(f"   📝 Name: {model_config.name}")
                print(f"   🆔 ID: {model_config.model_id}")
                
                # Check if model exists on Hugging Face
                info = model_info(model_config.model_id)
                print(f"   ✅ Model exists on Hugging Face")
                print(f"   📊 Downloads: {info.downloads:,}")
                print(f"   🏷️  Tags: {', '.join(info.tags[:5])}")
                
            except Exception as e:
                print(f"   ❌ Model not available: {e}")
                print(f"   💡 This model may require authentication or doesn't exist")
                
    except ImportError:
        print("❌ huggingface_hub not available for model testing")
        print("💡 Install with: pip install huggingface_hub")
    
    print("\n💡 If models fail to load, try the alternative smaller models:")
    for alt_name, alt_config in ALTERNATIVE_MODELS.items():
        print(f"   - {alt_name}: {alt_config.name} ({alt_config.model_id})")

def get_experiment_summary():
    """Get a summary of the current experiment configuration"""
    summary = f"""
🔬 EXPERIMENT CONFIGURATION SUMMARY
{'='*50}

📊 MODELS ({len(execution_config.models_to_run)}):
{chr(10).join(f"  • {MODELS[model].name} ({model})" for model in execution_config.models_to_run)}

📚 DATASETS ({len(execution_config.datasets_to_run)}):
{chr(10).join(f"  • {DATASETS[dataset].name} ({dataset})" for dataset in execution_config.datasets_to_run)}

🔍 SEARCH METHODS ({len(execution_config.search_methods_to_run)}):
{chr(10).join(f"  • {SEARCH_METHODS[method].name} ({method})" for method in execution_config.search_methods_to_run)}

⚙️  CONFIGURATION:
  • Output Directory: {experiment_config.output_dir}
  • Output Format: {experiment_config.output_format}
  • Parallel Execution: {experiment_config.run_parallel}
  • Max Questions per Dataset: {execution_config.max_questions_per_dataset}
  • Timeout per Question: {execution_config.timeout_per_question}s

📈 METRICS TO COLLECT:
  • Tokens: {execution_config.collect_tokens}
  • Latency: {execution_config.collect_latency}
  • Accuracy: {execution_config.collect_accuracy}

🎯 TOTAL EXPERIMENTS: {len(execution_config.models_to_run) * len(execution_config.datasets_to_run) * len(execution_config.search_methods_to_run)}
"""
    return summary

# ============================================================================
# QUICK CONFIGURATION MODIFIERS
# ============================================================================

def quick_test_config():
    """Quick configuration for testing with minimal data"""
    execution_config.max_questions_per_dataset = 5
    execution_config.models_to_run = ["qwen2.5-1.5b"]
    execution_config.datasets_to_run = ["gsm8k"]
    execution_config.search_methods_to_run = ["ssdp"]
    execution_config.load_models_sequentially = True  # Enable sequential loading for memory efficiency
    execution_config.unload_model_after_use = True
    print("🔧 Quick test configuration applied!")
    print("💾 Sequential model loading enabled to save memory")

def full_experiment_config():
    """Full configuration for complete experiments"""
    execution_config.max_questions_per_dataset = None  # Use all samples
    execution_config.models_to_run = list(MODELS.keys())
    execution_config.datasets_to_run = list(DATASETS.keys())
    execution_config.search_methods_to_run = list(SEARCH_METHODS.keys())
    print("🚀 Full experiment configuration applied!")

def single_model_config(model_name: str):
    """Configure for single model experiments"""
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODELS.keys())}")
    execution_config.models_to_run = [model_name]
    print(f"🎯 Single model configuration applied for {MODELS[model_name].name}!")

def single_dataset_config(dataset_name: str):
    """Configure for single dataset experiments"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")
    execution_config.datasets_to_run = [dataset_name]
    print(f"📚 Single dataset configuration applied for {DATASETS[dataset_name].name}!")

def use_alternative_models():
    """Switch to smaller, more accessible models for testing"""
    execution_config.models_to_run = list(ALTERNATIVE_MODELS.keys())
    print("🔧 Switched to alternative smaller models:")
    for model_name in execution_config.models_to_run:
        model_config = ALTERNATIVE_MODELS[model_name]
        print(f"   • {model_config.name} ({model_config.model_id})")
    print("💡 These models are smaller and more likely to work with limited disk space")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Validate configuration
    try:
        validate_config()
        print(get_experiment_summary())
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        exit(1)