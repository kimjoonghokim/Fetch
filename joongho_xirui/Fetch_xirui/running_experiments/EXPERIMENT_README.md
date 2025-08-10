# 🧪 Experiment Runner for Tree-of-Thoughts Research

This guide explains how to run experiments for your **"Tree-of-Thoughts with State Merging for Mathematical Reasoning"** research paper.

## 🚀 Quick Start

### 1. Run All Experiments
```bash
cd /workspace/joongho_xirui/Fetch/running_experiments
python3 run_experiments.py
```

### 2. Run Specific Algorithm
```bash
# Test only BFS
python3 run_experiments.py --algorithm BFS

# Test only MCTS
python3 run_experiments.py --algorithm MCTS

# Test only Beam Search
python3 run_experiments.py --algorithm BeamSearch

# Test only Beam Search with Merging
python3 run_experiments.py --algorithm BeamSearchMerge
```

### 3. Custom Dataset
```bash
python3 run_experiments.py --dataset ../path/to/your/dataset.jsonl
```

## 📊 What Gets Tested

### **BFS (Breadth-First Search)**
- **Parameters**: Max depth, budget, temperature
- **What it measures**: How deep the search goes, computational efficiency
- **Random scoring**: Each node gets a random value 0-1

### **MCTS (Monte Carlo Tree Search)**
- **Parameters**: Number of rollouts, time limits, exploration constant
- **What it measures**: Exploration vs. exploitation balance
- **Random scoring**: Each state gets a random value 0-1

### **Beam Search**
- **Parameters**: Beam size, budget, temperature
- **What it measures**: How many promising paths to maintain
- **Random scoring**: Each node gets a random value 0-1

### **Beam Search + State Merging**
- **Parameters**: Beam size, budget, merge threshold
- **What it measures**: Impact of clustering similar reasoning steps
- **Random scoring**: Each node gets a random value 0-1

## ⚙️ Configuration

Edit `experiment_config.py` to customize:
- **Parameter ranges** to test
- **Dataset settings**
- **Output options**
- **Experiment schedules**

## 📁 Output Files

Results are saved in `experiment_results/`:
- `experiment_results_YYYYMMDD_HHMMSS.pkl` - Raw results (pickle format)
- `experiment_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary

## 🔬 Research Questions You Can Answer

1. **Algorithm Comparison**: Which search algorithm performs best with random scoring?
2. **Parameter Sensitivity**: How do different parameters affect performance?
3. **State Merging Impact**: Does clustering similar reasoning steps improve efficiency?
4. **Random Scoring Baseline**: How well do random scores work compared to learned value functions?

## 📈 Example Results Analysis

```python
import pickle

# Load results
with open('experiment_results/experiment_results_20241201_143022.pkl', 'rb') as f:
    results = pickle.load(f)

# Analyze BFS performance
bfs_results = results['BFS']
for depth, exp_results in bfs_results.items():
    avg_value = sum(r['best_value'] for r in exp_results) / len(exp_results)
    avg_time = sum(r['time_taken'] for r in exp_results) / len(exp_results)
    print(f"BFS {depth}: Avg Value={avg_value:.3f}, Avg Time={avg_time:.2f}s")
```

## 🎯 Next Steps for Your Paper

1. **Run baseline experiments** with random scoring (what you have now)
2. **Implement your actual value function** to replace random scoring
3. **Compare performance** between random and learned scoring
4. **Analyze state merging** impact on search efficiency
5. **Generate plots and tables** for your paper

## 🐛 Troubleshooting

### Policy Server Not Running
```bash
# Start vLLM server (if you have GPU)
python3 -m vllm.entrypoints.openai.api_server \
    --model xmu-nlp/Llama-3-8b-gsm8k \
    --port 8000
```

### Clustering Server Not Running
```bash
cd cluster
python3 server_cluster.py
```

### Memory Issues
- Reduce `max_problems` in config
- Use smaller parameter ranges
- Run experiments separately

## 📚 Files Overview

- `run_experiments.py` - Main experiment runner
- `experiment_config.py` - Configuration parameters
- `search/bfs/` - BFS implementation with random scoring
- `search/mcts/` - MCTS implementation with random scoring  
- `search/beamsearch/` - Beam search with random scoring
- `search/beamsearch/beamsearch_merge.py` - Beam search + state merging
- `dataset/toy.jsonl` - Sample mathematical problems

## 🎉 You're Ready!

Your research pipeline is complete with:
- ✅ Random scoring baseline implemented
- ✅ All search algorithms working
- ✅ State merging infrastructure ready
- ✅ Comprehensive experiment runner
- ✅ Results analysis tools

**Start running experiments and collecting data for your paper!** 🚀📊 