# SSDP: Semantic Similarity based Dynamic Pruning

SSDP is an advanced tree search algorithm that combines parallel path exploration with intelligent pruning based on comprehensive scoring from the `scoring.py` system.

## Key Features

🌳 **Parallel Path Exploration**: Explores multiple reasoning paths simultaneously
🎯 **Dynamic Pruning**: Removes low-scoring paths based on overall scores
🔄 **Semantic Merging**: Combines similar reasoning paths to reduce redundancy  
📊 **Comprehensive Scoring**: Uses the full scoring.py system (confidence + future components)
🚀 **Adaptive Expansion**: Adjusts exploration budget based on path quality
🚫 **No Verifier Dependency**: Uses scoring.py instead of separate verifier model

## Algorithm Overview

1. **Expansion**: Generate multiple reasoning steps for the best nodes
2. **Scoring**: Evaluate each node using the comprehensive scoring.py system
3. **Merging**: Combine semantically similar nodes at the same level
4. **Pruning**: Remove nodes below overall score threshold
5. **Selection**: Choose best nodes for next iteration

## Scoring Integration

SSDP is fully integrated with your `scorer/scoring.py` system:

- **Primary Score**: `get_overall_answer_score()` for node ranking
- **Detailed Analysis**: Full score breakdown available for each node
- **Future Ready**: Automatically benefits from new scoring components
- **No Verifier**: Eliminates dependency on separate verifier model

```python
# Each node gets comprehensive scoring
node.overall_score = get_overall_answer_score(question, path)
node.detailed_scores = scorer.get_overall_score(question, path)
```

## Files

- `SSDP.py` - Main algorithm implementation (verifier-free)
- `config.py` - Configuration parameters  
- `eval_ssdp.py` - Evaluation and analysis tools
- `README.md` - This documentation

## Usage

### Basic Usage
```bash
cd /workspace/Fetch/search/SSDP
python SSDP.py
```

### Evaluation
```bash
python eval_ssdp.py test_gsm8k_ssdp_p8_t0.3.pkl results.json
```

## Configuration

Key parameters in `config.py`:

- `MAX_PARALLEL_PATHS`: Number of parallel reasoning paths (default: 8)
- `OVERALL_SCORE_THRESHOLD`: Minimum overall score to keep a path (default: 0.3)
- `SIMILARITY_THRESHOLD`: Threshold for merging similar nodes (default: 0.85)
- `PRUNE_FREQUENCY`: How often to prune (default: every 3 iterations)

## Algorithm Details

### Node Scoring
Each node receives comprehensive scoring from `scoring.py`:
- **Overall Score**: Primary ranking metric (0-1)
- **Confidence Score**: Confidence component for analysis
- **Detailed Breakdown**: Full component scores when available
- **Future Components**: Automatically included as scoring.py expands

### Semantic Similarity
Uses TF-IDF vectors and cosine similarity:
- Nodes with similarity > threshold are merged
- Better scoring node is kept as primary
- Merged node information is preserved

### Dynamic Pruning
- Removes nodes below overall score threshold
- Triggered every N iterations
- Focuses search on promising paths

### Adaptive Expansion
Expansion budget based on overall score:
- High score (≥0.8): 5 expansions
- Medium-high score (≥0.6): 4 expansions  
- Medium score (≥0.4): 4 expansions
- Low score (<0.4): 3 expansions

## Comparison with Other Algorithms

| Algorithm | Exploration | Pruning | Merging | Scoring |
|-----------|-------------|---------|---------|---------|
| Beam Search | Fixed width | No | No | Verifier only |
| MCTS | UCB-guided | No | No | Verifier only |
| **SSDP** | **Adaptive** | **Dynamic** | **Semantic** | **Comprehensive** |

## Benefits of Scoring.py Integration

1. **🔄 Future-Proof**: Automatically benefits from new scoring components
2. **🎯 Unified Scoring**: Single scoring system across all algorithms
3. **📊 Rich Analysis**: Detailed score breakdowns for every node
4. **🚫 Simplified**: No need to manage separate verifier model
5. **⚡ Flexible**: Easy to adjust scoring weights and components

## Performance Tips

1. **Adjust thresholds** based on your dataset and scoring.py performance
2. **Tune expansion budget** for speed vs quality tradeoff
3. **Monitor score distributions** to optimize thresholds
4. **Use custom weights** in scoring.py for domain-specific optimization

## Example Score Analysis

```python
# Access detailed scoring for any node
node = tree.get_best_terminal_node()
breakdown = node.get_score_breakdown()

print(f"Overall Score: {breakdown['overall_score']:.3f}")
print(f"Confidence: {breakdown['confidence_score']:.3f}")
print("Component Scores:")
for component, data in breakdown['component_scores'].items():
    print(f"  {component}: {data['score']:.3f}")
```

## Future Enhancements

As your `scoring.py` system grows, SSDP will automatically benefit from:
- **Parent/child quality scoring** for tree-aware evaluation
- **Semantic similarity scoring** for better merging decisions  
- **Coherence scoring** for reasoning quality
- **Factual consistency** checking
- **Custom domain-specific** scoring components

SSDP provides a robust, extensible search framework that grows with your scoring system! 