# 🌳 SSDP: Semantic Similarity–based Dynamic Pruning

**Goal:**  
Efficiently explore a reasoning tree by balancing **exploration** (diverse paths) and **exploitation** (high-confidence paths), while avoiding redundant computation through semantic similarity clustering and dynamic pruning.

---

## 🔹 Core Ideas

1. **Batch Expansion (B)**  
Generate up to B candidate child nodes at each level.  
Use embeddings to measure semantic similarity among candidates.

2. **Clustering & Merging (Intra-Cluster)**  
Cluster semantically similar nodes.  
Merge each cluster into a representative node:  
- Keep the highest-confidence node.  
- Add a similarity bonus proportional to merged siblings’ scores.

3. **Leader Node & Diversity Reward (Inter-Cluster)**  
Leader = highest-scoring cluster representative.  
Other clusters sufficiently different from leader receive a diversity reward.  
Diversity reward decays with depth: strong early, weaker deeper.

4. **Node Scoring Function**  

Each node’s score is computed as:
```
score(node) = confidence + similarity_bonus (intra-cluster) + diversity_reward (inter-cluster) + parent_score
```


5. **Dynamic Window Width (N)**  
Window width = number of clusters ≤ N.  
If fewer than N clusters exist → window narrows naturally.  
Ensures the window adapts to problem complexity.

---

## 🔹 Expansion Budget Allocation

- Allocate expansions per node proportional to its score relative to siblings.  
- Ensures promising but diverse nodes get more chance to expand.

---

## 🔹 Hybrid Pruning Strategy

**Goal:** Reduce wasted computation while balancing exploration/exploitation.

**Step 1: Compute thresholds**  
- Relative-to-leader threshold: `α * leader_score` (preserve diversity early).  
- Depth-scaled minimum: `min_score(d) = β + γ * depth` (tighter pruning deeper).

**Step 2: Apply hybrid pruning**

keep node if score(node) >= max(α * leader_score, min_score(d))



- Early levels → relative-to-leader dominates → keeps diverse paths.  
- Later levels → depth-scaled dominates → focuses on high-confidence exploitation.

---

## 🔹 Stopping Criteria

Stop expanding if any of the following are met:

1. High-confidence terminal node: score ≥ threshold (e.g., 0.9).  
2. Terminal convergence: new terminals don’t improve best score by more than δ.  
3. Max terminal nodes reached: e.g., 5–10.  
4. Window collapse: window shrinks to 1 cluster for L consecutive levels.

---

## 🔹 Efficiency Guidelines

- Batch embeddings for all candidates per level.  
- Approximate clustering using centroids rather than full pairwise comparisons.  
- Recommended B ≈ 3–5 × N.  
- Accept fewer than N clusters as a natural indication of low complexity.

---

## 🔹 Benefits

- Adaptive exploration/exploitation: window adjusts naturally.  
- Redundant path elimination: similarity-based merging reduces wasted compute.  
- Dynamic pruning: hybrid threshold ensures efficient, depth-aware pruning.  
- Flexible: works with any model that can provide confidence scores and embeddings.

---

## 🔹 High-Level Flow

1. Generate B candidates at current level.  
2. Cluster semantically similar nodes → merge → add similarity bonus.  
3. Score representatives (confidence + similarity bonus + diversity + parent score).  
4. Keep top N clusters → assign diversity rewards relative to leader.  
5. Apply hybrid pruning (relative-to-leader + depth-scaled cutoff).  
6. Allocate expansions proportionally to scores.  
7. Repeat until stopping criteria are met.

---

⚡ **In short:**  
SSDP = *tree search with semantic clustering, dynamic scoring, diversity incentives, and depth-aware pruning*.  
It efficiently explores reasoning paths, balances exploration/exploitation, and adapts to problem complexity.