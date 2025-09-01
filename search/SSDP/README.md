# SSDP: Semantic Similarity based Dynamic Pruning

This document provides an overview of the new and improved SSDP algorithm, a sophisticated search strategy for solving complex problems. This implementation is designed to be efficient, powerful, and easy to use, even for those who are new to the concepts of tree-of-thoughts search.

## What is SSDP?

At its core, SSDP is a search algorithm that explores a "tree of thoughts" to find the best solution to a problem. It starts with an initial thought (the problem statement) and then explores different lines of reasoning (branches of the tree) to find the best possible answer.

What makes this implementation of SSDP special is that it is designed to be highly efficient and intelligent. It uses a combination of advanced techniques to explore the search space in a smart way, which saves time and computational resources.

### Key Features

*   **No Verifier Model:** Unlike other search algorithms, SSDP does not require a separate "verifier" model to score the different lines of reasoning. Instead, it uses a sophisticated, hardcoded scoring algorithm that is based on a combination of factors, including the confidence of the language model, the score of the parent thought, and a "voting" mechanism based on semantic similarity.
*   **Semantic Merging:** SSDP is smart enough to recognize when it is exploring the same line of reasoning multiple times. It uses an embedding model to identify semantically similar thoughts at the same level of the tree and merges them into a single, more confident thought. This prevents the algorithm from wasting time on redundant exploration.
*   **Advanced Pruning:** SSDP uses a combination of advanced pruning techniques to eliminate unpromising paths early on. This includes:
    *   **Score-based pruning:** Paths with low scores are pruned.
    *   **Heuristic pruning:** Paths that are too long, contain repetitive phrases, or are not relevant to the original question are pruned.
    *   **Depth-aware and budget-aware pruning:** The pruning is more aggressive at deeper levels of the tree and when the search is approaching its computational budget.
*   **Explore/Exploit Status:** Each line of reasoning is labeled as either "explore" or "exploit." This allows the algorithm to be more flexible and to backtrack from unpromising paths.
*   **Early Stopping:** The algorithm will automatically stop early if it is no longer making significant progress, which saves time and resources.

## How to Run SSDP

Running the SSDP algorithm is a two-step process:

1.  **Run the Search:** First, you need to run the `SSDP.py` script to perform the search and generate a results file. This file will contain all the information about the search process, including all the thoughts that were explored and their scores.

    ```bash
    python SSDP.py
    ```

2.  **Evaluate the Results:** After the search is complete, you can use the `eval_ssdp.py` script to analyze the results and get a detailed report on the performance of the algorithm.

    ```bash
    python eval_ssdp.py <results_file.pkl>
    ```

    Replace `<results_file.pkl>` with the name of the results file that was generated in the previous step (e.g., `test_gsm8k_ssdp_v2.pkl`).

## How to Configure SSDP

All of the parameters for the SSDP algorithm can be configured in the `config.py` file. This file is well-commented and easy to understand, even for novices. Here is an overview of the most important parameters:

*   **`LIMIT`:** The maximum number of iterations for the search.
*   **`MAX_DEPTH`:** The maximum depth of the search tree.
*   **`OVERALL_SCORE_THRESHOLD`:** The minimum score for a path to be considered promising. This is the most important parameter for controlling the trade-off between performance and accuracy.
*   **`SIMILARITY_THRESHOLD`:** The threshold for merging semantically similar nodes.
*   **`MAX_PARALLEL_PATHS`:** The maximum number of paths to explore in parallel.
*   **`MIN_EXPANSION_BUDGET` and `MAX_EXPANSION_BUDGET`:** The minimum and maximum number of times to expand each node.

By tuning these parameters, you can find the optimal settings for your specific needs.

## Conclusion

This new and improved implementation of the SSDP algorithm is a powerful and efficient tool for solving complex problems. It is designed to be easy to use and configure, even for novices. I am confident that you will find it to be a valuable addition to your toolkit.