"""
Evaluation script for SSDP search results.
Updated to work with scoring.py system instead of verifier model.
"""

import pickle
import json
import re
from typing import List, Dict

# Import SSDP classes so pickle can reconstruct the objects
from SSDP import SSDPTree, SSDPNode

def extract_answer(text: str) -> str:
    """Extract the final answer from reasoning text."""
    # Look for "The answer is X" pattern
    pattern = r"The answer is\s*([+-]?\d*\.?\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Fallback: look for numbers at the end
    numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
    if numbers:
        return numbers[-1]
    
    return "[invalid]"

def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from the dataset format."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    
    # Try to extract number
    pattern = r"([+-]?\d*\.?\d+)"
    match = re.search(pattern, answer_text)
    if match:
        return match.group(1)
    
    return "[invalid]"

def evaluate_ssdp_results(results_file: str, output_file: str = None):
    """Evaluate SSDP search results."""
    
    print(f"Loading results from {results_file}")
    with open(results_file, "rb") as f:
        problems = pickle.load(f)
    
    total_problems = len(problems)
    correct = 0
    finished = 0
    total_nodes = 0
    total_expansions = 0
    total_merges = 0
    total_prunes = 0
    
    # Score distribution analysis
    overall_scores = []
    confidence_scores = []
    
    results = []
    
    for i, problem in enumerate(problems):
        # Get problem info
        question = problem.question
        gold_answer = extract_gold_answer(problem.answer)
        
        # Find best solution
        best_terminal = problem.get_best_terminal_node()
        
        if best_terminal:
            finished += 1
            prediction_text = best_terminal.print_path()
            predicted_answer = extract_answer(prediction_text)
            
            # Check correctness
            is_correct = predicted_answer == gold_answer
            if is_correct:
                correct += 1
            
            # Collect scores for analysis
            overall_scores.append(best_terminal.overall_score)
            confidence_scores.append(best_terminal.confidence_score)
            
            result = {
                "problem_id": i,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "prediction_text": prediction_text,
                "is_correct": is_correct,
                "overall_score": best_terminal.overall_score,
                "confidence_score": best_terminal.confidence_score,
                "depth": best_terminal.get_depth(),
                "merged_nodes": len(best_terminal.merged_nodes),
                "score_breakdown": best_terminal.get_score_breakdown()
            }
        else:
            result = {
                "problem_id": i,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "[no_solution]",
                "prediction_text": "",
                "is_correct": False,
                "overall_score": 0.0,
                "confidence_score": 0.0,
                "depth": 0,
                "merged_nodes": 0,
                "score_breakdown": {}
            }
        
        results.append(result)
        
        # Aggregate statistics
        total_nodes += len(problem.all_nodes)
        total_expansions += problem.total_expansions
        total_merges += problem.total_merges
        total_prunes += problem.total_prunes
    
    # Print results
    accuracy = correct / total_problems if total_problems > 0 else 0
    completion_rate = finished / total_problems if total_problems > 0 else 0
    
    print("\n=== SSDP Evaluation Results ===")
    print(f"Total problems: {total_problems}")
    print(f"Completed problems: {finished}")
    print(f"Correct solutions: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Completion rate: {completion_rate:.3f}")
    
    print(f"\n=== Search Statistics ===")
    print(f"Total nodes created: {total_nodes}")
    print(f"Average nodes per problem: {total_nodes / total_problems:.1f}")
    print(f"Total expansions: {total_expansions}")
    print(f"Total merges: {total_merges}")
    print(f"Total prunes: {total_prunes}")
    print(f"Merge rate: {total_merges / total_expansions:.3f}" if total_expansions > 0 else "N/A")
    print(f"Prune rate: {total_prunes / total_nodes:.3f}" if total_nodes > 0 else "N/A")
    
    # Score analysis
    if overall_scores:
        print(f"\n=== Scoring Analysis ===")
        print(f"Overall Score - Mean: {sum(overall_scores)/len(overall_scores):.3f}, "
              f"Min: {min(overall_scores):.3f}, Max: {max(overall_scores):.3f}")
        print(f"Confidence Score - Mean: {sum(confidence_scores)/len(confidence_scores):.3f}, "
              f"Min: {min(confidence_scores):.3f}, Max: {max(confidence_scores):.3f}")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "total_problems": total_problems,
                    "correct": correct,
                    "finished": finished,
                    "accuracy": accuracy,
                    "completion_rate": completion_rate,
                    "total_nodes": total_nodes,
                    "total_expansions": total_expansions,
                    "total_merges": total_merges,
                    "total_prunes": total_prunes,
                    "average_overall_score": sum(overall_scores)/len(overall_scores) if overall_scores else 0,
                    "average_confidence_score": sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0
                },
                "detailed_results": results
            }, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval_ssdp.py <results_file.pkl> [output_file.json]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    evaluate_ssdp_results(results_file, output_file)
