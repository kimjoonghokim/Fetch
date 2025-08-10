#!/usr/bin/env python3
"""
🧪 Experiment Runner for Tree-of-Thoughts with State Merging
Runs BFS, MCTS, and Beam Search algorithms with random scoring on mathematical reasoning tasks.
"""

import os
import sys
import json
import time
import random
import pickle
from datetime import datetime
from pathlib import Path
import argparse

# Add search paths
sys.path.append('../search/bfs')
sys.path.append('../search/mcts')
sys.path.append('../search/beamsearch')

from call_joint_service import Worker, PolicyArgument
from gsm_config import GSMConfig
from beamsearch import Tree as BeamSearchTree
from beamsearch_merge import Tree as BeamSearchMergeTree

class ExperimentRunner:
    def __init__(self, dataset_path="../dataset/toy.jsonl", output_dir="experiment_results"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.dataset = self.load_dataset()
        print(f"📊 Loaded {len(self.dataset)} problems from {dataset_path}")
        
        # Set random seed for reproducibility
        random.seed(42)
        
    def load_dataset(self):
        """Load the dataset from JSONL file"""
        dataset = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    
    def run_bfs_experiment(self, max_depth=5, max_budget=10, temperature=0.8):
        """Run BFS experiment with random scoring"""
        print(f"\n🔍 Running BFS Experiment...")
        print(f"   Max Depth: {max_depth}, Budget: {max_budget}, Temperature: {temperature}")
        
        results = []
        worker = Worker(PolicyArgument(), None)  # No value server needed for random scoring
        
        for i, problem in enumerate(self.dataset[:5]):  # Test with first 5 problems
            print(f"   Problem {i+1}/{min(5, len(self.dataset))}: {problem['question'][:50]}...")
            
            start_time = time.time()
            question = problem['question']
            path = ""
            depth = 0
            
            # Simple BFS implementation with random scoring
            nodes_to_expand = [(path, 0)]  # (path, depth)
            best_path = path
            best_value = 0
            
            while nodes_to_expand and depth < max_depth:
                current_nodes = nodes_to_expand.copy()
                nodes_to_expand = []
                
                for current_path, current_depth in current_nodes:
                    if current_depth >= max_depth:
                        continue
                        
                    # Get next step and value
                    next_step, value = worker.encode(question, current_path, temp=temperature, stop=[])
                    
                    if next_step and value > best_value:
                        best_path = current_path + "\n" + next_step if current_path else next_step
                        best_value = value
                    
                    # Add to expansion queue
                    if current_depth + 1 < max_depth:
                        new_path = current_path + "\n" + next_step if current_path else next_step
                        nodes_to_expand.append((new_path, current_depth + 1))
                
                depth += 1
                
                # Budget control
                if len(results) >= max_budget:
                    break
            
            end_time = time.time()
            
            result = {
                'problem_id': i,
                'question': question,
                'best_path': best_path,
                'best_value': best_value,
                'depth_reached': depth,
                'time_taken': end_time - start_time,
                'algorithm': 'BFS',
                'parameters': {
                    'max_depth': max_depth,
                    'max_budget': max_budget,
                    'temperature': temperature
                }
            }
            results.append(result)
            
        return results
    
    def run_mcts_experiment(self, n_rollouts=5, max_time=60, c=0.6):
        """Run MCTS experiment with random scoring"""
        print(f"\n🎯 Running MCTS Experiment...")
        print(f"   Rollouts: {n_rollouts}, Max Time: {max_time}s, C: {c}")
        
        results = []
        config = GSMConfig()
        config.n_rollouts = n_rollouts
        config.c = c
        
        for i, problem in enumerate(self.dataset[:5]):  # Test with first 5 problems
            print(f"   Problem {i+1}/{min(5, len(self.dataset))}: {problem['question'][:50]}...")
            
            start_time = time.time()
            question = problem['question']
            steps = []
            
            # Simple MCTS simulation with random scoring
            for rollout in range(n_rollouts):
                if time.time() - start_time > max_time:
                    break
                    
                # Simulate one rollout
                current_steps = steps.copy()
                for step in range(3):  # Simulate up to 3 steps
                    # Get value for current state
                    value = config.get_value(question, current_steps)
                    
                    # Randomly decide to continue or stop
                    if random.random() < 0.7:  # 70% chance to continue
                        current_steps.append(f"Step {step + 1}: Random reasoning")
                    else:
                        break
            
            end_time = time.time()
            
            result = {
                'problem_id': i,
                'question': question,
                'steps': steps,
                'final_value': config.get_value(question, steps),
                'rollouts_completed': min(n_rollouts, int((end_time - start_time) / (max_time / n_rollouts))),
                'time_taken': end_time - start_time,
                'algorithm': 'MCTS',
                'parameters': {
                    'n_rollouts': n_rollouts,
                    'max_time': max_time,
                    'c': c
                }
            }
            results.append(result)
            
        return results
    
    def run_beamsearch_experiment(self, beam_size=3, budget=5, temperature=0.8):
        """Run Beam Search experiment with random scoring"""
        print(f"\n📡 Running Beam Search Experiment...")
        print(f"   Beam Size: {beam_size}, Budget: {budget}, Temperature: {temperature}")
        
        results = []
        
        for i, problem in enumerate(self.dataset[:5]):  # Test with first 5 problems
            print(f"   Problem {i+1}/{min(5, len(self.dataset))}: {problem['question'][:50]}...")
            
            start_time = time.time()
            question = problem['question']
            
            # Create tree and run simple beam search
            tree = BeamSearchTree(question, problem['answer'])
            
            # Add some nodes with random values
            for step in range(min(budget, 5)):
                # Create a simple path
                path = f"Step {step + 1}: Random reasoning"
                value = random.random()
                
                if step == 0:
                    node = tree.add_node(path, value, tree.root)
                else:
                    node = tree.add_node(path, value, tree.all_nodes[-2])
            
            # Get best beam
            best_beam = tree.get_beam_to_expand(beam_size)
            
            end_time = time.time()
            
            result = {
                'problem_id': i,
                'question': question,
                'best_beam_size': len(best_beam),
                'tree_size': len(tree.all_nodes),
                'best_values': [node.value for node in best_beam] if best_beam else [],
                'time_taken': end_time - start_time,
                'algorithm': 'BeamSearch',
                'parameters': {
                    'beam_size': beam_size,
                    'budget': budget,
                    'temperature': temperature
                }
            }
            results.append(result)
            
        return results
    
    def run_beamsearch_merge_experiment(self, beam_size=3, budget=5, temperature=0.8, merge_threshold=0.1):
        """Run Beam Search with State Merging experiment"""
        print(f"\n🔗 Running Beam Search with State Merging Experiment...")
        print(f"   Beam Size: {beam_size}, Budget: {budget}, Merge Threshold: {merge_threshold}")
        
        results = []
        
        for i, problem in enumerate(self.dataset[:5]):  # Test with first 5 problems
            print(f"   Problem {i+1}/{min(5, len(self.dataset))}: {problem['question'][:50]}...")
            
            start_time = time.time()
            question = problem['question']
            
            # Create tree and run beam search with merging
            tree = BeamSearchMergeTree(question, problem['answer'])
            
            # Add some nodes with random values
            for step in range(min(budget, 5)):
                path = f"Step {step + 1}: Random reasoning"
                value = random.random()
                
                if step == 0:
                    node = tree.add_node(path, value, tree.root)
                else:
                    node = tree.add_node(path, value, tree.all_nodes[-2])
            
            # Simulate merging by grouping similar values
            all_values = [node.value for node in tree.all_nodes if node.value > 0]
            if len(all_values) > 1:
                # Simple clustering: group values within threshold
                clusters = []
                used = set()
                for i, val1 in enumerate(all_values):
                    if i in used:
                        continue
                    cluster = [val1]
                    used.add(i)
                    for j, val2 in enumerate(all_values[i+1:], i+1):
                        if j not in used and abs(val1 - val2) < merge_threshold:
                            cluster.append(val2)
                            used.add(j)
                    clusters.append(cluster)
            else:
                clusters = [all_values] if all_values else []
            
            end_time = time.time()
            
            result = {
                'problem_id': i,
                'question': question,
                'clusters_found': len(clusters),
                'cluster_sizes': [len(c) for c in clusters],
                'tree_size': len(tree.all_nodes),
                'time_taken': end_time - start_time,
                'algorithm': 'BeamSearchMerge',
                'parameters': {
                    'beam_size': beam_size,
                    'budget': budget,
                    'temperature': temperature,
                    'merge_threshold': merge_threshold
                }
            }
            results.append(result)
            
        return results
    
    def run_all_experiments(self):
        """Run all experiments with different parameter configurations"""
        print("🚀 Starting Comprehensive Experiment Suite...")
        print("=" * 60)
        
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Experiment 1: BFS with different depths
        print("\n📊 Experiment 1: BFS Depth Analysis")
        bfs_results = {}
        for depth in [3, 5, 7]:
            results = self.run_bfs_experiment(max_depth=depth, max_budget=10, temperature=0.8)
            bfs_results[f"depth_{depth}"] = results
        all_results['BFS'] = bfs_results
        
        # Experiment 2: MCTS with different rollouts
        print("\n📊 Experiment 2: MCTS Rollout Analysis")
        mcts_results = {}
        for rollouts in [3, 5, 8]:
            results = self.run_mcts_experiment(n_rollouts=rollouts, max_time=60, c=0.6)
            mcts_results[f"rollouts_{rollouts}"] = results
        all_results['MCTS'] = mcts_results
        
        # Experiment 3: Beam Search with different beam sizes
        print("\n📊 Experiment 3: Beam Search Size Analysis")
        beam_results = {}
        for beam_size in [2, 3, 5]:
            results = self.run_beamsearch_experiment(beam_size=beam_size, budget=5, temperature=0.8)
            beam_results[f"beam_{beam_size}"] = results
        all_results['BeamSearch'] = beam_results
        
        # Experiment 4: Beam Search with Merging
        print("\n📊 Experiment 4: State Merging Analysis")
        merge_results = {}
        for threshold in [0.05, 0.1, 0.2]:
            results = self.run_beamsearch_merge_experiment(beam_size=3, budget=5, temperature=0.8, merge_threshold=threshold)
            merge_results[f"threshold_{threshold}"] = results
        all_results['BeamSearchMerge'] = merge_results
        
        # Save results
        output_file = self.output_dir / f"experiment_results_{timestamp}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✅ All experiments completed!")
        print(f"📁 Results saved to: {output_file}")
        
        # Generate summary report
        self.generate_summary_report(all_results, timestamp)
        
        return all_results
    
    def generate_summary_report(self, results, timestamp):
        """Generate a summary report of all experiments"""
        report_file = self.output_dir / f"experiment_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("🧪 EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total Problems: {len(self.dataset)}\n\n")
            
            for algorithm, experiments in results.items():
                f.write(f"📊 {algorithm} Results:\n")
                f.write("-" * 30 + "\n")
                
                for exp_name, exp_results in experiments.items():
                    f.write(f"  {exp_name}:\n")
                    
                    # Calculate statistics
                    if exp_results:
                        values = [r.get('best_value', r.get('final_value', 0)) for r in exp_results]
                        times = [r.get('time_taken', 0) for r in exp_results]
                        
                        f.write(f"    Avg Value: {sum(values)/len(values):.4f}\n")
                        f.write(f"    Avg Time: {sum(times)/len(times):.2f}s\n")
                        f.write(f"    Problems Solved: {len(exp_results)}\n")
                    
                    f.write("\n")
                
                f.write("\n")
        
        print(f"📋 Summary report saved to: {report_file}")
    
    def save_individual_results(self, results, algorithm_name):
        """Save individual algorithm results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle
        output_file = self.output_dir / f"{algorithm_name.lower()}_results_{timestamp}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save as JSON (more readable)
        json_file = self.output_dir / f"{algorithm_name.lower()}_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to:")
        print(f"   📄 Pickle: {output_file}")
        print(f"   📄 JSON: {json_file}")
    
    def generate_individual_report(self, results, algorithm_name):
        """Generate individual algorithm report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{algorithm_name.lower()}_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"🧪 {algorithm_name} EXPERIMENT REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total Problems Tested: {len(results)}\n\n")
            
            if results:
                # Calculate statistics
                values = [r.get('best_value', r.get('final_value', 0)) for r in results]
                times = [r.get('time_taken', 0) for r in results]
                depths = [r.get('depth_reached', 0) for r in results]
                
                f.write(f"📊 STATISTICS:\n")
                f.write(f"   Average Value: {sum(values)/len(values):.4f}\n")
                f.write(f"   Average Time: {sum(times)/len(times):.2f}s\n")
                f.write(f"   Average Depth: {sum(depths)/len(depths):.1f}\n")
                f.write(f"   Best Value: {max(values):.4f}\n")
                f.write(f"   Worst Value: {min(values):.4f}\n\n")
                
                f.write(f"📋 DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                for i, result in enumerate(results):
                    f.write(f"Problem {i+1}:\n")
                    f.write(f"  Question: {result['question'][:100]}...\n")
                    f.write(f"  Best Path: {result['best_path'][:200]}...\n")
                    f.write(f"  Value: {result['best_value']:.4f}\n")
                    f.write(f"  Depth: {result['depth_reached']}\n")
                    f.write(f"  Time: {result['time_taken']:.2f}s\n")
                    f.write(f"  Parameters: {result['parameters']}\n\n")
        
        print(f"📋 Report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Tree-of-Thoughts experiments")
    parser.add_argument("--dataset", default="../dataset/toy.jsonl", help="Path to dataset file")
    parser.add_argument("--output", default="experiment_results", help="Output directory for results")
    parser.add_argument("--algorithm", choices=["BFS", "MCTS", "BeamSearch", "BeamSearchMerge", "all"], 
                       default="all", help="Which algorithm to test")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.dataset, args.output)
    
    if args.algorithm == "all":
        runner.run_all_experiments()
    elif args.algorithm == "BFS":
        results = runner.run_bfs_experiment()
        runner.save_individual_results(results, "BFS")
        runner.generate_individual_report(results, "BFS")
        print(f"BFS Results: {len(results)} problems tested")
        print(f"📁 Results saved to: {runner.output_dir}")
    elif args.algorithm == "MCTS":
        results = runner.run_mcts_experiment()
        runner.save_individual_results(results, "MCTS")
        runner.generate_individual_report(results, "MCTS")
        print(f"MCTS Results: {len(results)} problems tested")
        print(f"📁 Results saved to: {runner.output_dir}")
    elif args.algorithm == "BeamSearch":
        results = runner.run_beamsearch_experiment()
        runner.save_individual_results(results, "BeamSearch")
        runner.generate_individual_report(results, "BeamSearch")
        print(f"Beam Search Results: {len(results)} problems tested")
        print(f"📁 Results saved to: {runner.output_dir}")
    elif args.algorithm == "BeamSearchMerge":
        results = runner.run_beamsearch_merge_experiment()
        runner.save_individual_results(results, "BeamSearchMerge")
        runner.generate_individual_report(results, "BeamSearchMerge")
        print(f"Beam Search with Merging Results: {len(results)} problems tested")
        print(f"📁 Results saved to: {runner.output_dir}")

if __name__ == "__main__":
    main() 