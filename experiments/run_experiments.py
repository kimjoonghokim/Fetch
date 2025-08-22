#!/usr/bin/env python3
"""
Example script showing how to use the config.py file to run experiments.
This script demonstrates loading datasets and models, and running experiments in parallel.
"""

import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import the configuration
from config import (
    experiment_config, verifier_config, scoring_config,
    execution_config, MODELS, DATASETS, SEARCH_METHODS,
    validate_config, get_experiment_summary,
    quick_test_config, full_experiment_config,
    load_dataset, load_model, get_dataset_sample
)

def run_single_experiment(args):
    """
    Run a single experiment combination.
    This function is designed to work with parallel execution.
    
    Args:
        args: Tuple of (model_name, dataset_name, search_method, question_index, question_data)
    """
    model_name, dataset_name, search_method, question_index, question_data = args
    
    print(f"\n🔬 Running experiment: {MODELS[model_name].name} + {DATASETS[dataset_name].name} + {SEARCH_METHODS[search_method].name} (Q{question_index})")
    
    # Record start time for latency measurement
    start_time = time.time()
    
    try:
        # Get the question from dataset
        question = question_data["question"]
        expected_answer = question_data["answer"]
        
        # TODO: Implement actual search algorithm execution here
        # This is where you would call your search methods (beamsearch, bfs, mcts, ssdp)
        
        # For now, simulate the search process
        if search_method == "ssdp":
            # SSDP uses scoring.py system
            print(f"  🔍 Using SSDP with scoring system for: {question[:100]}...")
            # Simulate SSDP execution
            time.sleep(0.5)
            final_answer = "42"  # Simulated answer
            answer_found = True
            search_steps = 12
            tokens_used = 180
            
        else:
            # Other methods use verifier model
            print(f"  🔍 Using {search_method} with verifier model for: {question[:100]}...")
            # Simulate verifier-based search
            time.sleep(0.8)
            final_answer = "42"  # Simulated answer
            answer_found = True
            search_steps = 15
            tokens_used = 200
        
        # Calculate execution time
        execution_time = time.time() - start_time
        latency_ms = execution_time * 1000
        
        # Determine accuracy (1 if answer found, 0 if not)
        accuracy = 1 if answer_found else 0
        
        # Create results dictionary
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "search_method": search_method,
            "question_index": question_index,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "expected_answer": expected_answer[:200] + "..." if len(expected_answer) > 200 else expected_answer,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "answer_found": answer_found,
            "final_answer": final_answer,
            "search_steps": search_steps,
            "status": "success",
            "config_used": {
                "model_config": MODELS[model_name].__dict__,
                "dataset_name": dataset_name,
                "search_method": search_method
            }
        }
        
        print(f"  ✅ Experiment completed successfully in {execution_time:.2f}s")
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  ❌ Experiment failed: {e}")
        
        # Return error results
        error_results = {
            "model": model_name,
            "dataset": dataset_name,
            "search_method": search_method,
            "question_index": question_index,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "error": str(e),
            "status": "failed",
            "tokens_used": 0,
            "latency_ms": execution_time * 1000,
            "accuracy": 0,
            "answer_found": False
        }
        return error_results

def run_single_experiment_with_model(args):
    """
    Run a single experiment with a pre-loaded model.
    This function is designed for sequential model loading.
    
    Args:
        args: Tuple of (model_name, dataset_name, search_method, question_index, question_data, model, tokenizer)
    """
    model_name, dataset_name, search_method, question_index, question_data, model, tokenizer = args
    
    print(f"\n🔬 Running experiment: {MODELS[model_name].name} + {DATASETS[dataset_name].name} + {SEARCH_METHODS[search_method].name} (Q{question_index})")
    
    # Record start time for latency measurement
    start_time = time.time()
    
    try:
        # Get the question from dataset
        question = question_data["question"]
        expected_answer = question_data["answer"]
        
        # TODO: Implement actual search algorithm execution here
        # This is where you would call your search methods (beamsearch, bfs, mcts, ssdp)
        # You now have access to the loaded model and tokenizer
        
        # For now, simulate the search process
        if search_method == "ssdp":
            # SSDP uses scoring.py system
            print(f"  🔍 Using SSDP with scoring system for: {question[:100]}...")
            # Simulate SSDP execution
            time.sleep(0.5)
            final_answer = "42"  # Simulated answer
            answer_found = True
            search_steps = 12
            tokens_used = 180
            
        else:
            # Other methods use verifier model
            print(f"  🔍 Using {search_method} with verifier model for: {question[:100]}...")
            # Simulate verifier-based search
            time.sleep(0.8)
            final_answer = "42"  # Simulated answer
            answer_found = True
            search_steps = 15
            tokens_used = 200
        
        # Calculate execution time
        execution_time = time.time() - start_time
        latency_ms = execution_time * 1000
        
        # Determine accuracy (1 if answer found, 0 if not)
        accuracy = 1 if answer_found else 0
        
        # Create results dictionary
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "search_method": search_method,
            "question_index": question_index,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "expected_answer": expected_answer[:200] + "..." if len(expected_answer) > 200 else expected_answer,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "answer_found": answer_found,
            "final_answer": final_answer,
            "search_steps": search_steps,
            "status": "success",
            "config_used": {
                "model_config": MODELS[model_name].__dict__,
                "dataset_name": dataset_name,
                "search_method": search_method
            }
        }
        
        print(f"  ✅ Experiment completed successfully in {execution_time:.2f}s")
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  ❌ Experiment failed: {e}")
        
        # Return error results
        error_results = {
            "model": model_name,
            "dataset": dataset_name,
            "search_method": search_method,
            "question_index": question_index,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "error": str(e),
            "status": "failed",
            "tokens_used": 0,
            "latency_ms": execution_time * 1000,
            "accuracy": 0,
            "answer_found": False
        }
        return error_results

def save_results(all_results: list, output_dir: str, output_format: str):
    """Save results to file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format.lower() == "json":
        output_file = output_path / f"experiment_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
        
    elif output_format.lower() == "pkl":
        output_file = output_path / f"experiment_results_{timestamp}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"💾 Results saved to: {output_file}")
    
    return output_file

def run_experiments():
    """Main function to run all configured experiments"""
    
    # Validate configuration first
    try:
        validate_config()
        print(get_experiment_summary())
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return
    
    # Run experiments with TRUE sequential model loading to save memory
    all_results = []
    
    if execution_config.load_models_sequentially:
        print(f"🔄 Running experiments with TRUE sequential model loading to save memory...")
        print(f"💾 This will load only ONE model at a time to save disk space")
        
        # Load datasets ONCE at the beginning (they're small)
        print(f"\n📚 Loading datasets once (they're small and reusable)...")
        datasets = {}
        for dataset_name in execution_config.datasets_to_run:
            dataset = load_dataset(dataset_name, execution_config.max_questions_per_dataset)
            if dataset is not None:
                datasets[dataset_name] = dataset
                print(f"✅ Dataset {dataset_name} loaded: {len(dataset)} questions")
            else:
                print(f"❌ Failed to load dataset: {dataset_name}")
                return
        
        if not datasets:
            print(f"❌ No datasets loaded, cannot continue")
            return
        
        # Process each model completely before moving to the next
        for model_name in execution_config.models_to_run:
            print(f"\n🤖 ==========================================")
            print(f"🤖 LOADING MODEL: {MODELS[model_name].name}")
            print(f"🤖 ==========================================")
            
            try:
                # Load the model
                print(f"🔄 Loading model: {model_name}")
                model, tokenizer = load_model(model_name)
                if model is None or tokenizer is None:
                    print(f"❌ Failed to load model {model_name}, skipping all experiments for this model")
                    continue
                print(f"✅ Model {model_name} loaded successfully")
                
                # Calculate experiments for this model
                total_questions = sum(len(dataset) for dataset in datasets.values())
                model_experiments = len(execution_config.datasets_to_run) * len(execution_config.search_methods_to_run) * total_questions
                
                print(f"📊 Running {model_experiments} experiments for model {model_name}")
                
                # Run all experiments for this model
                experiment_count = 0
                for dataset_name in execution_config.datasets_to_run:
                    if dataset_name not in datasets:
                        continue
                    dataset = datasets[dataset_name]
                    
                    for search_method in execution_config.search_methods_to_run:
                        for question_index in range(len(dataset)):
                            experiment_count += 1
                            print(f"\n📊 Progress: {experiment_count}/{model_experiments} (Model: {model_name})")
                            
                            try:
                                question_data = get_dataset_sample(dataset, question_index)
                                if question_data:
                                    # Add model and tokenizer to args for this experiment
                                    experiment_args = (model_name, dataset_name, search_method, question_index, question_data, model, tokenizer)
                                    results = run_single_experiment_with_model(experiment_args)
                                    all_results.append(results)
                                    
                            except Exception as e:
                                print(f"❌ Experiment failed: {e}")
                                # Add error result
                                error_results = {
                                    "model": model_name,
                                    "dataset": dataset_name,
                                    "search_method": search_method,
                                    "question_index": question_index,
                                    "timestamp": datetime.now().isoformat(),
                                    "error": str(e),
                                    "status": "failed",
                                    "tokens_used": 0,
                                    "latency_ms": 0,
                                    "accuracy": 0,
                                    "answer_found": False
                                }
                                all_results.append(error_results)
                            
                            # Save intermediate results if configured
                            if experiment_config.save_intermediate_results and experiment_count % 5 == 0:
                                print(f"💾 Saving intermediate results...")
                                save_results(all_results, experiment_config.output_dir, experiment_config.output_format)
                
                # Unload the model to free memory
                if execution_config.unload_model_after_use:
                    print(f"\n🗑️  Unloading model {model_name} to free memory...")
                    del model, tokenizer
                    import gc
                    gc.collect()
                    print(f"✅ Model {model_name} unloaded and memory freed")
                    
                    # Also clear Hugging Face cache for this model
                    try:
                        import shutil
                        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                        if cache_dir.exists():
                            # Find and remove this model's cache
                            for item in cache_dir.iterdir():
                                if model_name.replace("-", "_") in item.name or model_name.replace("-", "") in item.name:
                                    if item.is_dir():
                                        shutil.rmtree(item)
                                        print(f"🗑️  Cleared cache for {model_name}")
                                    elif item.is_file():
                                        item.unlink()
                                        print(f"🗑️  Cleared cache file for {model_name}")
                    except Exception as e:
                        print(f"⚠️  Could not clear cache: {e}")
                
            except Exception as e:
                print(f"❌ Error processing model {model_name}: {e}")
                continue
    
    else:
        # Original parallel execution logic (not recommended for limited disk space)
        print(f"⚠️  WARNING: Parallel execution enabled - this requires lots of disk space!")
        print(f"💡 Consider setting load_models_sequentially = True in config")
        
        # Load all datasets first (this uses lots of memory)
        print(f"\n📚 Loading ALL datasets...")
        datasets = {}
        for dataset_name in execution_config.datasets_to_run:
            dataset = load_dataset(dataset_name, execution_config.max_questions_per_dataset)
            if dataset is not None:
                datasets[dataset_name] = dataset
            else:
                print(f"❌ Failed to load dataset: {dataset_name}")
                return
        
        # Calculate total experiments
        total_questions = sum(len(dataset) for dataset in datasets.values())
        total_experiments = len(execution_config.models_to_run) * len(execution_config.datasets_to_run) * len(execution_config.search_methods_to_run) * total_questions
        
        print(f"\n🎯 This will run {total_experiments} experiments:")
        print(f"   • Models: {len(execution_config.models_to_run)}")
        print(f"   • Datasets: {len(execution_config.datasets_to_run)}")
        print(f"   • Search methods: {len(execution_config.search_methods_to_run)}")
        print(f"   • Questions per dataset: {total_questions}")
        
        response = input("Do you want to continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Experiment cancelled.")
            return
        
        # Prepare experiment arguments for parallel execution
        experiment_args = []
        
        for model_name in execution_config.models_to_run:
            for dataset_name in execution_config.datasets_to_run:
                for search_method in execution_config.search_methods_to_run:
                    dataset = datasets[dataset_name]
                    for question_index in range(len(dataset)):
                        question_data = get_dataset_sample(dataset, question_index)
                        if question_data:
                            experiment_args.append((model_name, dataset_name, search_method, question_index, question_data))
        
        print(f"\n🚀 Starting {len(experiment_args)} experiments...")
        
        if experiment_config.run_parallel and len(experiment_args) > 1:
            print(f"🔄 Running experiments in parallel with {experiment_config.max_workers} workers...")
            
            # Use ProcessPoolExecutor for parallel execution
            with ProcessPoolExecutor(max_workers=experiment_config.max_workers) as executor:
                # Submit all experiments
                future_to_args = {executor.submit(run_single_experiment, args): args for args in experiment_args}
                
                # Process completed experiments
                for i, future in enumerate(as_completed(future_to_args)):
                    args = future_to_args[future]
                    experiment_count = i + 1
                    
                    try:
                        results = future.result()
                        all_results.append(results)
                        print(f"📊 Progress: {experiment_count}/{len(experiment_args)} - {results['status']}")
                        
                    except Exception as e:
                        print(f"❌ Experiment failed: {e}")
                        # Add error result
                        model_name, dataset_name, search_method, question_index, _ = args
                        error_results = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "search_method": search_method,
                            "question_index": question_index,
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                            "status": "failed",
                            "tokens_used": 0,
                            "latency_ms": 0,
                            "accuracy": 0,
                            "answer_found": False
                        }
                        all_results.append(error_results)
                    
                    # Save intermediate results if configured
                    if experiment_config.save_intermediate_results and experiment_count % 10 == 0:
                        print(f"💾 Saving intermediate results...")
                        save_results(all_results, experiment_config.output_dir, experiment_config.output_format)
            
        else:
            print(f"🔄 Running experiments sequentially...")
            
            for i, args in enumerate(experiment_args):
                experiment_count = i + 1
                print(f"\n📊 Progress: {experiment_count}/{len(experiment_args)}")
                
                try:
                    results = run_single_experiment(args)
                    all_results.append(results)
                    
                except Exception as e:
                    print(f"❌ Experiment failed: {e}")
                    # Add error result
                    model_name, dataset_name, search_method, question_index, _ = args
                    error_results = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "search_method": search_method,
                        "question_index": question_index,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "status": "failed",
                        "tokens_used": 0,
                        "latency_ms": 0,
                        "accuracy": 0,
                        "answer_found": False
                    }
                    all_results.append(error_results)
                
                # Save intermediate results if configured
                if experiment_config.save_intermediate_results and experiment_count % 10 == 0:
                    print(f"💾 Saving intermediate results...")
                    save_results(all_results, experiment_config.output_dir, experiment_config.output_format)
    
    # Save final results
    print(f"\n💾 Saving final results...")
    output_file = save_results(all_results, experiment_config.output_dir, experiment_config.output_format)
    
    # Print summary
    successful_experiments = len([r for r in all_results if r.get('status') != 'failed'])
    failed_experiments = len([r for r in all_results if r.get('status') == 'failed'])
    
    print(f"\n🎉 EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successful: {successful_experiments}")
    print(f"❌ Failed: {failed_experiments}")
    print(f"📊 Total: {len(all_results)}")
    print(f"💾 Results saved to: {output_file}")
    
    # Print per-model/dataset summary
    print(f"\n📊 DETAILED SUMMARY")
    print(f"{'='*50}")
    
    for model_name in execution_config.models_to_run:
        for dataset_name in execution_config.datasets_to_run:
            for search_method in execution_config.search_methods_to_run:
                model_results = [r for r in all_results 
                               if r['model'] == model_name 
                               and r['dataset'] == dataset_name 
                               and r['search_method'] == search_method]
                
                if model_results:
                    successful = len([r for r in model_results if r.get('status') != 'failed'])
                    total = len(model_results)
                    avg_latency = sum(r.get('latency_ms', 0) for r in model_results if r.get('status') != 'failed') / max(successful, 1)
                    avg_tokens = sum(r.get('tokens_used', 0) for r in model_results if r.get('status') != 'failed') / max(successful, 1)
                    
                    print(f"🔬 {MODELS[model_name].name} + {DATASETS[dataset_name].name} + {SEARCH_METHODS[search_method].name}")
                    print(f"   ✅ Success: {successful}/{total} ({successful/total*100:.1f}%)")
                    print(f"   ⏱️  Avg Latency: {avg_latency:.1f}ms")
                    print(f"   🎯 Avg Tokens: {avg_tokens:.1f}")
                    print()

def main():
    """Main entry point with configuration options"""
    print("🔬 Math Reasoning Experiments Runner")
    print("="*50)
    
    while True:
        print("\n📋 Available options:")
        print("1. Run quick test (5 questions, 1 model, 1 dataset, 1 search method)")
        print("2. Run full experiments (all combinations)")
        print("3. Show current configuration")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            print("\n🔧 Applying quick test configuration...")
            quick_test_config()
            run_experiments()
            break
            
        elif choice == "2":
            print("\n🚀 Applying full experiment configuration...")
            full_experiment_config()
            run_experiments()
            break
            
        elif choice == "3":
            print(get_experiment_summary())
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()