#!/usr/bin/env python3
"""
Test script to verify sequential model loading is working correctly.
This will help debug the disk space issue.
"""

import os
import gc
from pathlib import Path
from config import (
    execution_config, MODELS, DATASETS, SEARCH_METHODS,
    load_dataset, load_model, get_dataset_sample
)

def check_disk_space():
    """Check current disk usage"""
    print("💾 CHECKING DISK SPACE")
    print("="*50)
    
    try:
        import subprocess
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        print("📊 Current disk usage:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Could not check disk usage: {e}")

def test_sequential_loading():
    """Test loading models one at a time"""
    print("\n🧪 TESTING SEQUENTIAL MODEL LOADING")
    print("="*60)
    
    if not execution_config.load_models_sequentially:
        print("❌ Sequential loading is DISABLED in config!")
        print("💡 Set load_models_sequentially = True in config.py")
        return False
    
    print("✅ Sequential loading is ENABLED in config")
    
    # Test with just 2 models to save time
    test_models = execution_config.models_to_run[:2]
    print(f"🧪 Testing with {len(test_models)} models: {test_models}")
    
    for i, model_name in enumerate(test_models):
        print(f"\n🤖 ==========================================")
        print(f"🤖 TESTING MODEL {i+1}/{len(test_models)}: {MODELS[model_name].name}")
        print(f"🤖 ==========================================")
        
        # Check disk space before loading
        print(f"💾 Disk space BEFORE loading {model_name}:")
        check_disk_space()
        
        try:
            # Load the model
            print(f"🔄 Loading model: {model_name}")
            model, tokenizer = load_model(model_name)
            
            if model is None or tokenizer is None:
                print(f"❌ Failed to load model {model_name}")
                continue
                
            print(f"✅ Model {model_name} loaded successfully")
            
            # Check disk space after loading
            print(f"💾 Disk space AFTER loading {model_name}:")
            check_disk_space()
            
            # Simulate some work
            print(f"🔬 Simulating experiments with {model_name}...")
            import time
            time.sleep(2)  # Simulate work
            
            # Unload the model
            if execution_config.unload_model_after_use:
                print(f"🗑️  Unloading model {model_name}...")
                del model, tokenizer
                gc.collect()
                print(f"✅ Model {model_name} unloaded")
                
                # Check disk space after unloading
                print(f"💾 Disk space AFTER unloading {model_name}:")
                check_disk_space()
            else:
                print(f"⚠️  Model unloading is DISABLED - this will use more memory!")
                
        except Exception as e:
            print(f"❌ Error testing model {model_name}: {e}")
            continue
    
    print(f"\n🎉 Sequential loading test completed!")
    return True

def test_dataset_loading():
    """Test loading datasets"""
    print("\n📚 TESTING DATASET LOADING")
    print("="*50)
    
    # Test with just 1 dataset to save time
    test_datasets = execution_config.datasets_to_run[:1]
    print(f"🧪 Testing with {len(test_datasets)} dataset: {test_datasets}")
    
    for dataset_name in test_datasets:
        print(f"\n📚 Testing dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, 5)  # Just 5 questions for testing
            if dataset is not None:
                print(f"✅ Dataset {dataset_name} loaded successfully: {len(dataset)} questions")
                
                # Test getting a sample
                sample = get_dataset_sample(dataset, 0)
                if sample:
                    print(f"✅ Sample question: {sample['question'][:100]}...")
                else:
                    print(f"❌ Failed to get sample from dataset")
            else:
                print(f"❌ Failed to load dataset {dataset_name}")
                
        except Exception as e:
            print(f"❌ Error testing dataset {dataset_name}: {e}")

def main():
    """Main test function"""
    print("🚀 SEQUENTIAL LOADING TEST")
    print("="*60)
    
    # Check initial disk space
    check_disk_space()
    
    # Test dataset loading first
    test_dataset_loading()
    
    # Test sequential model loading
    success = test_sequential_loading()
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print(f"💡 If you still get disk space errors, the issue might be:")
        print(f"   1. Hugging Face cache is too large")
        print(f"   2. Other processes are using disk space")
        print(f"   3. Network volume setup needed")
    else:
        print(f"\n❌ Tests failed - check the error messages above")

if __name__ == "__main__":
    main()
