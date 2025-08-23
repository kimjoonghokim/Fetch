#!/usr/bin/env python3
"""
Test script to verify the configuration system works correctly.
Run this to check if all configurations are valid and datasets can be loaded.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config import (
    validate_config, get_experiment_summary, test_model_availability,
    MODELS, DATASETS, SEARCH_METHODS, ALTERNATIVE_MODELS,
    load_dataset, load_model, use_alternative_models
)
    print("✅ Successfully imported configuration modules")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the experiments directory and all dependencies are installed")
    sys.exit(1)

def test_configuration():
    """Test the configuration system"""
    print("\n" + "="*60)
    print("🧪 TESTING CONFIGURATION SYSTEM")
    print("="*60)
    
    # Test configuration validation
    try:
        validate_config()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test configuration summary
    try:
        summary = get_experiment_summary()
        print("✅ Configuration summary generated")
        print(summary)
    except Exception as e:
        print(f"❌ Configuration summary failed: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\n" + "="*60)
    print("📚 TESTING DATASET LOADING")
    print("="*60)
    
    success_count = 0
    total_count = len(DATASETS)
    
    for dataset_name in DATASETS.keys():
        try:
            print(f"\n🔄 Testing dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, max_samples=2)  # Load only 2 samples for testing
            
            if dataset is not None:
                print(f"  ✅ Successfully loaded {dataset_name}")
                print(f"  📊 Dataset size: {len(dataset)}")
                print(f"  🔍 Sample fields: {list(dataset[0].keys())}")
                success_count += 1
            else:
                print(f"  ❌ Failed to load {dataset_name}")
                
        except Exception as e:
            print(f"  ❌ Error loading {dataset_name}: {e}")
    
    print(f"\n📊 Dataset loading results: {success_count}/{total_count} successful")
    return success_count == total_count

def test_model_configs():
    """Test model configuration structure"""
    print("\n" + "="*60)
    print("🤖 TESTING MODEL CONFIGURATIONS")
    print("="*60)
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_config in MODELS.items():
        try:
            print(f"\n🔍 Testing model: {model_name}")
            print(f"  📝 Name: {model_config.name}")
            print(f"  🆔 Model ID: {model_config.model_id}")
            print(f"  🌡️  Temperature: {model_config.temperature}")
            print(f"  🎯 Max Tokens: {model_config.max_tokens}")
            print(f"  🛑 Stop Tokens: {model_config.stop_tokens}")
            
            # Test that all required fields are present
            required_fields = ['name', 'model_id', 'max_tokens', 'temperature', 'stop_tokens']
            if all(hasattr(model_config, field) for field in required_fields):
                print(f"  ✅ All required fields present")
                success_count += 1
            else:
                print(f"  ❌ Missing required fields")
                
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
    
    print(f"\n📊 Model configuration results: {success_count}/{total_count} successful")
    return success_count == total_count

def test_search_methods():
    """Test search method configurations"""
    print("\n" + "="*60)
    print("🔍 TESTING SEARCH METHOD CONFIGURATIONS")
    print("="*60)
    
    success_count = 0
    total_count = len(SEARCH_METHODS)
    
    for method_name, method_config in SEARCH_METHODS.items():
        try:
            print(f"\n🔍 Testing search method: {method_name}")
            print(f"  📝 Name: {method_config.name}")
            print(f"  🔍 Uses Verifier: {method_config.use_verifier}")
            print(f"  🎯 Uses Scorer: {method_config.use_scorer}")
            print(f"  📖 Description: {method_config.description}")
            
            # Test that all required fields are present
            required_fields = ['name', 'use_verifier', 'use_scorer', 'description']
            if all(hasattr(method_config, field) for field in required_fields):
                print(f"  ✅ All required fields present")
                success_count += 1
            else:
                print(f"  ❌ Missing required fields")
                
        except Exception as e:
            print(f"  ❌ Error testing {method_name}: {e}")
    
    print(f"\n📊 Search method configuration results: {success_count}/{total_count} successful")
    return success_count == total_count

def test_alternative_models():
    """Test alternative model configurations"""
    print("\n" + "="*60)
    print("🔍 TESTING ALTERNATIVE MODELS")
    print("="*60)
    
    success_count = 0
    total_count = len(ALTERNATIVE_MODELS)
    
    for model_name, model_config in ALTERNATIVE_MODELS.items():
        try:
            print(f"\n🔍 Testing alternative model: {model_name}")
            print(f"  📝 Name: {model_config.name}")
            print(f"  🆔 Model ID: {model_config.model_id}")
            print(f"  🌡️  Temperature: {model_config.temperature}")
            print(f"  🎯 Max Tokens: {model_config.max_tokens}")
            
            # Test that all required fields are present
            required_fields = ['name', 'model_id', 'max_tokens', 'temperature', 'stop_tokens']
            if all(hasattr(model_config, field) for field in required_fields):
                print(f"  ✅ All required fields present")
                success_count += 1
            else:
                print(f"  ❌ Missing required fields")
                
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
    
    print(f"\n📊 Alternative model configuration results: {success_count}/{total_count} successful")
    return success_count == total_count

def main():
    """Run all tests"""
    print("🚀 Starting configuration system tests...")
    
    # Run tests
    config_ok = test_configuration()
    datasets_ok = test_dataset_loading()
    models_ok = test_model_configs()
    search_ok = test_search_methods()
    alt_models_ok = test_alternative_models()
    
    # Summary
    print("\n" + "="*60)
    print("🎉 TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Configuration Validation", config_ok),
        ("Dataset Loading", datasets_ok),
        ("Model Configurations", models_ok),
        ("Search Method Configurations", search_ok),
        ("Alternative Models", alt_models_ok)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
