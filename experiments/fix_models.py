#!/usr/bin/env python3
"""
Script to help fix model loading issues and test alternative models.
This will help you get your experiments running even with limited disk space.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config import (
        MODELS, ALTERNATIVE_MODELS, execution_config,
        test_model_availability, use_alternative_models,
        quick_test_config, full_experiment_config
    )
    print("✅ Successfully imported configuration modules")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the experiments directory")
    sys.exit(1)

def show_current_models():
    """Show which models are currently configured"""
    print("\n📊 CURRENT MODEL CONFIGURATION")
    print("="*50)
    
    print("🔧 Models currently set to run:")
    for model_name in execution_config.models_to_run:
        if model_name in MODELS:
            model_config = MODELS[model_name]
            print(f"   • {model_config.name} ({model_config.model_id})")
        else:
            print(f"   • {model_name} (not found in MODELS)")
    
    print(f"\n📈 Total models to run: {len(execution_config.models_to_run)}")

def show_alternative_models():
    """Show available alternative models"""
    print("\n🔍 ALTERNATIVE MODELS AVAILABLE")
    print("="*50)
    
    for model_name, model_config in ALTERNATIVE_MODELS.items():
        print(f"   • {model_name}: {model_config.name}")
        print(f"     ID: {model_config.model_id}")
        print(f"     Size: ~{model_name.split('-')[-1]} parameters")
        print()

def test_model_access():
    """Test if models can be accessed on Hugging Face"""
    print("\n🧪 TESTING MODEL ACCESS ON HUGGING FACE")
    print("="*50)
    
    try:
        from huggingface_hub import model_info
        
        print("🔍 Testing current models...")
        for model_name in execution_config.models_to_run:
            if model_name in MODELS:
                model_config = MODELS[model_name]
                try:
                    info = model_info(model_config.model_id)
                    print(f"   ✅ {model_config.name}: Accessible")
                except Exception as e:
                    print(f"   ❌ {model_config.name}: {e}")
        
        print("\n🔍 Testing alternative models...")
        for model_name, model_config in ALTERNATIVE_MODELS.items():
            try:
                info = model_info(model_config.model_id)
                print(f"   ✅ {model_config.name}: Accessible")
            except Exception as e:
                print(f"   ❌ {model_config.name}: {e}")
                
    except ImportError:
        print("❌ huggingface_hub not available")
        print("💡 Install with: pip install huggingface_hub")

def switch_to_alternative_models():
    """Switch to smaller alternative models"""
    print("\n🔄 SWITCHING TO ALTERNATIVE MODELS")
    print("="*50)
    
    use_alternative_models()
    
    print("\n✅ Configuration updated!")
    print("💡 Now you can run your experiments with smaller models")
    print("   Run: python run_experiments.py")

def quick_test_with_alternatives():
    """Set up a quick test with alternative models"""
    print("\n🔧 SETTING UP QUICK TEST WITH ALTERNATIVE MODELS")
    print("="*50)
    
    # Apply quick test config
    quick_test_config()
    
    # Switch to alternative models
    use_alternative_models()
    
    print("\n✅ Quick test configuration applied with alternative models!")
    print("💡 This will run:")
    print("   • 1 alternative model (smaller)")
    print("   • 1 dataset (GSM8K)")
    print("   • 1 search method (SSDP)")
    print("   • 5 questions per dataset")
    print("\n🚀 Ready to run: python run_experiments.py")

def main():
    """Main menu for fixing model issues"""
    print("🔧 MODEL LOADING ISSUE FIXER")
    print("="*50)
    
    while True:
        print("\n📋 Available options:")
        print("1. Show current model configuration")
        print("2. Show alternative models available")
        print("3. Test model access on Hugging Face")
        print("4. Switch to alternative models")
        print("5. Set up quick test with alternatives")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            show_current_models()
            
        elif choice == "2":
            show_alternative_models()
            
        elif choice == "3":
            test_model_access()
            
        elif choice == "4":
            switch_to_alternative_models()
            
        elif choice == "5":
            quick_test_with_alternatives()
            
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
