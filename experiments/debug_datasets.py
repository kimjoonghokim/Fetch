#!/usr/bin/env python3
"""
Debug script to troubleshoot dataset loading issues.
This will help identify the correct dataset IDs and any installation problems.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_dataset_installation():
    """Test if the datasets library is properly installed"""
    print("🧪 TESTING DATASETS LIBRARY INSTALLATION")
    print("="*50)
    
    try:
        import datasets
        print(f"✅ Datasets library version: {datasets.__version__}")
        
        # Check if streaming is available
        try:
            from datasets import load_dataset
            print("✅ load_dataset function available")
        except ImportError as e:
            print(f"❌ load_dataset not available: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Datasets library not found: {e}")
        print("💡 Install with: pip install datasets[streaming]")
        return False

def test_gsm8k_loading():
    """Test different ways to load GSM8K dataset"""
    print("\n🔍 TESTING GSM8K DATASET LOADING")
    print("="*50)
    
    from datasets import load_dataset
    
    gsm8k_variants = [
        ("openai/gsm8k", "main"),
        ("openai/gsm8k", "socratic"),
        ("gsm8k", "main"),
        ("gsm8k", "socratic")
    ]
    
    for i, (variant, config) in enumerate(gsm8k_variants):
        try:
            print(f"\n🔄 Trying variant {i+1}: {variant} with config '{config}'")
            dataset = load_dataset(variant, config, split="test")
            print(f"✅ SUCCESS: Loaded {variant} with config '{config}' - {len(dataset)} samples")
            print(f"   📊 Sample fields: {list(dataset[0].keys())}")
            print(f"   📝 First question: {dataset[0]['question'][:100]}...")
            return dataset
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("❌ All GSM8K loading attempts failed")
    return None

def test_aime_loading():
    """Test different ways to load AIME dataset"""
    print("\n🔍 TESTING AIME DATASET LOADING")
    print("="*50)
    
    from datasets import load_dataset
    
    aime_variants = [
        ("opencompass/AIME2025", "AIME2025-I"),
        ("opencompass/AIME2025", "AIME2025-II"),
        ("opencompass/AIME2025", "AIME2025-I"),  # Test with token
        ("opencompass/AIME2025", "AIME2025-II")   # Test with token
    ]
    
    for i, (variant, config) in enumerate(aime_variants):
        try:
            print(f"\n🔄 Trying variant {i+1}: {variant} with config '{config}'")
            dataset = load_dataset(variant, config, split="test")
            print(f"✅ SUCCESS: Loaded {variant} with config '{config}' - {len(dataset)} samples")
            print(f"   📊 Sample fields: {list(dataset[0].keys())}")
            if len(dataset) > 0:
                print(f"   📝 Sample data: {dataset[0]}")
            return dataset
        except Exception as e:
            print(f"❌ Failed: {e}")
            # Check if it's an authentication error
            if "authentication" in str(e).lower() or "token" in str(e).lower() or "login" in str(e).lower() or "unauthorized" in str(e).lower():
                print("   🔐 This appears to be an authentication issue")
    
    print("❌ All AIME loading attempts failed")
    return None

def test_math500_loading():
    """Test MATH-500 dataset loading"""
    print("\n🔍 TESTING MATH-500 DATASET LOADING")
    print("="*50)
    
    from datasets import load_dataset
    
    try:
        print("🔄 Trying HuggingFaceH4/MATH-500")
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"✅ SUCCESS: Loaded MATH-500 with {len(dataset)} samples")
        print(f"   📊 Sample fields: {list(dataset[0].keys())}")
        if len(dataset) > 0:
            print(f"   📝 Sample data: {dataset[0]}")
        return dataset
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def check_huggingface_access():
    """Check if we can access Hugging Face datasets"""
    print("\n🌐 CHECKING HUGGING FACE ACCESS")
    print("="*50)
    
    try:
        from datasets import load_dataset
        
        # Try to load a simple, public dataset
        print("🔄 Testing with a simple public dataset...")
        dataset = load_dataset("squad", split="train[:1]")
        print(f"✅ SUCCESS: Can access Hugging Face datasets")
        print(f"   📊 Loaded SQuAD sample with {len(dataset)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Cannot access Hugging Face datasets: {e}")
        print("💡 This might be a network or authentication issue")
        return False

def check_authentication_status():
    """Check Hugging Face authentication status"""
    print("\n🔐 CHECKING AUTHENTICATION STATUS")
    print("="*50)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get user info
        try:
            user = api.whoami()
            print(f"✅ Authenticated as: {user}")
            return True
        except Exception as e:
            print(f"❌ Not authenticated: {e}")
            return False
            
    except ImportError:
        print("❌ huggingface_hub not available")
        print("💡 Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 DATASET LOADING DIAGNOSTICS")
    print("="*60)
    
    # Test basic installation
    if not test_dataset_installation():
        print("\n❌ Datasets library not properly installed")
        print("💡 Please install with: pip install datasets[streaming]")
        return False
    
    # Check Hugging Face access
    if not check_huggingface_access():
        print("\n❌ Cannot access Hugging Face datasets")
        print("💡 Check your internet connection and Hugging Face access")
        return False
    
    # Check authentication status
    auth_status = check_authentication_status()
    if not auth_status:
        print("\n⚠️  Not authenticated with Hugging Face")
        print("💡 Some datasets may require authentication")
        print("   Run: huggingface-cli login")
        print("   Get token from: https://huggingface.co/settings/tokens")
    
    # Test individual datasets
    print("\n" + "="*60)
    print("📚 TESTING INDIVIDUAL DATASETS")
    print("="*60)
    
    gsm8k_ok = test_gsm8k_loading() is not None
    aime_ok = test_aime_loading() is not None
    math500_ok = test_math500_loading() is not None
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    datasets_status = [
        ("GSM8K", gsm8k_ok),
        ("AIME2025", aime_ok),
        ("MATH-500", math500_ok)
    ]
    
    for name, status in datasets_status:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {'Working' if status else 'Failed'}")
    
    working_count = sum(1 for _, status in datasets_status if status)
    total_count = len(datasets_status)
    
    print(f"\n📈 Overall: {working_count}/{total_count} datasets working")
    
    if working_count == total_count:
        print("🎉 All datasets are working! You can run your experiments.")
    else:
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Make sure you have the latest datasets library:")
        print("   pip install --upgrade datasets[streaming]")
        print("2. Try installing datasets individually:")
        print("   python -c \"from datasets import load_dataset; load_dataset('gsm8k', split='test')\"")
        print("3. Check if you need authentication for private datasets")
        print("4. Verify your internet connection")
    
    return working_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
