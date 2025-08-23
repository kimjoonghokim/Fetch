#!/usr/bin/env python3
"""
Simple script to help set up Hugging Face authentication.
This will guide you through getting a token and logging in.
"""

import os
import subprocess
import sys

def check_hf_token():
    """Check if HF_TOKEN environment variable is set"""
    token = os.getenv('HF_TOKEN')
    if token:
        print(f"✅ HF_TOKEN environment variable is set")
        return token
    else:
        print("❌ HF_TOKEN environment variable not set")
        return None

def check_hf_config():
    """Check if user is logged in via huggingface-cli"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Logged in as: {result.stdout.strip()}")
            return True
        else:
            print("❌ Not logged in via huggingface-cli")
            return False
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        return False
    except Exception as e:
        print(f"❌ Error checking login status: {e}")
        return False

def setup_environment_token():
    """Set up HF_TOKEN environment variable"""
    print("\n🔧 SETTING UP ENVIRONMENT TOKEN")
    print("="*40)
    
    token = input("Enter your Hugging Face token: ").strip()
    if not token:
        print("❌ No token provided")
        return False
    
    # Set environment variable for current session
    os.environ['HF_TOKEN'] = token
    
    # Add to shell profile
    shell_profile = os.path.expanduser("~/.bashrc")
    if os.path.exists(shell_profile):
        with open(shell_profile, 'a') as f:
            f.write(f'\n# Hugging Face token\nexport HF_TOKEN="{token}"\n')
        print(f"✅ Added token to {shell_profile}")
        print("💡 Restart your terminal or run: source ~/.bashrc")
    
    return True

def login_via_cli():
    """Login using huggingface-cli"""
    print("\n🔐 LOGGING IN VIA CLI")
    print("="*40)
    
    try:
        subprocess.run(['huggingface-cli', 'login'], check=True)
        print("✅ Login successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Login failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        print("💡 Install with: pip install huggingface_hub")
        return False

def test_dataset_access():
    """Test if we can access AIME dataset"""
    print("\n🧪 TESTING AIME DATASET ACCESS")
    print("="*40)
    
    try:
        from datasets import load_dataset
        
        # Try with environment token
        token = os.getenv('HF_TOKEN')
        if token:
            print("🔄 Trying with HF_TOKEN environment variable...")
            try:
                dataset = load_dataset("opencompass/AIME2025", split="test", token=token)
                print(f"✅ SUCCESS: Loaded AIME dataset with {len(dataset)} samples")
                return True
            except Exception as e:
                print(f"❌ Failed with HF_TOKEN: {e}")
        
        # Try with CLI login
        print("🔄 Trying with CLI login...")
        try:
            dataset = load_dataset("opencompass/AIME2025", split="test")
            print(f"✅ SUCCESS: Loaded AIME dataset with {len(dataset)} samples")
            return True
        except Exception as e:
            print(f"❌ Failed with CLI login: {e}")
        
        return False
        
    except ImportError:
        print("❌ datasets library not available")
        return False

def main():
    """Main setup function"""
    print("🚀 HUGGING FACE AUTHENTICATION SETUP")
    print("="*50)
    
    print("This script will help you set up authentication for Hugging Face datasets.")
    print("Some datasets (like AIME) require authentication even if they're publicly available.")
    
    # Check current status
    print("\n📊 CURRENT STATUS")
    print("-" * 30)
    
    env_token = check_hf_token()
    cli_login = check_hf_config()
    
    if env_token and cli_login:
        print("🎉 You're all set! Both authentication methods are working.")
        return True
    
    # Setup options
    print("\n🔧 SETUP OPTIONS")
    print("-" * 30)
    print("1. Set up environment variable (HF_TOKEN)")
    print("2. Login via huggingface-cli")
    print("3. Both")
    print("4. Skip setup")
    
    choice = input("\nSelect an option (1-4): ").strip()
    
    if choice in ['1', '3']:
        setup_environment_token()
    
    if choice in ['2', '3']:
        login_via_cli()
    
    if choice == '4':
        print("⏭️  Skipping setup")
    
    # Test access
    print("\n🧪 TESTING ACCESS")
    print("-" * 30)
    
    if test_dataset_access():
        print("🎉 AIME dataset access successful!")
        return True
    else:
        print("❌ AIME dataset access failed")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Make sure you have a valid token from https://huggingface.co/settings/tokens")
        print("2. Try logging in again: huggingface-cli login")
        print("3. Check if the dataset requires special access")
        print("4. Verify your internet connection")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
