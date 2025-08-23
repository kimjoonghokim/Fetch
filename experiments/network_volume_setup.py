#!/usr/bin/env python3
"""
Script to set up and use network volume for storing models and cache.
This will help alleviate disk space issues on the container.
"""

import os
import shutil
from pathlib import Path
import subprocess

def check_network_volume():
    """Check if network volume is mounted and accessible"""
    print("🔍 CHECKING NETWORK VOLUME")
    print("="*50)
    
    # Common network volume mount points
    possible_mounts = [
        "/mnt/network",
        "/mnt/storage", 
        "/mnt/data",
        "/mnt/volume",
        "/network",
        "/storage"
    ]
    
    network_volume = None
    for mount_point in possible_mounts:
        if os.path.exists(mount_point) and os.path.ismount(mount_point):
            # Check if it's writable
            try:
                test_file = Path(mount_point) / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                network_volume = mount_point
                print(f"✅ Found network volume at: {mount_point}")
                break
            except Exception:
                print(f"⚠️  Found mount point at {mount_point} but not writable")
    
    if not network_volume:
        print("❌ No writable network volume found")
        print("💡 Common locations to check:")
        for mount_point in possible_mounts:
            if os.path.exists(mount_point):
                print(f"   - {mount_point} (exists)")
            else:
                print(f"   - {mount_point} (not found)")
        
        # Check all mounted filesystems
        try:
            result = subprocess.run(['df', '-h'], capture_output=True, text=True)
            print(f"\n📊 All mounted filesystems:")
            print(result.stdout)
        except Exception as e:
            print(f"Could not check mounted filesystems: {e}")
        
        return None
    
    return network_volume

def setup_huggingface_cache_on_network(network_volume):
    """Set up Hugging Face cache on network volume"""
    print(f"\n🔧 SETTING UP HUGGING FACE CACHE ON NETWORK VOLUME")
    print("="*60)
    
    # Create cache directories on network volume
    network_cache = Path(network_volume) / "huggingface_cache"
    network_cache.mkdir(exist_ok=True)
    
    # Create subdirectories
    (network_cache / "hub").mkdir(exist_ok=True)
    (network_cache / "datasets").mkdir(exist_ok=True)
    (network_cache / "transformers").mkdir(exist_ok=True)
    
    print(f"✅ Created cache directories on network volume:")
    print(f"   • {network_cache}")
    print(f"   • {network_cache / 'hub'}")
    print(f"   • {network_cache / 'datasets'}")
    print(f"   • {network_cache / 'transformers'}")
    
    # Set environment variables
    os.environ['HF_HOME'] = str(network_cache)
    os.environ['TRANSFORMERS_CACHE'] = str(network_cache / "transformers")
    os.environ['HF_DATASETS_CACHE'] = str(network_cache / "datasets")
    
    print(f"✅ Set environment variables:")
    print(f"   • HF_HOME = {os.environ['HF_HOME']}")
    print(f"   • TRANSFORMERS_CACHE = {os.environ['TRANSFORMERS_CACHE']}")
    print(f"   • HF_DATASETS_CACHE = {os.environ['HF_DATASETS_CACHE']}")
    
    return network_cache

def move_existing_cache_to_network(network_cache):
    """Move existing cache to network volume if it exists"""
    print(f"\n🔄 MOVING EXISTING CACHE TO NETWORK VOLUME")
    print("="*60)
    
    home_cache = Path.home() / ".cache" / "huggingface"
    
    if home_cache.exists():
        print(f"📁 Found existing cache at: {home_cache}")
        
        try:
            # Move the cache
            shutil.move(str(home_cache), str(network_cache / "old_cache"))
            print(f"✅ Moved existing cache to: {network_cache / 'old_cache'}")
            
            # Create symlink from home to network
            home_cache.symlink_to(network_cache)
            print(f"✅ Created symlink: {home_cache} -> {network_cache}")
            
        except Exception as e:
            print(f"❌ Failed to move cache: {e}")
            print("💡 You may need to manually move the cache")
            return False
    else:
        print(f"📁 No existing cache found at: {home_cache}")
        
        # Create symlink from home to network
        try:
            home_cache.symlink_to(network_cache)
            print(f"✅ Created symlink: {home_cache} -> {network_cache}")
        except Exception as e:
            print(f"❌ Failed to create symlink: {e}")
            return False
    
    return True

def test_network_volume_setup(network_cache):
    """Test if the network volume setup is working"""
    print(f"\n🧪 TESTING NETWORK VOLUME SETUP")
    print("="*50)
    
    try:
        # Test writing to network volume
        test_file = network_cache / "test_file.txt"
        test_file.write_text("Network volume test")
        print(f"✅ Write test passed: {test_file}")
        
        # Test reading from network volume
        content = test_file.read_text()
        print(f"✅ Read test passed: {content}")
        
        # Clean up test file
        test_file.unlink()
        print(f"✅ Cleanup test passed")
        
        # Test Hugging Face cache environment
        print(f"✅ Environment variables set:")
        print(f"   • HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
        print(f"   • TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
        print(f"   • HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network volume test failed: {e}")
        return False

def show_disk_usage():
    """Show current disk usage"""
    print(f"\n💾 CURRENT DISK USAGE")
    print("="*50)
    
    try:
        # Check container disk usage
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        print("📊 Container disk usage:")
        print(result.stdout)
        
        # Check network volume usage if available
        network_volume = check_network_volume()
        if network_volume:
            result = subprocess.run(['df', '-h', network_volume], capture_output=True, text=True)
            print(f"\n📊 Network volume disk usage:")
            print(result.stdout)
            
    except Exception as e:
        print(f"❌ Could not check disk usage: {e}")

def main():
    """Main setup function"""
    print("🚀 NETWORK VOLUME SETUP FOR HUGGING FACE CACHE")
    print("="*60)
    
    print("This script will help you use your network volume to store")
    print("Hugging Face models and cache, freeing up container disk space.")
    
    # Check current disk usage
    show_disk_usage()
    
    # Check for network volume
    network_volume = check_network_volume()
    if not network_volume:
        print("\n❌ No network volume found or accessible")
        print("💡 Please check your RunPod network volume configuration")
        return False
    
    # Set up cache on network volume
    network_cache = setup_huggingface_cache_on_network(network_volume)
    
    # Move existing cache if it exists
    cache_moved = move_existing_cache_to_network(network_cache)
    
    # Test the setup
    setup_working = test_network_volume_setup(network_cache)
    
    if setup_working:
        print(f"\n🎉 NETWORK VOLUME SETUP COMPLETE!")
        print("="*50)
        print(f"✅ Hugging Face cache now uses network volume: {network_cache}")
        print(f"💾 Container disk space should be freed up")
        print(f"🚀 You can now run your experiments with more disk space")
        
        # Show updated disk usage
        show_disk_usage()
        
        return True
    else:
        print(f"\n❌ NETWORK VOLUME SETUP FAILED")
        print("="*50)
        print(f"💡 Please check the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n💡 Next steps:")
        print(f"   1. Restart your terminal or run: source ~/.bashrc")
        print(f"   2. Try running your experiments again")
        print(f"   3. Models will now be cached on the network volume")
    else:
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if network volume is properly mounted")
        print(f"   2. Verify network volume has write permissions")
        print(f"   3. Check RunPod network volume configuration")
