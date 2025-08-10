#!/usr/bin/env python3

from beamsearch import call_value, call_policy

def test_basic_beamsearch():
    print("🧪 Testing Basic Beam Search Random Scoring...")
    
    # Test data
    question = "Johnny has 10 apples. He gives 4 to his friend, then he eats half of the remaining apples. How many apples does he have left?"
    paths = [
        "Let me think about this step by step.",
        "I need to solve this mathematically.",
        "This is a simple addition problem."
    ]
    
    print(f"Question: {question}")
    
    # Test call_value function (random scoring)
    print("\n🔢 Testing call_value (random scoring for now)...")
    for i, path in enumerate(paths):
        print(f"\nPath {i+1}: {path}")
        
        # Test multiple runs to see randomness
        for run in range(3):
            value = call_value(question, path)
            print(f"  Run {run+1}: {value:.4f}")
    
    # Test call_policy function (policy model calls)
    print("\n🤖 Testing call_policy (policy model)...")
    print("Note: This will try to call the policy server at 127.0.0.1:8000")
    print("If server is not running, this will fail")
    
    try:
        for i, path in enumerate(paths[:3]):  # Test first 2 paths
            print(f"\nPath {i+1}: {path}")
            response = call_policy(question, path)
            print(f"  Policy response: {response}.")  # Show first 100 chars
    except Exception as e:
        print(f"  Policy server error: {e}")

if __name__ == "__main__":
    test_basic_beamsearch() 