#!/usr/bin/env python3

from beamsearch_merge import call_value, call_policy, call_esm

def test_beamsearch_merge():
    print("🧪 Testing Beam Search with State Merging Random Scoring...")
    
    # Test data
    question = "What is 2+2?"
    paths = [
        "Let me think about this step by step.",
        "I need to solve this mathematically.",
        "This is a simple addition problem."
    ]
    
    print(f"Question: {question}")
    
    # Test call_value function (random scoring)
    print("\n🔢 Testing call_value (random scoring)...")
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
        for i, path in enumerate(paths[:2]):  # Test first 2 paths
            print(f"\nPath {i+1}: {path}")
            response = call_policy(question, path)
            print(f"  Policy response: {response[:100]}...")  # Show first 100 chars
    except Exception as e:
        print(f"  Policy server error: {e}")
    
    # Test call_esm function (clustering/merging)
    print("\n🔗 Testing call_esm (clustering/merging)...")
    print("Note: This will try to call the clustering server at 127.0.0.1:8003")
    print("If server is not running, this will fail")
    
    try:
        # Test with some sample texts
        sample_texts = [
            "Let me think about this step by step.",
            "I need to solve this mathematically.",
            "This is a simple addition problem.",
            "Let me solve this step by step.",
            "I need to think about this carefully."
        ]
        
        print(f"\nSample texts: {sample_texts}")
        labels = call_esm(sample_texts)
        print(f"  Clustering labels: {labels}")
    except Exception as e:
        print(f"  Clustering server error: {e}")

if __name__ == "__main__":
    test_beamsearch_merge() 