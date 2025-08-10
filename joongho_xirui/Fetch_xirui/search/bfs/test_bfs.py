#!/usr/bin/env python3

from call_joint_service import Worker, PolicyArgument

def test_random_scoring():
    print("�� Testing BFS Random Scoring...")
    
    # Create worker (no verifier needed)
    worker = Worker(PolicyArgument(), None)
    
    # Test data
    question = "What is 2+2?"
    path = "Let me think about this step by step."
    next_step = "First, I need to add 2 and 2."
    
    print(f"Question: {question}")
    print(f"Path: {path}")
    print(f"Next step: {next_step}")
    
    # Test random value generation
    print("\n🔢 Generating random values...")
    for i in range(5):
        next_step, value = worker.encode(question, path, temp=None, stop=[]) #THIS ENCODE RANDOMLY GENERATES VALUE IN CALL_JOINT_SERVICE.PY
        print(f"  Run {i+1}: {value:.4f}")
    
    # Test with different paths
    print("\n🔄 Testing different paths...")
    different_paths = [
        "Let me solve this step by step.",
        "I need to think about this carefully.",
        "This is a mathematical problem."
    ]
    
    for i, test_path in enumerate(different_paths):
        next_step, value = worker.encode(question, test_path, temp=None, stop=[])
        print(f"  Path {i+1}: {value:.4f}")

if __name__ == "__main__":
    test_random_scoring()