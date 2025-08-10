#!/usr/bin/env python3

from gsm_config import GSMConfig

def test_mcts_random():
    
    config = GSMConfig()
    
    
    # Test the get_value method
    print("\n🔄 Testing get_value method...")
    questions = [
        "What is 2+2?",
        "What is 5+3?",
        "What is 10-4?"
    ]
    
    steps_list = [
        ["Let me think about this step by step."],
        ["I need to solve this mathematically."],
        ["This is a simple subtraction problem."]
    ]
    
    for i, (question, steps) in enumerate(zip(questions, steps_list)):
        print(f"\nQuestion {i+1}: {question}")
        print(f"Steps: {steps}")
        
        # Test multiple runs to see randomness
        for run in range(1):
            value = config.get_value(question, steps)
            print(f"  Run {run+1}: {value:.4f}")

if __name__ == "__main__":
    test_mcts_random() 