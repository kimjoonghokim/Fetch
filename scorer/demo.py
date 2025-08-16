#!/usr/bin/env python3
"""
Demo test file for the scoring system.

Make sure your vLLM policy server is running before testing:
python3 -m vllm.entrypoints.openai.api_server --model xmu-nlp/Llama-3-8b-gsm8k --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096

Run this demo:
cd /workspace/Fetch
python scorer/demo.py
"""

from scoring import AnswerScorer, get_simple_confidence, ScoringMethod


def demo_basic_confidence():
    """Demo basic confidence scoring."""
    print("🎯 Basic Confidence Scoring Demo")
    print("=" * 50)
    
    # Simple questions to test
    questions = [
        "What is 2+2?",
        "What is 15*8?",
        "What is the capital of France?"
    ]
    
    for question in questions:
        print(f"\n📝 Question: {question}")
        try:
            confidence = get_simple_confidence(question)
            print(f"✅ Confidence Score: {confidence:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_detailed_scoring():
    """Demo detailed scoring with all metrics."""
    print("\n\n📊 Detailed Scoring Demo")
    print("=" * 50)
    
    scorer = AnswerScorer()
    question = "What is 12*15?"
    
    print(f"📝 Question: {question}")
    
    try:
        result = scorer.get_confidence_score(question, "", include_raw=False)
        
        print(f"\n🤖 Generated Answer: '{result.text}'")
        print(f"📈 Average Confidence: {result.avg_confidence:.3f}" if result.avg_confidence else "No avg confidence")
        print(f"⬇️  Minimum Confidence: {result.min_confidence:.3f}" if result.min_confidence else "No min confidence") 
        print(f"📐 Geometric Confidence: {result.geometric_confidence:.3f}" if result.geometric_confidence else "No geometric confidence")
        print(f"🌀 Perplexity: {result.perplexity:.2f}" if result.perplexity else "No perplexity")
        
        if result.tokens and result.token_logprobs:
            print(f"\n🔤 Token Details:")
            for token, logprob in zip(result.tokens, result.token_logprobs):
                if logprob is not None:
                    confidence = round(2.718 ** logprob, 3)  # exp(logprob)
                    print(f"   '{token}' → {confidence:.3f}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_step_by_step():
    """Demo step-by-step reasoning with confidence."""
    print("\n\n🪜 Step-by-Step Reasoning Demo")
    print("=" * 50)
    
    scorer = AnswerScorer()
    question = "What is 25*16?"
    
    # Simulate step-by-step reasoning
    paths = [
        "",  # No previous steps
        "I need to multiply 25 by 16.",  # First step
        "I need to multiply 25 by 16. Let me break this down: 25 * 16 = 25 * (10 + 6)"  # More context
    ]
    
    for i, path in enumerate(paths):
        print(f"\n🔹 Step {i+1}:")
        print(f"   Previous context: '{path if path else 'None'}'")
        
        try:
            result = scorer.get_confidence_score(question, path)
            print(f"   🤖 Generated: '{result.text}'")
            print(f"   📈 Confidence: {result.avg_confidence:.3f}" if result.avg_confidence else "   No confidence")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_different_methods():
    """Demo different scoring methods."""
    print("\n\n🔄 Different Scoring Methods Demo")
    print("=" * 50)
    
    scorer = AnswerScorer()
    question = "What is 7*9?"
    
    methods = [
        ("Average Confidence", ScoringMethod.CONFIDENCE_AVERAGE),
        ("Minimum Confidence", ScoringMethod.CONFIDENCE_MIN),
        ("Geometric Confidence", ScoringMethod.CONFIDENCE_GEOMETRIC),
        ("Perplexity (inverted)", ScoringMethod.PERPLEXITY)
    ]
    
    print(f"📝 Question: {question}")
    
    for method_name, method in methods:
        try:
            result = scorer.get_confidence_score(question, "", method=method)
            primary_score = scorer.get_primary_score(result, method)
            print(f"   📊 {method_name}: {primary_score:.3f}")
        except Exception as e:
            print(f"   ❌ {method_name}: Error - {e}")


def check_server():
    """Check if the vLLM server is running."""
    print("🌐 Checking Server Connection...")
    print("=" * 50)
    
    import requests
    
    try:
        # Try to connect to the server
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"✅ Server is running! Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to vLLM server at http://127.0.0.1:8000")
        print("\n🚀 To start the server, run:")
        print("python3 -m vllm.entrypoints.openai.api_server --model xmu-nlp/Llama-3-8b-gsm8k --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096")
        return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False


def main():
    """Run all demos."""
    print("🚀 Scoring System Demo")
    print("=" * 60)
    
    # Check server first
    if not check_server():
        print("\n⚠️  Cannot run demos without server connection.")
        return
    
    # Run demos
    demo_basic_confidence()
    demo_detailed_scoring()
    demo_step_by_step()
    demo_different_methods()
    
    print("\n\n🎉 Demo Complete!")
    print("=" * 60)
    print("\n💡 Try running individual functions:")
    print("   from scorer.demo import demo_basic_confidence")
    print("   demo_basic_confidence()")


if __name__ == "__main__":
    main() 