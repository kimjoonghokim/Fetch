#!/usr/bin/env python3
"""
Demo test file for the scoring system.

Make sure your vLLM policy server is running before testing:
python3 -m vllm.entrypoints.openai.api_server --model xmu-nlp/Llama-3-8b-gsm8k --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096

Also ensure ESM service is running for semantic similarity:
python cluster/server_cluster.py

Run this demo:
cd /workspace/Fetch
python scorer/demo.py
"""

from scoring import (
    AnswerScorer, 
    get_simple_confidence, 
    get_overall_answer_score, 
    get_vote_score_simple,
    check_esm_service_status,
    ScoringMethod
)


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


def demo_overall_scoring():
    """Demo the new overall scoring system."""
    print("\n\n🏆 Overall Scoring Demo")
    print("=" * 50)
    
    questions = [
        "What is 12*15?",
        "What is the square root of 64?",
        "Solve 2x + 5 = 15"
    ]
    
    for question in questions:
        print(f"\n📝 Question: {question}")
        try:
            # Get simple overall score
            overall_score = get_overall_answer_score(question)
            print(f"🏆 Overall Score: {overall_score:.3f}")
            
            # Get detailed breakdown
            scorer = AnswerScorer()
            detailed_result = scorer.get_overall_score(question)
            
            print(f"📊 Component Breakdown:")
            for component, data in detailed_result['component_scores'].items():
                if data['score'] > 0:
                    print(f"   • {component}: {data['score']:.3f}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_custom_weights():
    """Demo custom weighting for overall scoring."""
    print("\n\n⚖️ Custom Weights Demo")
    print("=" * 50)
    
    question = "What is 25*16?"
    
    # Test different weight configurations
    weight_configs = [
        ("Default (confidence only)", None),
        ("With length penalty", {"confidence": 0.8, "length_penalty": 0.2}),
        ("Future example", {"confidence": 0.5, "parent_child_quality": 0.3, "semantic_similarity": 0.2})
    ]
    
    print(f"📝 Question: {question}")
    
    for config_name, weights in weight_configs:
        try:
            score = get_overall_answer_score(question, "", weights=weights)
            print(f"\n⚖️ {config_name}: {score:.3f}")
            
            if weights:
                print(f"   Weights: {weights}")
                
        except Exception as e:
            print(f"❌ {config_name}: Error - {e}")


def demo_detailed_breakdown():
    """Demo detailed scoring breakdown."""
    print("\n\n🔍 Detailed Breakdown Demo")
    print("=" * 50)
    
    scorer = AnswerScorer()
    question = "What is 7*9?"
    
    print(f"📝 Question: {question}")
    
    try:
        result = scorer.get_overall_score(question, "", weights={"confidence": 0.8, "length_penalty": 0.2})
        
        print(f"\n🏆 Overall Score: {result['overall_score']:.3f}")
        print(f"⚖️ Total Weight: {result['total_weight']:.1f}")
        
        print(f"\n📊 Component Details:")
        for component, data in result['component_scores'].items():
            score = data['score']
            status = data['details'].get('status', 'active')
            
            if score > 0 or status == 'active':
                print(f"   • {component}: {score:.3f}")
                if 'text_generated' in data['details']:
                    print(f"     Generated: '{data['details']['text_generated']}'")
                if 'word_count' in data['details']:
                    print(f"     Word count: {data['details']['word_count']}")
            else:
                print(f"   • {component}: {status}")
        
        print(f"\n⚖️ Weights Used:")
        for component, weight in result['weights_used'].items():
            print(f"   • {component}: {weight:.1f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_vote_scoring():
    """Demo the new vote scoring system based on semantic similarity merges."""
    print("\n\n🗳️ Vote Scoring Demo")
    print("=" * 50)
    
    # Check ESM service first
    esm_available = check_esm_service_status()
    if esm_available:
        print("✅ Using ESM service for semantic clustering")
    else:
        print("⚠️  Using fallback clustering (ESM service unavailable)")
    
    # Test cases with different levels of semantic similarity
    test_cases = [
        {
            "question": "What is 15*8?",
            "path": "Let me solve this step by step",
            "candidates": [
                "I'll calculate this multiplication",
                "Let me work through this problem",
                "I need to multiply 15 by 8"
            ]
        },
        {
            "question": "Solve x^2 = 16",
            "path": "I need to find the square root",
            "candidates": [
                "Let me find the square root of 16",
                "I'll solve this quadratic equation",
                "The answer is x = ±4"
            ]
        },
        {
            "question": "What is the capital of France?",
            "path": "The capital of France is Paris",
            "candidates": [
                "Paris is the capital city",
                "France's capital is Paris",
                "The main city is Paris"
            ]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['question']}")
        print(f"Path: {case['path']}")
        print(f"Candidates: {case['candidates']}")
        
        try:
            # Get vote score only
            vote_score = get_vote_score_simple(
                case['question'], 
                case['path'], 
                case['candidates']
            )
            print(f"🗳️ Vote Score: {vote_score:.3f}")
            
            # Get detailed vote information
            scorer = AnswerScorer()
            vote_details = scorer.get_vote_score(
                case['question'], 
                case['path'], 
                case['candidates'],
                include_raw=True
            )
            print(f"🔗 Merge Count: {vote_details.merge_count}")
            print(f"📊 Cluster Sizes: {vote_details.cluster_sizes}")
            
            # Show clustering method if raw response is available
            if vote_details.raw_response:
                method = vote_details.raw_response.get('clustering_method', 'unknown')
                print(f"🔧 Clustering Method: {method}")
            
            # Get combined score (confidence + vote)
            combined_score = get_overall_answer_score(
                case['question'], 
                case['path'], 
                candidate_paths=case['candidates']
            )
            print(f"🏆 Combined Score: {combined_score:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)


def check_esm():
    """Check if the ESM service is running."""
    print("🔍 Checking ESM Service Connection...")
    print("=" * 50)
    
    try:
        esm_available = check_esm_service_status()
        if esm_available:
            print("✅ ESM service is running! Semantic clustering will use ESM.")
        else:
            print("❌ Cannot connect to ESM service at http://127.0.0.1:8003")
            print("\n🚀 To start the ESM service, run:")
            print("python cluster/server_cluster.py")
            print("\n⚠️  Vote scoring will use fallback clustering.")
        return esm_available
    except Exception as e:
        print(f"❌ ESM service connection error: {e}")
        return False


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
    print("🚀 Enhanced Scoring System Demo")
    print("=" * 60)
    
    # Check servers first
    policy_available = check_server()
    esm_available = check_esm()
    
    if not policy_available:
        print("\n⚠️  Cannot run demos without policy server connection.")
        return
    
    if not esm_available:
        print("\n⚠️  Vote scoring will use fallback clustering without ESM service.")
    
    # Run demos
    demo_basic_confidence()
    demo_overall_scoring()
    demo_custom_weights()
    demo_detailed_breakdown()
    demo_vote_scoring()
    
    print("\n\n🎉 Demo Complete!")
    print("=" * 60)
    print("\n💡 For search algorithms, use:")
    print("   from scorer.scoring import get_overall_answer_score")
    print("   score = get_overall_answer_score(question, path, candidate_paths=candidates)")
    print("\n🗳️ For vote scoring only:")
    print("   from scorer.scoring import get_vote_score_simple")
    print("   vote_score = get_vote_score_simple(question, path, candidate_paths)")
    print("\n🔍 For detailed vote information:")
    print("   from scorer.scoring import AnswerScorer")
    print("   scorer = AnswerScorer()")
    print("   details = scorer.get_vote_score(question, path, candidates, include_raw=True)")
    print("\n🔍 To check ESM service:")
    print("   from scorer.scoring import check_esm_service_status")
    print("   esm_available = check_esm_service_status()")


if __name__ == "__main__":
    main()