import json
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ScoringMethod(Enum):
    """Available scoring methods"""
    CONFIDENCE_AVERAGE = "confidence_average"
    CONFIDENCE_MIN = "confidence_min"
    CONFIDENCE_GEOMETRIC = "confidence_geometric"
    PERPLEXITY = "perplexity"
    # Future methods to be implemented:
    # PARENT_CHILD_QUALITY = "parent_child_quality"
    # SEMANTIC_SIMILARITY = "semantic_similarity"
    # ENSEMBLE_VOTING = "ensemble_voting"


@dataclass
class ConfidenceScore:
    """Container for confidence scoring results"""
    text: str
    avg_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    geometric_confidence: Optional[float] = None
    perplexity: Optional[float] = None
    token_logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    top_logprobs: Optional[List[Dict]] = None
    raw_response: Optional[Dict] = None


@dataclass
class PolicyConfig:
    """Configuration for policy model API calls"""
    url: str = "http://127.0.0.1"
    port: int = 8000
    model_name: str = "xmu-nlp/Llama-3-8b-gsm8k"
    temperature: float = 0.8
    max_tokens: int = 512
    stop_tokens: List[str] = None
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ["\n"]
    
    @property
    def endpoint_url(self) -> str:
        return f"{self.url}:{self.port}/v1/completions"


class AnswerScorer:
    """
    Comprehensive scoring system for LLM-generated answers.
    
    Current capabilities:
    - Confidence scoring using average log probability
    - Multiple confidence calculation methods
    - Perplexity calculation
    
    Future capabilities (planned):
    - Parent/child node quality scoring
    - Semantic similarity voting
    - Ensemble scoring methods
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        """
        Initialize the scorer with configuration.
        
        Args:
            config: Policy model configuration. Uses defaults if None.
        """
        self.config = config or PolicyConfig()
        
    def get_confidence_score(
        self, 
        question: str, 
        path: str = "", 
        method: ScoringMethod = ScoringMethod.CONFIDENCE_AVERAGE,
        include_raw: bool = False
    ) -> ConfidenceScore:
        """
        Get confidence score for an answer using the specified method.
        
        Args:
            question: The question being answered
            path: Current path/partial answer (for step-by-step reasoning)
            method: Scoring method to use
            include_raw: Whether to include raw API response
            
        Returns:
            ConfidenceScore object with scoring results
        """
        try:
            # Make API call with logprobs to get confidence data
            response_data = self._call_policy_with_confidence(question, path)
            
            # Extract confidence information
            return self._calculate_confidence_scores(response_data, method, include_raw)
            
        except Exception as e:
            print(f"Error getting confidence score: {e}")
            return ConfidenceScore(text="", raw_response={"error": str(e)})
    
    def _call_policy_with_confidence(self, question: str, path: str) -> Dict:
        """
        Call policy model API with confidence data enabled.
        
        Args:
            question: The question
            path: Current path/partial answer
            
        Returns:
            Raw API response with logprobs data
        """
        query = f"Question: {question}\nAnswer:{path}"
        
        payload = {
            "prompt": query,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stop": self.config.stop_tokens,
            "logprobs": 5,  # Request top 5 token probabilities
            "echo": False,  # Don't echo the prompt
            "include_stop_str_in_output": True,
            "skip_special_tokens": False
        }
        
        response = requests.post(self.config.endpoint_url, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.json()
    
    def _calculate_confidence_scores(
        self, 
        response_data: Dict, 
        method: ScoringMethod,
        include_raw: bool = False
    ) -> ConfidenceScore:
        """
        Calculate confidence scores from API response data.
        
        Args:
            response_data: Raw API response
            method: Primary scoring method
            include_raw: Whether to include raw response
            
        Returns:
            ConfidenceScore with calculated metrics
        """
        try:
            choice = response_data["choices"][0]
            text = choice["text"]
            logprobs_data = choice.get("logprobs", {})
            
            # Initialize result
            result = ConfidenceScore(
                text=text,
                raw_response=response_data if include_raw else None
            )
            
            if not logprobs_data:
                return result
            
            # Extract token-level data
            token_logprobs = logprobs_data.get("token_logprobs", [])
            tokens = logprobs_data.get("tokens", [])
            top_logprobs = logprobs_data.get("top_logprobs", [])
            
            result.token_logprobs = token_logprobs
            result.tokens = tokens
            result.top_logprobs = top_logprobs
            
            # Filter out None values (can happen for first token)
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            
            if not valid_logprobs:
                return result
            
            # Calculate all confidence metrics
            result.avg_confidence = self._calculate_average_confidence(valid_logprobs)
            result.min_confidence = self._calculate_min_confidence(valid_logprobs)
            result.geometric_confidence = self._calculate_geometric_confidence(valid_logprobs)
            result.perplexity = self._calculate_perplexity(valid_logprobs)
            
            return result
            
        except Exception as e:
            print(f"Error calculating confidence scores: {e}")
            return ConfidenceScore(text="", raw_response={"error": str(e)})
    
    def _calculate_average_confidence(self, logprobs: List[float]) -> float:
        """Calculate average log probability confidence."""
        avg_logprob = np.mean(logprobs)
        return float(np.exp(avg_logprob))
    
    def _calculate_min_confidence(self, logprobs: List[float]) -> float:
        """Calculate minimum token confidence (weakest link)."""
        min_logprob = min(logprobs)
        return float(np.exp(min_logprob))
    
    def _calculate_geometric_confidence(self, logprobs: List[float]) -> float:
        """Calculate geometric mean confidence."""
        probabilities = [np.exp(lp) for lp in logprobs]
        geometric_mean = np.prod(probabilities) ** (1.0 / len(probabilities))
        return float(geometric_mean)
    
    def _calculate_perplexity(self, logprobs: List[float]) -> float:
        """Calculate perplexity (lower = more confident)."""
        avg_logprob = np.mean(logprobs)
        return float(np.exp(-avg_logprob))
    
    def get_primary_score(self, confidence_score: ConfidenceScore, method: ScoringMethod) -> float:
        """
        Get the primary score based on the specified method.
        
        Args:
            confidence_score: ConfidenceScore object
            method: Scoring method to use
            
        Returns:
            Primary score value, or 0.0 if not available
        """
        if method == ScoringMethod.CONFIDENCE_AVERAGE:
            return confidence_score.avg_confidence or 0.0
        elif method == ScoringMethod.CONFIDENCE_MIN:
            return confidence_score.min_confidence or 0.0
        elif method == ScoringMethod.CONFIDENCE_GEOMETRIC:
            return confidence_score.geometric_confidence or 0.0
        elif method == ScoringMethod.PERPLEXITY:
            # For perplexity, lower is better, so return inverse
            return 1.0 / (confidence_score.perplexity or 1.0)
        else:
            return 0.0
    
    def batch_score_answers(
        self, 
        question_answer_pairs: List[Tuple[str, str]], 
        method: ScoringMethod = ScoringMethod.CONFIDENCE_AVERAGE
    ) -> List[ConfidenceScore]:
        """
        Score multiple answers in batch.
        
        Args:
            question_answer_pairs: List of (question, path) tuples
            method: Scoring method to use
            
        Returns:
            List of ConfidenceScore objects
        """
        results = []
        for question, path in question_answer_pairs:
            score = self.get_confidence_score(question, path, method)
            results.append(score)
        return results
    
    # Future methods for planned features:
    
    def score_parent_child_quality(self, node, parent_nodes: List, child_nodes: List) -> float:
        """
        Score based on parent/child node quality.
        TODO: Implement in future version.
        """
        # Placeholder for future implementation
        pass
    
    def score_semantic_similarity(self, answer: str, reference_answers: List[str]) -> float:
        """
        Score based on semantic similarity to reference answers.
        TODO: Implement in future version.
        """
        # Placeholder for future implementation
        pass
    
    def ensemble_voting_score(self, scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """
        Combine multiple scoring methods using weighted voting.
        TODO: Implement in future version.
        """
        # Placeholder for future implementation
        pass
    
    def get_overall_score(
        self,
        question: str,
        path: str = "",
        node=None,
        parent_nodes: Optional[List] = None,
        child_nodes: Optional[List] = None,
        reference_answers: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ) -> Dict[str, Union[float, Dict]]:
        """
        Calculate an overall score combining multiple scoring methods.
        
        This is the main scoring method that search algorithms should use.
        It combines various scoring components into a single comprehensive score.
        
        Args:
            question: The question being answered
            path: Current path/partial answer
            node: Tree node object (for parent/child analysis) - Future use
            parent_nodes: List of parent nodes for quality scoring - Future use
            child_nodes: List of child nodes for quality scoring - Future use
            reference_answers: Reference answers for similarity scoring - Future use
            weights: Custom weights for different scoring components
            normalize: Whether to normalize the final score to 0-1 range
            
        Returns:
            Dictionary containing:
            - 'overall_score': Final weighted score (0-1 if normalized)
            - 'component_scores': Individual scores from each method
            - 'weights_used': The weights applied to each component
            
        Example:
            scorer = AnswerScorer()
            result = scorer.get_overall_score("What is 2+2?", "")
            print(f"Overall score: {result['overall_score']:.3f}")
        """
        
        # Default weights for scoring components
        default_weights = {
            'confidence': 1.0,          # Currently implemented
            'parent_child_quality': 0.0,  # Future: weight = 0.3
            'semantic_similarity': 0.0,   # Future: weight = 0.2
            'length_penalty': 0.0,        # Future: weight = 0.1
            'coherence': 0.0,            # Future: weight = 0.2
            'factual_consistency': 0.0   # Future: weight = 0.2
        }
        
        # Use provided weights or defaults
        active_weights = weights or default_weights
        
        # Initialize component scores
        component_scores = {}
        
        # 1. CONFIDENCE SCORING (Currently implemented)
        try:
            confidence_result = self.get_confidence_score(question, path)
            component_scores['confidence'] = {
                'score': confidence_result.avg_confidence or 0.0,
                'details': {
                    'avg_confidence': confidence_result.avg_confidence,
                    'min_confidence': confidence_result.min_confidence,
                    'geometric_confidence': confidence_result.geometric_confidence,
                    'perplexity': confidence_result.perplexity,
                    'text_generated': confidence_result.text
                }
            }
        except Exception as e:
            print(f"Warning: Confidence scoring failed: {e}")
            component_scores['confidence'] = {'score': 0.0, 'details': {'error': str(e)}}
        
        # 2. PARENT/CHILD QUALITY SCORING (Future implementation)
        if active_weights.get('parent_child_quality', 0) > 0 and node is not None:
            try:
                # TODO: Implement when parent/child scoring is ready
                pc_score = self.score_parent_child_quality(node, parent_nodes or [], child_nodes or [])
                component_scores['parent_child_quality'] = {'score': pc_score or 0.0, 'details': {}}
            except:
                component_scores['parent_child_quality'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        else:
            component_scores['parent_child_quality'] = {'score': 0.0, 'details': {'status': 'disabled'}}
        
        # 3. SEMANTIC SIMILARITY SCORING (Future implementation)
        if active_weights.get('semantic_similarity', 0) > 0 and reference_answers:
            try:
                # TODO: Implement when semantic similarity is ready
                generated_text = component_scores['confidence']['details'].get('text_generated', '')
                sim_score = self.score_semantic_similarity(generated_text, reference_answers)
                component_scores['semantic_similarity'] = {'score': sim_score or 0.0, 'details': {}}
            except:
                component_scores['semantic_similarity'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        else:
            component_scores['semantic_similarity'] = {'score': 0.0, 'details': {'status': 'disabled'}}
        
        # 4. LENGTH PENALTY (Simple implementation for now)
        if active_weights.get('length_penalty', 0) > 0:
            try:
                text = component_scores['confidence']['details'].get('text_generated', '')
                # Simple length penalty: penalize very short or very long answers
                length = len(text.split())
                if length < 2:
                    length_score = 0.5  # Too short
                elif length > 50:
                    length_score = max(0.3, 1.0 - (length - 50) * 0.01)  # Too long
                else:
                    length_score = 1.0  # Good length
                component_scores['length_penalty'] = {'score': length_score, 'details': {'word_count': length}}
            except:
                component_scores['length_penalty'] = {'score': 1.0, 'details': {'status': 'error'}}
        else:
            component_scores['length_penalty'] = {'score': 1.0, 'details': {'status': 'disabled'}}
        
        # 5. COHERENCE SCORING (Future implementation)
        component_scores['coherence'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        
        # 6. FACTUAL CONSISTENCY (Future implementation) 
        component_scores['factual_consistency'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        
        # Calculate weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        weights_used = {}
        
        for component, weight in active_weights.items():
            if weight > 0 and component in component_scores:
                score = component_scores[component]['score']
                weighted_sum += score * weight
                total_weight += weight
                weights_used[component] = weight
        
        # Calculate final score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0
        
        # Normalize to 0-1 range if requested
        if normalize:
            overall_score = max(0.0, min(1.0, overall_score))
        
        return {
            'overall_score': overall_score,
            'component_scores': component_scores,
            'weights_used': weights_used,
            'total_weight': total_weight,
            'metadata': {
                'question': question,
                'path': path,
                'normalized': normalize
            }
        }
    
    def get_simple_overall_score(
        self,
        question: str,
        path: str = "",
        custom_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Get a simple overall score (0-1) for easy integration with search algorithms.
        
        Args:
            question: The question being answered
            path: Current path/partial answer  
            custom_weights: Optional custom weights for scoring components
            
        Returns:
            Float score between 0.0 and 1.0
            
        Example:
            scorer = AnswerScorer()
            score = scorer.get_simple_overall_score("What is 2+2?", "")
            print(f"Score: {score:.3f}")
        """
        result = self.get_overall_score(question, path, weights=custom_weights)
        return result['overall_score']


# Convenience functions for easy integration with existing code

def get_answer_confidence(
    question: str, 
    path: str = "", 
    config: Optional[PolicyConfig] = None
) -> ConfidenceScore:
    """
    Convenience function to get confidence score for a single answer.
    
    Args:
        question: The question being answered
        path: Current path/partial answer
        config: Policy configuration (uses defaults if None)
        
    Returns:
        ConfidenceScore object
    """
    scorer = AnswerScorer(config)
    return scorer.get_confidence_score(question, path)


def get_simple_confidence(question: str, path: str = "") -> float:
    """
    Get a simple confidence score (0-1) using average log probability.
    
    Args:
        question: The question
        path: Current path/partial answer
        
    Returns:
        Confidence score between 0 and 1
    """
    result = get_answer_confidence(question, path)
    return result.avg_confidence or 0.0


def get_overall_answer_score(
    question: str, 
    path: str = "", 
    config: Optional[PolicyConfig] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Convenience function to get an overall score (0-1) for a single answer.
    This is the main function that search algorithms should use.
    
    Args:
        question: The question being answered
        path: Current path/partial answer
        config: Policy configuration (uses defaults if None)
        weights: Custom weights for scoring components
        
    Returns:
        Overall score between 0 and 1
        
    Example:
        # Basic usage
        score = get_overall_answer_score("What is 2+2?", "")
        
        # With custom weights (when more components are available)
        custom_weights = {"confidence": 0.7, "length_penalty": 0.3}
        score = get_overall_answer_score("What is 2+2?", "", weights=custom_weights)
    """
    scorer = AnswerScorer(config)
    return scorer.get_simple_overall_score(question, path, weights)


# Example usage and testing functions

def demo_scoring():
    """Demonstrate the scoring system."""
    scorer = AnswerScorer()
    
    # Test questions
    test_cases = [
        ("What is 2+2?", ""),
        ("What is 15*23?", "Let me calculate step by step."),
        ("Solve x^2 = 16", "I need to find the square root.")
    ]
    
    print("=== Confidence Scoring Demo ===")
    for question, path in test_cases:
        print(f"\nQuestion: {question}")
        print(f"Path: {path}")
        
        score = scorer.get_confidence_score(question, path, include_raw=False)
        
        print(f"Generated: {score.text}")
        print(f"Avg Confidence: {score.avg_confidence:.3f}" if score.avg_confidence else "N/A")
        print(f"Min Confidence: {score.min_confidence:.3f}" if score.min_confidence else "N/A")
        print(f"Perplexity: {score.perplexity:.2f}" if score.perplexity else "N/A")
        print("-" * 50)


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_scoring()
