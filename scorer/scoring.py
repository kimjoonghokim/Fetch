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
    REPEATING_NODE = "repeating_node"  # NEW: Repeating node scoring
    # Future methods to be implemented:
    # PARENT_CHILD_QUALITY = "parent_child_quality"
    # SEMANTIC_SIMILARITY = "semantic_similarity"
    # ENSEMBLE_VOTING = "ensemble_voting"

#class ParentScore is not needed because Parent scores are already calculated using the existing ConfidenceScore class and they just become multipliers

@dataclass
class VoteScore:
    """Container for vote scoring results"""
    text: str
    vote_score: Optional[float] = None
    merge_count: int = 0  # Number of semantic similarity merges
    cluster_sizes: Optional[List[int]] = None  # Sizes of merged clusters
    semantic_groups: Optional[List[List[str]]] = None  # Groups of semantically similar texts
    raw_response: Optional[Dict] = None

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
class RepeatingNodeScore:
    """Container for repeating node scoring results"""
    text: str
    repeating_score: Optional[float] = None
    early_boost: float = 0.0  # Boost for appearing earlier
    similar_nodes_found: int = 0  # Number of similar nodes found later
    node_depth: int = 0  # Depth of this node in the tree
    similar_node_depths: Optional[List[int]] = None  # Depths of similar nodes
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
    
    # ESM service configuration
    esm_url: str = "http://127.0.0.1"
    esm_port: int = 8003
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ["\n"]
    
    @property
    def endpoint_url(self) -> str:
        return f"{self.url}:{self.port}/v1/completions"
    
    @property
    def esm_endpoint_url(self) -> str:
        return f"{self.esm_url}:{self.esm_port}/predict"


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
    
    def score_parent_quality(self, node, parent_nodes: List, child_nodes: List) -> float:
        """
        Score based on parent node quality.
        
        Scoring logic:
        - Parent's confidence score gets normalized to create visible multiplier differences
        - HIGHER parent confidence = HIGHER multiplier (boosts children)
        - LOWER parent confidence = LOWER multiplier (penalizes children)
        - Formula: 0.5 + (parent_confidence * 0.5)
        
        Args:
            node: Current node being scored
            parent_nodes: List of parent nodes
            child_nodes: List of child nodes (for future use)
            
        Returns:
            Parent quality score (0.1 to 1.0)
        """
        if not parent_nodes:
            return 1.0  # No parents = neutral multiplier
        
        try:
            # Calculate average parent confidence score
            parent_scores = []
            
            for parent in parent_nodes:
                # Handle both dictionary objects and objects with attributes
                if isinstance(parent, dict):
                    if 'question' in parent and 'path' in parent:
                        parent_confidence = self.get_confidence_score(
                            question=parent['question'],
                            path=parent['path'],
                            method=ScoringMethod.CONFIDENCE_AVERAGE
                        )
                        parent_score = parent_confidence.avg_confidence or 0.0
                        parent_scores.append(parent_score)
                    else:
                        parent_scores.append(0.5)
                elif hasattr(parent, 'question') and hasattr(parent, 'path'):
                    parent_confidence = self.get_confidence_score(
                        question=parent.question,
                        path=parent.path,
                        method=ScoringMethod.CONFIDENCE_AVERAGE
                    )
                    parent_score = parent_confidence.avg_confidence or 0.0
                    parent_scores.append(parent_score)
                elif hasattr(parent, 'overall_score'):
                    parent_scores.append(parent.overall_score)
                else:
                    parent_scores.append(0.5)
            
            # Calculate average parent confidence score
            avg_parent_confidence = np.mean(parent_scores) if parent_scores else 0.5
            
            # Transform confidence to create correct multiplier direction
            # Formula: 0.5 + (confidence * 0.5)
            parent_multiplier = 0.5 + (avg_parent_confidence * 0.5)
            
            # Cap the multiplier to reasonable bounds (0.1 to 1.0)
            parent_multiplier = max(0.1, min(1.0, parent_multiplier))
            
            return parent_multiplier
            
        except Exception as e:
            print(f"Warning: Parent-child scoring failed: {e}")
            return 1.0  # Fallback to neutral multiplier
    
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
    
    def check_esm_service(self) -> bool:
        """
        Check if ESM service is available.
        
        Returns:
            True if ESM service is reachable, False otherwise
        """
        try:
            import requests
            
            # Try to connect to the ESM service
            response = requests.get(f"{self.config.esm_url}:{self.config.esm_port}/health", timeout=5)
            return response.status_code == 200
        except:
            # Try a simple predict call with minimal data
            try:
                response = requests.post(
                    self.config.esm_endpoint_url, 
                    json={"texts": ["test"], "d": 0.1}, 
                    timeout=5
                )
                return response.status_code == 200
            except:
                return False
    
    def get_vote_score(
        self,
        question: str,
        path: str = "",
        candidate_paths: Optional[List[str]] = None,
        similarity_threshold: float = 0.15,
        include_raw: bool = False
    ) -> VoteScore:
        """
        Get vote score based on semantic similarity merges.
        
        Args:
            question: The question being answered
            path: Current path/partial answer
            candidate_paths: List of candidate paths to compare against
            similarity_threshold: Distance threshold for semantic clustering (0.0-1.0)
            include_raw: Whether to include raw clustering results
            
        Returns:
            VoteScore object with voting results
        """
        try:
            if not candidate_paths:
                # No candidates to compare against
                return VoteScore(
                    text=path,
                    vote_score=1.0,
                    merge_count=0,
                    cluster_sizes=[1],
                    semantic_groups=[[path]] if path else []
                )
            
            # Add current path to candidates for clustering
            all_paths = [path] + candidate_paths if path else candidate_paths
            
            # Check ESM service availability
            esm_available = self.check_esm_service()
            
            # Perform semantic clustering
            if esm_available:
                clusters = self._cluster_semantically(all_paths, similarity_threshold)
                clustering_method = "ESM"
            else:
                print("Warning: ESM service unavailable, using fallback clustering")
                clusters = self._fallback_clustering_simple(all_paths, similarity_threshold)
                clustering_method = "fallback"
            
            # Find the cluster containing the current path
            current_cluster = None
            for cluster in clusters:
                if path in cluster:
                    current_cluster = cluster
                    break
            
            if not current_cluster:
                # Current path not found in any cluster (shouldn't happen)
                return VoteScore(
                    text=path,
                    vote_score=1.0,
                    merge_count=0,
                    cluster_sizes=[1],
                    semantic_groups=[[path]] if path else []
                )
            
            # Calculate vote score based on cluster size
            cluster_size = len(current_cluster)
            merge_count = max(0, cluster_size - 1)  # Number of potential merges
            
            # Base score is 1.0, each merge adds 0.1
            vote_score = 1.0 + (merge_count * 0.1)
            
            # Cap the score at 2.0
            vote_score = min(2.0, vote_score)
            
            # Get all cluster sizes for analysis
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            raw_response = {
                "clusters": clusters, 
                "threshold": similarity_threshold,
                "clustering_method": clustering_method,
                "esm_available": esm_available
            } if include_raw else None
            
            return VoteScore(
                text=path,
                vote_score=vote_score,
                merge_count=merge_count,
                cluster_sizes=cluster_sizes,
                semantic_groups=clusters,
                raw_response=raw_response
            )
            
        except Exception as e:
            print(f"Error calculating vote score: {e}")
            return VoteScore(
                text=path,
                vote_score=1.0,
                merge_count=0,
                cluster_sizes=[1],
                semantic_groups=[[path]] if path else [],
                raw_response={"error": str(e)}
            )
    
    def _cluster_semantically(self, texts: List[str], threshold: float) -> List[List[str]]:
        """
        Perform semantic clustering on texts using ESM service.
        
        Args:
            texts: List of text strings to cluster
            threshold: Distance threshold for clustering (0.0-1.0)
            
        Returns:
            List of clusters, where each cluster is a list of similar texts
        """
        if len(texts) <= 1:
            return [texts] if texts else []
        
        try:
            # Call ESM service for semantic clustering
            labels = self._call_esm_service(texts, threshold)
            
            # Group texts by cluster labels
            clusters = {}
            for text, label in zip(texts, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text)
            
            # Return list of clusters
            return list(clusters.values())
            
        except Exception as e:
            print(f"Error in ESM semantic clustering: {e}")
            # Fallback: each text is its own cluster
            return [[text] for text in texts]
    
    def _call_esm_service(self, texts: List[str], threshold: float) -> List[int]:
        """
        Call ESM service for semantic clustering.
        
        Args:
            texts: List of text strings to cluster
            threshold: Distance threshold for clustering
            
        Returns:
            List of cluster labels for each text
        """
        try:
            import requests
            
            # ESM service endpoint from configuration
            url = self.config.esm_endpoint_url
            
            # Prepare payload
            payload = {
                "texts": texts,
                "d": threshold
            }
            
            # Make request to ESM service
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            # Extract cluster labels
            result = response.json()
            return result["labels"]
            
        except Exception as e:
            print(f"ESM service call failed: {e}")
            # Fallback to simple clustering if ESM service is unavailable
            return self._fallback_clustering(texts, threshold)
    
    def _fallback_clustering(self, texts: List[str], threshold: float) -> List[int]:
        """
        Fallback clustering method when ESM service is unavailable.
        Uses simple word overlap similarity.
        
        Args:
            texts: List of text strings to cluster
            threshold: Similarity threshold for clustering
            
        Returns:
            List of cluster labels for each text
        """
        if len(texts) <= 1:
            return [0] if texts else []
        
        clusters = []
        used_indices = set()
        next_label = 0
        
        for i, text1 in enumerate(texts):
            if i in used_indices:
                continue
            
            # Start a new cluster
            cluster_label = next_label
            next_label += 1
            clusters.append(cluster_label)
            used_indices.add(i)
            
            # Find similar texts
            for j, text2 in enumerate(texts[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Simple similarity check
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity >= threshold:
                    clusters.append(cluster_label)
                    used_indices.add(j)
                else:
                    clusters.append(next_label)
                    next_label += 1
                    used_indices.add(j)
        
        return clusters
    
    def _fallback_clustering_simple(self, texts: List[str], threshold: float) -> List[List[str]]:
        """
        Simple fallback clustering when ESM service is unavailable.
        Each text gets its own cluster.
        
        Args:
            texts: List of text strings to cluster
            threshold: Distance threshold (ignored in fallback)
            
        Returns:
            List of clusters, where each cluster contains one text
        """
        return [[text] for text in texts]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        This is a simplified version - in practice, use the ESM service.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple word overlap similarity as fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_repeating_node_score(
        self,
        question: str,
        path: str = "",
        node_depth: int = 0,
        all_tree_paths: Optional[List[Tuple[str, int]]] = None,
        similarity_threshold: float = 0.15,
        early_boost_factor: float = 0.3,  # Increased from 0.1 to 0.3 for more noticeable boosts
        include_raw: bool = False
    ) -> RepeatingNodeScore:
        """
        Calculate repeating node score based on semantic similarity with earlier reasoning.
        
        Scoring logic:
        - Base score: 1.0
        - Early boost: (1 + num_similar_nodes) * early_boost_factor * depth_advantage
        - Depth advantage: (max_depth - current_depth) / max_depth
        - Later nodes get minimal boost, earlier nodes get larger boost
        """
        if not all_tree_paths or len(all_tree_paths) < 2:
            return RepeatingNodeScore(
                text=path,
                repeating_score=1.0,
                early_boost=0.0,
                similar_nodes_found=0,
                node_depth=node_depth
            )
        
        # DEBUG: Print all tree paths for inspection
        print(f"\n🔍 DEBUG: Analyzing repeating node score for path: '{path}' at depth {node_depth}")
        print(f"📊 All tree paths ({len(all_tree_paths)} total):")
        for i, (p, d) in enumerate(all_tree_paths):
            print(f"   [{i}] Depth {d}: '{p}'")
        
        similar_nodes = []
        max_depth = max(depth for _, depth in all_tree_paths)
        
        # Find semantically similar nodes that appear later in the tree
        for other_path, other_depth in all_tree_paths:
            if other_depth > node_depth:  # Only look at deeper nodes
                similarity = self._calculate_text_similarity(path, other_path)
                if similarity >= similarity_threshold:
                    similar_nodes.append((other_path, other_depth, similarity))
        
        if not similar_nodes:
            return RepeatingNodeScore(
                text=path,
                repeating_score=1.0,
                early_boost=0.0,
                similar_nodes_found=0,
                node_depth=node_depth
            )
        
        # Calculate early boost based on depth advantage
        num_similar = len(similar_nodes)
        depth_advantage = (max_depth - node_depth) / max_depth  # Higher for shallower nodes
        
        # Early boost formula: (1 + num_similar) * early_boost_factor * depth_advantage
        early_boost = (1 + num_similar) * early_boost_factor * depth_advantage
        
        # Final score: base + early boost
        final_score = 1.0 + early_boost
        
        # Store similar node depths for debugging
        similar_depths = [depth for _, depth, _ in similar_nodes]
        
        print(f"📊 Similar nodes found: {num_similar}")
        print(f"   • Depth advantage: {depth_advantage:.3f}")
        print(f"   • Early boost: {early_boost:.3f}")
        print(f"   • Final score: {final_score:.3f}")
        
        return RepeatingNodeScore(
            text=path,
            repeating_score=final_score,
            early_boost=early_boost,
            similar_nodes_found=num_similar,
            node_depth=node_depth,
            similar_node_depths=similar_depths,
            raw_response={"similar_nodes": similar_nodes} if include_raw else None
        )
    
    def get_overall_score(
        self,
        question: str,
        path: str = "",
        node=None,
        parent_nodes: Optional[List] = None,
        child_nodes: Optional[List] = None,
        reference_answers: Optional[List[str]] = None,
        candidate_paths: Optional[List[str]] = None,
        all_tree_paths: Optional[List[Tuple[str, int]]] = None,  # NEW: for repeating node scoring
        node_depth: int = 0,  # NEW: depth of current node
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
            candidate_paths: List of candidate paths for vote scoring
            all_tree_paths: List of all paths in tree with depths for repeating node scoring
            node_depth: Depth of current node in the tree for repeating node scoring
            weights: Custom weights for different scoring components
            normalize: Whether to normalize the final score to 0-1 range
            
        Returns:
            Dictionary containing:
            - 'overall_score': Final weighted score (0-1 if normalized)
            - 'component_scores': Individual scores from each method
            - 'weights_used': The weights applied to each component
        """
        
        # Default weights for scoring components
        default_weights = {
            'confidence': 0.4,          # Reduced to make room for parent quality
            'vote_score': 0.25,         # Reduced slightly
            'repeating_node': 0.2,      # Keep repeating node scoring
            'parent_quality': 0.15,     # RENAMED: Parent quality scoring (was parent_child_quality)
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
        
        # 2. VOTE SCORING (Based on semantic similarity merges)
        if active_weights.get('vote_score', 0) > 0:
            try:
                vote_result = self.get_vote_score(question, path, candidate_paths)
                # Normalize vote score from 1.0-2.0 range to 0.0-1.0 range
                normalized_vote_score = (vote_result.vote_score - 1.0)  # Now 0.0-1.0
                component_scores['vote_score'] = {
                    'score': normalized_vote_score,
                    'details': {
                        'raw_vote_score': vote_result.vote_score,
                        'merge_count': vote_result.merge_count,
                        'cluster_sizes': vote_result.cluster_sizes,
                        'semantic_groups': vote_result.semantic_groups
                    }
                }
            except Exception as e:
                print(f"Warning: Vote scoring failed: {e}")
                component_scores['vote_score'] = {'score': 0.0, 'details': {'error': str(e)}}
        else:
            component_scores['vote_score'] = {'score': 0.0, 'details': {'status': 'disabled'}}
        
        # 3. REPEATING NODE SCORING (NEW: Rewards earlier reasoning when similar reasoning appears later)
        if active_weights.get('repeating_node', 0) > 0:
            try:
                repeating_result = self.get_repeating_node_score(
                    question, path, node_depth, all_tree_paths
                )
                # Normalize repeating node score from 1.0-2.0 range to 0.0-1.0 range
                normalized_repeating_score = (repeating_result.repeating_score - 1.0)  # Now 0.0-1.0
                component_scores['repeating_node'] = {
                    'score': normalized_repeating_score,
                    'details': {
                        'raw_repeating_score': repeating_result.repeating_score,
                        'early_boost': repeating_result.early_boost,
                        'similar_nodes_found': repeating_result.similar_nodes_found,
                        'node_depth': repeating_result.node_depth,
                        'similar_node_depths': repeating_result.similar_node_depths
                    }
                }
            except Exception as e:
                print(f"Warning: Repeating node scoring failed: {e}")
                component_scores['repeating_node'] = {'score': 0.0, 'details': {'error': str(e)}}
        else:
            component_scores['repeating_node'] = {'score': 0.0, 'details': {'status': 'disabled'}}
        
        # 4. PARENT QUALITY SCORING (Now implemented!)
        if active_weights.get('parent_quality', 0) > 0 and node is not None and parent_nodes:
            try:
                pc_score = self.score_parent_quality(node, parent_nodes, child_nodes or [])
                component_scores['parent_quality'] = {
                    'score': pc_score, 
                    'details': {
                        'parent_multiplier': pc_score,
                        'num_parents': len(parent_nodes),
                        'num_children': len(child_nodes or [])
                    }
                }
            except Exception as e:
                print(f"Warning: Parent quality scoring failed: {e}")
                component_scores['parent_quality'] = {'score': 1.0, 'details': {'error': str(e)}}
        else:
            component_scores['parent_quality'] = {'score': 1.0, 'details': {'status': 'disabled'}}
        
        # 5. SEMANTIC SIMILARITY SCORING (Future implementation)
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
        
        # 6. LENGTH PENALTY (Simple implementation for now)
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
            # Don't include length_penalty in component_scores when weight is 0
            pass
        
        # 7. COHERENCE SCORING (Future implementation)
        if active_weights.get('coherence', 0) > 0:
            component_scores['coherence'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        
        # 8. FACTUAL CONSISTENCY (Future implementation) 
        if active_weights.get('factual_consistency', 0) > 0:
            component_scores['factual_consistency'] = {'score': 0.0, 'details': {'status': 'not_implemented'}}
        
        # Calculate weighted overall score
        total_score = 0.0
        total_weight = 0.0
        
        # Get parent multiplier for child scoring
        parent_multiplier = component_scores.get('parent_quality', {}).get('score', 1.0)
        
        # FIXED: Only include components that have actual scores and weights > 0
        for component, weight in active_weights.items():
            if weight > 0 and component in component_scores:
                score = component_scores[component]['score']
                
                # Apply parent multiplier to child scores (except parent_quality itself)
                if component != 'parent_quality' and parent_multiplier != 1.0:
                    score = score * parent_multiplier
                    # Update the component score to show the adjusted value
                    component_scores[component]['score'] = score
                    component_scores[component]['details']['parent_adjusted'] = True
                    component_scores[component]['details']['parent_multiplier'] = parent_multiplier
                
                total_score += score * weight
                total_weight += weight
        
        # Normalize to 0-1 range if requested and weights are available
        if normalize and total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = total_score
        
        return {
            'overall_score': overall_score,
            'component_scores': component_scores,
            'weights_used': active_weights
        }
    
    def get_simple_overall_score(
        self,
        question: str,
        path: str = "",
        candidate_paths: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Get a simple overall score (0-1) for easy integration with search algorithms.
        
        Args:
            question: The question being answered
            path: Current path/partial answer  
            candidate_paths: List of candidate paths for vote scoring
            custom_weights: Optional custom weights for scoring components
            
        Returns:
            Float score between 0.0 and 1.0
            
        Example:
            scorer = AnswerScorer()
            score = scorer.get_simple_overall_score("What is 2+2?", "", ["Let me solve", "I'll calculate"])
            print(f"Score: {score:.3f}")
        """
        result = self.get_overall_score(question, path, candidate_paths=candidate_paths, weights=custom_weights)
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
    candidate_paths: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Convenience function to get an overall score (0-1) for a single answer.
    This is the main function that search algorithms should use.
    
    Args:
        question: The question being answered
        path: Current path/partial answer
        config: Policy configuration (uses defaults if None)
        candidate_paths: List of candidate paths for vote scoring
        weights: Custom weights for scoring components
        
    Returns:
        Overall score between 0 and 1
        
    Example:
        # Basic usage
        score = get_overall_answer_score("What is 2+2?", "")
        
        # With candidate paths for vote scoring
        candidates = ["Let me solve this", "I'll calculate step by step"]
        score = get_overall_answer_score("What is 2+2?", "", candidate_paths=candidates)
        
        # With custom weights (when more components are available)
        custom_weights = {"confidence": 0.7, "vote_score": 0.3}
        score = get_overall_answer_score("What is 2+2?", "", candidate_paths=candidates, weights=custom_weights)
    """
    scorer = AnswerScorer(config)
    return scorer.get_simple_overall_score(question, path, candidate_paths, weights)


def get_vote_score_simple(
    question: str, 
    path: str = "", 
    candidate_paths: Optional[List[str]] = None,
    config: Optional[PolicyConfig] = None
) -> float:
    """
    Get a simple vote score (1.0-2.0) based on semantic similarity merges.
    
    Args:
        question: The question being answered
        path: Current path/partial answer
        candidate_paths: List of candidate paths to compare against
        config: Policy configuration (uses defaults if None)
        
    Returns:
        Vote score between 1.0 and 2.0 (higher = more merges)
        
    Example:
        # Basic usage
        score = get_vote_score_simple("What is 2+2?", "")
        
        # With candidate paths
        candidates = ["Let me solve this", "I'll calculate step by step"]
        score = get_vote_score_simple("What is 2+2?", "", candidates)
        print(f"Vote score: {score:.3f}")  # Will be 1.0-2.0
        
        # For detailed information, use AnswerScorer directly:
        # scorer = AnswerScorer()
        # details = scorer.get_vote_score(question, path, candidates, include_raw=True)
    """
    scorer = AnswerScorer(config)
    vote_result = scorer.get_vote_score(question, path, candidate_paths)
    return vote_result.vote_score or 1.0


def get_repeating_node_score_simple(
    question: str, 
    path: str = "", 
    node_depth: int = 0,
    all_tree_paths: Optional[List[Tuple[str, int]]] = None,
    config: Optional[PolicyConfig] = None
) -> float:
    """
    Simple function to get repeating node score.
    
    Args:
        question: The question being answered
        path: Current path/partial answer
        node_depth: Depth of current node in the tree
        all_tree_paths: List of all paths in tree with depths [(path, depth), ...]
        config: Policy configuration (uses defaults if None)
        
    Returns:
        Repeating node score (1.0-2.0 range)
    """
    scorer = AnswerScorer(config)
    result = scorer.get_repeating_node_score(question, path, node_depth, all_tree_paths)
    return result.repeating_score


def check_esm_service_status(config: Optional[PolicyConfig] = None) -> bool:
    """
    Check if ESM service is available.
    
    Args:
        config: Policy configuration (uses defaults if None)
        
    Returns:
        True if ESM service is reachable, False otherwise
        
    Example:
        if check_esm_service_status():
            print("ESM service is available")
        else:
            print("ESM service is unavailable")
    """
    scorer = AnswerScorer(config)
    return scorer.check_esm_service()


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