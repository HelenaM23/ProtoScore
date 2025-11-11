"""Score calculation utilities for prototype evaluation."""

import math
from enum import Enum
from typing import Dict
import numpy as np


class AggregationMethod(Enum):
    """Supported aggregation methods for score calculation."""
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    HARMONIC = "harmonic"


class ScoreValidator:
    """Validates inputs for score calculations."""
    
    @staticmethod
    def validate_results(results: Dict[str, float]) -> None:
        """Validate results dictionary."""
        if not results:
            raise ValueError("Results dictionary cannot be empty")
        
        for metric_name, score in results.items():
            ScoreValidator._validate_score_value(metric_name, score)
    
    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> None:
        """Validate weights dictionary."""
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")
        
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("All weights must be non-negative")
        
        if sum(weights.values()) == 0:
            raise ValueError("Sum of weights must be greater than 0")
    
    @staticmethod
    def validate_method(method: str) -> AggregationMethod:
        """Validate and convert aggregation method."""
        try:
            return AggregationMethod(method)
        except ValueError:
            valid_methods = [m.value for m in AggregationMethod]
            raise ValueError(f"Invalid method '{method}'. Valid methods: {valid_methods}")
    
    @staticmethod
    def _validate_score_value(metric_name: str, score: float) -> None:
        """Validate individual score value."""
        if not isinstance(score, (int, float, np.integer, np.floating)):
            raise ValueError(f"Score for '{metric_name}' must be numeric, got {type(score)}")
        
        if np.isnan(score) or np.isinf(score):
            raise ValueError(
                f"Score for '{metric_name}' is invalid: {score}"
            )
        
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score for '{metric_name}' must be between 0 and 1, got {score}")


class WeightNormalizer:
    """Handles weight normalization and filtering."""
    
    @staticmethod
    def filter_and_normalize(
        results: Dict[str, float], 
        weights: Dict[str, float]
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Filter results by weights and normalize weights."""
        filtered_results = {
            key: value for key, value in results.items() 
            if key in weights and weights[key] > 0
        }
        
        if not filtered_results:
            return {}, {}
        
        filtered_weights = {key: weights[key] for key in filtered_results.keys()}
        normalized_weights = WeightNormalizer._normalize_weights(filtered_weights)
        
        return filtered_results, normalized_weights
    
    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total_weight = sum(weights.values())
        return {key: weight / total_weight for key, weight in weights.items()}


class AggregationCalculator:
    """Calculates aggregated scores using different methods."""
    
    @staticmethod
    def calculate(
        results: Dict[str, float], 
        weights: Dict[str, float], 
        method: AggregationMethod
    ) -> float:
        """Calculate aggregated score."""
        if method == AggregationMethod.ARITHMETIC:
            return AggregationCalculator._arithmetic_mean(results, weights)
        elif method == AggregationMethod.GEOMETRIC:
            return AggregationCalculator._geometric_mean(results, weights)
        elif method == AggregationMethod.HARMONIC:
            return AggregationCalculator._harmonic_mean(results, weights)
    
    @staticmethod
    def _arithmetic_mean(results: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted arithmetic mean."""
        return sum(results[key] * weights[key] for key in results.keys())
    
    @staticmethod
    def _geometric_mean(results: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted geometric mean."""
        if AggregationCalculator._has_zero_values_with_positive_weights(results, weights):
            return 0.0
        
        log_sum = sum(weights[key] * math.log(results[key]) for key in results.keys())
        return math.exp(log_sum)
    
    @staticmethod
    def _harmonic_mean(results: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted harmonic mean."""
        if AggregationCalculator._has_zero_values_with_positive_weights(results, weights):
            return 0.0
        
        weighted_reciprocal_sum = sum(weights[key] / results[key] for key in results.keys())
        return 1.0 / weighted_reciprocal_sum
    
    @staticmethod
    def _has_zero_values_with_positive_weights(
        results: Dict[str, float], 
        weights: Dict[str, float]
    ) -> bool:
        """Check if any zero values have positive weights."""
        return any(results[key] <= 0 and weights[key] > 0 for key in results.keys())


class ScoreCalculator:
    """Main class for score calculation with multiple aggregation methods."""
    
    @staticmethod
    def compute_final_score(
        results: Dict[str, float], 
        weights: Dict[str, float], 
        method: str = "arithmetic"
    ) -> float:
        """
        Compute final combined score from results using weighted aggregation.
        
        Args:
            results: Dictionary of metric name -> score (0-1)
            weights: Dictionary of metric name -> weight (>= 0)
            method: Aggregation method ('arithmetic', 'geometric', 'harmonic')
            
        Returns:
            Final combined score between 0 and 1
        """
        # Validate inputs
        ScoreValidator.validate_results(results)
        ScoreValidator.validate_weights(weights)
        aggregation_method = ScoreValidator.validate_method(method)
        
        # Filter and normalize
        filtered_results, normalized_weights = WeightNormalizer.filter_and_normalize(
            results, weights
        )
        
        if not filtered_results:
            return 0.0
        
        # Calculate score
        score = AggregationCalculator.calculate(
            filtered_results, normalized_weights, aggregation_method
        )
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def compute_score_breakdown(
        results: Dict[str, float], 
        weights: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute detailed breakdown of score contributions.
        
        Returns:
            Dictionary with breakdown information for each metric
        """
        ScoreValidator.validate_results(results)
        ScoreValidator.validate_weights(weights)
        
        breakdown = {}
        total_weight = sum(weights.values())
        
        for metric, score in results.items():
            if metric in weights:
                weight = weights[metric]
                normalized_weight = weight / total_weight if total_weight > 0 else 0
                contribution = score * normalized_weight
                
                breakdown[metric] = {
                    'score': score,
                    'weight': weight,
                    'normalized_weight': normalized_weight,
                    'contribution': contribution
                }
        
        return breakdown