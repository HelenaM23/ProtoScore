"""Base classes for evaluation metrics and distance calculations."""

from typing import Optional, Protocol

import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf

from .dtw_utils import DTWUtils


class DistanceCalculator(Protocol):
    """Protocol for distance calculations."""
    
    def compute_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distance matrix between X and Y."""
        ...


class EuclideanDistance:
    """Euclidean distance calculator implementation."""
    
    def compute_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix between X and Y."""
        return cdist(X, Y, metric='euclidean')


class DTWDistance:
    """DTW distance calculator implementation."""
    
    def __init__(self, decoder) -> None:
        if decoder is None:
            raise ValueError("Decoder is required for DTW distance calculation")
        self.decoder = decoder
    
    def compute_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute DTW distance matrix with improved preprocessing."""
        if self._is_empty(X) or self._is_empty(Y):
            raise ValueError("Input arrays cannot be empty")
        
        X_decoded = self.decoder(X)
        Y_decoded = self.decoder(Y)
        
        self._validate_decoded_data(X_decoded, Y_decoded)

        return DTWUtils.dtw_cdist_with_progress(X_decoded, Y_decoded)
    
    def _is_empty(self, data):
        """Check if data is empty, works for both tf.Tensor and np.ndarray."""
        if hasattr(data, 'numpy'): 
            return tf.size(data) == 0
        else:
            return data.size == 0
    
    def _validate_decoded_data(self, X_decoded, Y_decoded):
        """Validiere dekodierte Daten."""
        if len(X_decoded) == 0 or len(Y_decoded) == 0:
            raise ValueError("Decoded sequences cannot be empty")
        
        if hasattr(X_decoded[0], 'shape') and hasattr(Y_decoded[0], 'shape'):
            x_dims = X_decoded[0].shape
            y_dims = Y_decoded[0].shape
            if len(x_dims) != len(y_dims):
                raise ValueError(f"Dimension mismatch: X={x_dims}, Y={y_dims}")


class DistanceCalculatorFactory:
    """Factory for creating distance calculators."""
    
    _SUPPORTED_METRICS = {"euclidean", "DTW"}
    
    @classmethod
    def create(cls, metric: str, decoder: Optional = None) -> DistanceCalculator:
        """Create appropriate distance calculator based on metric."""
        if metric not in cls._SUPPORTED_METRICS:
            raise ValueError(f"Unsupported distance metric: {metric}. "
                           f"Supported: {cls._SUPPORTED_METRICS}")

        if metric == "euclidean":
            return EuclideanDistance()
        elif metric == "DTW":
            return DTWDistance(decoder)

