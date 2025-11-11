"""
Model wrapper classes for prototype-based explainable AI methods.

This module implements the Adapter pattern to provide a unified interface
for different types of models (encoders, decoders, classifiers).
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import keras
import tensorflow as tf



class Callable(Protocol):
    """Protocol for callable models."""
    def __call__(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        ...


class ModelWrapper(ABC):
    """
    Abstract base wrapper implementing the Adapter pattern.
    
    Provides a unified interface for different model types while maintaining
    the Single Responsibility Principle.
    """
    
    def __init__(self, model: Callable) -> None:
        """
        Initialize wrapper with a model.
        
        Args:
            model: The model to wrap (must be callable)
            
        Raises:
            TypeError: If model is not callable
        """
        if not callable(model):
            raise TypeError("Model must be callable")
        self._model = model
    
    @abstractmethod
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Execute the wrapped model on inputs."""
        pass
    
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Make the wrapper callable."""
        return self.call(inputs)
    
    @property
    def model(self) -> Callable:
        """Get the wrapped model."""
        return self._model


class EncoderWrapper(ModelWrapper):
    """
    Wrapper for encoder models that transform inputs to latent representations.
    
    Follows the Single Responsibility Principle by handling only encoding logic.
    """
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Encode inputs to latent embeddings.
        
        Args:
            inputs: Input tensor to encode
            
        Returns:
            Encoded embeddings tensor
            
        Raises:
            RuntimeError: If encoding fails
        """
        try:
            return self._model(inputs)
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}") from e


class DecoderWrapper(ModelWrapper):
    """
    Wrapper for decoder models that reconstruct from latent representations.
    
    Follows the Single Responsibility Principle by handling only decoding logic.
    """
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Decode latent representations to outputs.
        
        Args:
            inputs: Latent representation tensor to decode
            
        Returns:
            Decoded output tensor
            
        Raises:
            RuntimeError: If decoding fails
        """
        try:
            return self._model(inputs)
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {str(e)}") from e


class ClassifierWrapper(ModelWrapper):
    """
    Wrapper for classifier models that predict class labels.
    
    Follows the Single Responsibility Principle by handling only classification logic.
    """
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Classify inputs and return predictions.
        
        Args:
            inputs: Input tensor to classify
            
        Returns:
            Classification predictions tensor
            
        Raises:
            RuntimeError: If classification fails
        """
        try:
            return self._model(inputs)
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}") from e


class CompositeClassifier(keras.Model):
    """
    Composite model combining encoder and predictor using the Composite pattern.
    
    This class demonstrates the Composite pattern by treating the combination 
    of encoder and predictor as a single unit.
    """
    
    def __init__(
        self, 
        encoder: Callable, 
        predictor: Callable, 
        name: str = "composite_classifier",
        **kwargs
    ) -> None:
        """
        Initialize the composite classifier.
        
        Args:
            encoder: The encoder component
            predictor: The predictor component  
            name: Model name
            **kwargs: Additional keras.Model arguments
            
        Raises:
            TypeError: If encoder or predictor is not callable
        """
        if not callable(encoder):
            raise TypeError("Encoder must be callable")
        if not callable(predictor):
            raise TypeError("Predictor must be callable")
            
        super().__init__(name=name, **kwargs)
        self._encoder = encoder
        self._predictor = predictor
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through encoder then predictor.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Classification predictions
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            latent = self._encoder(inputs, training=training)
            return self._predictor(latent, training=training)
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}") from e
    
    @property
    def encoder(self) -> Callable:
        """Get the encoder component."""
        return self._encoder
    
    @property
    def predictor(self) -> Callable:
        """Get the predictor component."""
        return self._predictor


class WrapperFactory:
    """
    Factory for creating model wrappers.
    
    Implements the Factory pattern to encapsulate wrapper creation logic.
    """
    
    @staticmethod
    def create_encoder_wrapper(model: Callable) -> EncoderWrapper:
        """Create an encoder wrapper."""
        return EncoderWrapper(model)
    
    @staticmethod
    def create_decoder_wrapper(model: Callable) -> DecoderWrapper:
        """Create a decoder wrapper."""
        return DecoderWrapper(model)
    
    @staticmethod
    def create_classifier_wrapper(
        model: Callable, 
    ) -> ClassifierWrapper:
        """Create a classifier wrapper."""
        return ClassifierWrapper(model)