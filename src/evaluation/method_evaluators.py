"""
Complete method evaluator module using helper.py classes correctly.
"""

import abc
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Type

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
import os

from ..core.data import DatasetFactory
from ..evaluation.evaluator import PrototypeEvaluator
from ..utils.helper import (ConfigurationManager, DataProcessingError,
                            FileOperationError, PickleHandler, PrototypeExtractor,)
from ..utils.visualize import ClusteringVisualizer
from ..utils.visualize import PlotConfig
from ..core.model import FCNBlock
from ..core.wrapper import (ClassifierWrapper, CompositeClassifier,
                                 DecoderWrapper, EncoderWrapper)

from ..methods.map.map_explainers.ae1d_1e_3d import AutoEncoder as AutoEncoderMAP
from ..methods.msp.msp_explainers.prototype1 import AutoEncoder as AutoEncoderMSP
from ..methods.ebe.ebe_explainer import separate_model
from ..methods.ebe.helper import build_fcn_block_functional


            

logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

class EvaluationConfig:
    """Configuration class for evaluation parameters."""
    
    def __init__(self, cfg: Any):
        """
        Initialize configuration from Hydra config.
        
        Args:
            cfg: Hydra configuration object
        """
        self.distance_metric = getattr(cfg, 'distance_metric', 'euclidean')
        self.criteria = getattr(cfg, 'evaluation_metrics', None)
        self.metric_weights = self._normalize_weights(getattr(cfg, 'metric_weights', {}))
        self.visualize = getattr(cfg, 'visualize_clustering', False)
        
        # Parse clustering configuration from Hydra - entferne Duplikate
        self.clustering_config = self._parse_clustering_config(cfg)
        
        # Parse plot configuration from Hydra
        self.plot_config = self._parse_plot_config(cfg)
        
        logger.debug(f"Initialized config: distance_metric={self.distance_metric}, "
                    f"visualize={self.visualize}")
    
    def _parse_clustering_config(self, cfg: Any) -> Dict[str, Any]:
        """Parse clustering configuration from Hydra config."""
        clustering_cfg = getattr(cfg, 'clustering_config', {})
        
        return {
            'default_random_state': getattr(clustering_cfg, 'default_random_state', 42),
            'default_n_init': getattr(clustering_cfg, 'default_n_init', 10),
            'min_clusters': getattr(clustering_cfg, 'min_clusters', 2),
            'max_clusters': getattr(clustering_cfg, 'max_clusters', 15),
            'min_silhouette_samples': getattr(clustering_cfg, 'min_silhouette_samples', 10),
            'default_n_components': getattr(clustering_cfg, 'default_n_components', 2),
            'max_cache_size': getattr(clustering_cfg, 'max_cache_size', 128)
        }
        
    def _parse_plot_config(self, cfg: Any) -> Dict[str, Any]:
        """Parse plot configuration from Hydra config."""
        plot_cfg = getattr(cfg, 'plot_config', {})
        
        return {
            'figure_size': getattr(plot_cfg, 'figure_size', [15, 10]),
            'dpi': getattr(plot_cfg, 'dpi', 300),
            'save_format': getattr(plot_cfg, 'save_format', 'pdf'),
            'xlabel_fontsize': getattr(plot_cfg, 'xlabel_fontsize', 25),
            'ylabel_fontsize': getattr(plot_cfg, 'ylabel_fontsize', 25),
            'tick_labelsize': getattr(plot_cfg, 'tick_labelsize', 20),
            'legend_fontsize': getattr(plot_cfg, 'legend_fontsize', 20),
            'legend_title_fontsize': getattr(plot_cfg, 'legend_title_fontsize', 22),
            'legend_position': getattr(plot_cfg, 'legend_position', 'upper left'),
            'legend_bbox_to_anchor': getattr(plot_cfg, 'legend_bbox_to_anchor', [1.05, 1.0]),
            'marker_size_embeddings': getattr(plot_cfg, 'marker_size_embeddings', 50),
            'marker_size_centroids': getattr(plot_cfg, 'marker_size_centroids', 400),
            'marker_size_prototypes': getattr(plot_cfg, 'marker_size_prototypes', 300),
            'line_alpha': getattr(plot_cfg, 'line_alpha', 1.5),
            'edge_linewidth': getattr(plot_cfg, 'edge_linewidth', 2.5),
            'dimensionality_reduction_method': getattr(plot_cfg, 'dimensionality_reduction_method', 'pca'),
            'default_random_state': getattr(plot_cfg, 'default_random_state', 42),
            'default_n_components': getattr(plot_cfg, 'default_n_components', 2)
        }
    
    def _normalize_weights(self, weights_dict: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Normalize metric weights to sum to 1.
        
        Args:
            weights_dict: Dictionary of metric weights
            
        Returns:
            Normalized weights or None if empty
        """
        if not weights_dict:
            return None
        
        total_weight = sum(weights_dict.values())
        if total_weight == 0:
            logger.warning("Total weight is zero, returning None")
            return None
            
        normalized = {k: v / total_weight for k, v in weights_dict.items()}
        logger.debug(f"Normalized weights: {normalized}")
        return normalized


# ==================== BASE EVALUATOR ====================


class BaseEvaluator(abc.ABC):
    """
    Abstract base class for method evaluators using Template Method pattern.
    
    This class defines the common evaluation workflow while allowing
    subclasses to implement method-specific details.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluator = PrototypeEvaluator(
            distance_metric=config.distance_metric,
            metric_weights=config.metric_weights,
            clustering_config=config.clustering_config
        )
        
        # Initialize helper classes
        self.prototype_extractor = PrototypeExtractor()
        self.config_manager = ConfigurationManager()
        self.pickle_handler = PickleHandler()
        
        # Initialize visualization components
        if self.config.visualize:
            plot_config = self._create_plot_config()
            self.visualizer = ClusteringVisualizer(plot_config)

    @classmethod
    @abc.abstractmethod
    def prepare_evaluation_params(
        cls, 
        config: DictConfig, 
        dataset: Any
    ) -> Dict[str, Any]:
        """
        Prepare method-specific evaluation parameters from configuration.
        
        Args:
            config: Hydra configuration object
            dataset: Dataset instance
            
        Returns:
            Dictionary of evaluation parameters
        """
        pass
    
    def _create_plot_config(self) -> PlotConfig:
        """Create PlotConfig from configuration."""
        return PlotConfig(self.config.plot_config)
    
    def evaluate(
        self,
        weight_path: str,
        directory: str,
        label_legends: list,
        dataset: Any,
        weight_path_replica: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Template method for evaluation workflow.
        
        Args:
            weight_path: Path to model weights
            directory: Output directory
            label_legends: Label legends for visualization
            dataset: Dataset instance
            weight_path_replica: List of paths to replica model weights
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            Exception: If evaluation fails
        """
        try:
            self.logger.info(f"Starting evaluation for {self.__class__.__name__}")
            
            # Load primary model components
            components = self._load_model_components(weight_path, dataset, **kwargs)
            prototypes = self._extract_prototypes(weight_path, **kwargs)
            data = self._prepare_data(dataset)

            # Load additional models for consistency metric (if provided)
            additional_decoders = None
            additional_prototypes = None
            
            if weight_path_replica:
                additional_decoders = []
                additional_prototypes = []
                
                for idx, replica_path in enumerate(weight_path_replica):
                    self.logger.info(f"Loading replica {idx+1}/{len(weight_path_replica)} from {replica_path}")

                    replica_components = self._load_model_components(replica_path, dataset, **kwargs)
                    additional_decoders.append(replica_components.get('decoder'))

                    replica_prototypes = self._extract_prototypes(replica_path , **kwargs)
                    additional_prototypes.append(replica_prototypes)

                    self.logger.info(f"Replica {idx+1} loaded: decoder={replica_components.get('decoder') is not None}, "
                               f"prototypes shape={replica_prototypes.shape}")

            results = self._run_evaluation(components, prototypes, data, directory, additional_decoders, additional_prototypes)
            
            if self.config.visualize:
                self._visualize_results(results, prototypes, label_legends, directory)
            
            self.logger.info("Evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    @abc.abstractmethod
    def _load_model_components(
        self, 
        weight_path: str,
        dataset: Any, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load method-specific model components.
        
        Args:
            weight_path: Path to model weights
            directory: Output directory
            dataset: Dataset instance
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing model components
        """
        pass
    
    @abc.abstractmethod
    def _extract_prototypes(self, directory: str, **kwargs) -> np.ndarray:
        """
        Extract method-specific prototypes.
        
        Args:
            directory: Output directory
            **kwargs: Method-specific parameters
            
        Returns:
            Array of prototypes
        """
        pass
    
    def _prepare_data(self, dataset: Any) -> SimpleNamespace:
        """
        Prepare data for evaluation.
        
        Args:
            dataset: Dataset instance
            
        Returns:
            Prepared data namespace
        """
        return SimpleNamespace(x=dataset.x_train, y=dataset.y_train)
    
    def _run_evaluation(
        self,
        components: Dict[str, Any],
        prototypes: np.ndarray,
        data: SimpleNamespace,
        directory: str,
        additional_decoders: Optional[list] = None,
        additional_prototypes: Optional[list] = None

    ) -> Dict[str, Any]:
        """
        Run the common evaluation logic.
        
        Args:
            components: Model components
            prototypes: Extracted prototypes
            data: Prepared data
            directory: Output directory
            label_legends: Label legends
            
        Returns:
            Evaluation results
        """
        return self.evaluator.evaluate(
            encoder=components['encoder'],
            decoder=components.get('decoder'),
            classifier=components['classifier'],
            data=data,
            prototypes=prototypes,
            save_path=directory,
            criteria=self.config.criteria,
            additional_decoders=additional_decoders,
            additional_prototypes=additional_prototypes
        )
    
    def _visualize_results(
        self,
        results: Dict[str, Any],
        prototypes: np.ndarray,
        label_legends: list,
        directory: str
    ) -> None:
        """
        Visualize evaluation results using ClusteringVisualizer.
        
        Args:
            results: Evaluation results
            prototypes: Extracted prototypes
            label_legends: Label legends
            directory: Output directory
        """
        if 'output_clustering' in results:
            try:
                # Verwende die ClusteringVisualizer Klasse
                method = self.config.plot_config['dimensionality_reduction_method']
                self.visualizer.visualize_clustering(
                    output_clustering=results['output_clustering'],
                    all_labels_metrics_centroids=results.get('all_labels_metrics_centroids'),
                    prototypes=prototypes,
                    mapping=results.get('prototype_mapping'),
                    label_legend=label_legends,
                    method=method,
                    save_path=directory,
                )
                self.logger.info(f"Visualization saved to {directory}")
            except Exception as e:
                self.logger.error(f"Visualization failed: {e}")
        else:
            self.logger.warning("No clustering output available for visualization")


# ==================== METHOD-SPECIFIC EVALUATORS ====================

class MAPEvaluator(BaseEvaluator):
    """Evaluator for MAP method."""
    
    def _load_model_components(
        self, 
        weight_path: str, 
        dataset: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Load MAP model components."""
        try:
            original_dim = dataset.x_train.shape[1:]
            latent_dim = kwargs.get('latent_dim', 50)
            num_classes = kwargs.get('num_classes', 2)
            classifier_weights = kwargs.get('classifier_weights')
            
            if not classifier_weights:
                raise ValueError("classifier_weights parameter is required for MAP")
            
            # Load autoencoder
            autoencoder = AutoEncoderMAP(original_dim=original_dim, latent_dim=latent_dim)
            sample_input = tf.zeros((1,) + original_dim)
            autoencoder(sample_input)
            autoencoder.load_weights(weight_path)
            
            # Load classifier
            classifier = FCNBlock(num_classes=num_classes)
            classifier(sample_input)
            classifier.load_weights(classifier_weights)
            
            return {
                'encoder': EncoderWrapper(autoencoder.encoder),
                'decoder': DecoderWrapper(autoencoder.decoder),
                'classifier': ClassifierWrapper(classifier)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load MAP components: {e}")
            raise
    
    def _extract_prototypes(self, directory: str, **kwargs) -> np.ndarray:
        """Extract MAP prototypes using PrototypeExtractor."""
        path = os.path.dirname(os.path.dirname(directory))
        prototype_path = os.path.join(path, "completeness_importance_concept_map.csv")
        try:
            return self.prototype_extractor.extract_from_csv_map(prototype_path)
        except DataProcessingError as e:
            self.logger.error(f"Failed to extract MAP prototypes: {e}")
            raise

    @classmethod
    def prepare_evaluation_params(
        cls, 
        config: DictConfig, 
        dataset: Any
    ) -> Dict[str, Any]:
        """Prepare MAP-specific parameters."""
        params = {
            'directory': config.output_directory,
            'label_legends': config.dataset.labels,
            'dataset': dataset,
            'weight_path': f"{config.input_directory}/explainer/ae_weights.weights.h5",
            'latent_dim': getattr(config, 'latent_dim', 50),
            'num_classes': config.dataset.n_classes,
            'classifier_weights': f"{config.input_directory}/best_model.weights.h5",
        }
        
        if config.n_replicas > 0:
            params['weight_path_replica'] = [
                f"{config.input_directory}/replica/replica_{i}/explainer/ae_weights.weights.h5" 
                for i in range(1, config.n_replicas + 1)
            ]
        
        return params


class MSPEvaluator(BaseEvaluator):
    """Evaluator for MSP method."""
    
    def __init__(self, config):
        super().__init__(config)
        self._msp_prototypes = {}
    
    def _load_model_components(
        self, 
        weight_path: str,
        dataset: Any, 
        **kwargs
    ) -> Dict[str, Any]:
        """Load MSP model components."""
        try:
            original_dim = dataset.x_train.shape[1:]
            n_concepts = kwargs.get('n_concepts', 5)
            
            # Load autoencoder
            autoencoder = AutoEncoderMSP(
                original_dim=original_dim, 
                latent_dim=n_concepts, 
                n_concepts=n_concepts
            )
            sample_input = tf.zeros((1,) + original_dim)
            autoencoder(sample_input)
            autoencoder.load_weights(weight_path)
            
            # Cache prototypes for later extraction
            prototypes = autoencoder.predictor.prototypes.numpy()
            path = os.path.dirname(weight_path)
            self._msp_prototypes[path] = prototypes
            
            classifier = CompositeClassifier(
                encoder=autoencoder.encoder, 
                predictor=autoencoder.predictor
            )
            
            return {
                'encoder': EncoderWrapper(autoencoder.encoder),
                'decoder': DecoderWrapper(autoencoder.decoder),
                'classifier': ClassifierWrapper(classifier)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load MSP components: {e}")
            raise
    
    def _extract_prototypes(self, directory: str, **kwargs) -> np.ndarray:
        """Extract MSP prototypes."""
        path = os.path.dirname(directory)
        if path not in self._msp_prototypes:
            raise RuntimeError("Prototypes not loaded. Call _load_model_components first.")
        return self._msp_prototypes[path]
    
    @classmethod
    def prepare_evaluation_params(
        cls, 
        config: DictConfig, 
        dataset: Any
    ) -> Dict[str, Any]:
        """Prepare MSP-specific parameters."""
        params = {
            'directory': config.output_directory,
            'label_legends': config.dataset.labels,
            'dataset': dataset,
            'weight_path': f"{config.input_directory}/msp/pexp_weights.weights.h5",
            'n_concepts': getattr(config, 'n_concepts', 5),
        }
        
        if config.n_replicas > 0:
            params['weight_path_replica'] = [
                f"{config.input_directory}/replica/replica_{i}/msp/pexp_weights.weights.h5" 
                for i in range(1, config.n_replicas + 1)
            ]
        
        return params


class EBEEvaluator(BaseEvaluator):
    """Evaluator for EBE method."""
    
    def _load_model_components(
        self, 
        weight_path: str, 
        dataset: Any, 
        **kwargs
    ) -> Dict[str, Any]:
        """Load EBE model components."""
        try:
            # Build classifier
            classifier = build_fcn_block_functional(
                input_shape=dataset.x_train.shape[1:], 
                num_classes=2
            )
            classifier.load_weights(weight_path)
            
            # Extract encoder
            encoder, _ = separate_model(classifier)
            
            return {
                'encoder': EncoderWrapper(encoder),
                'classifier': ClassifierWrapper(classifier)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load EBE components: {e}")
            raise
    
    def _extract_prototypes(self, directory: str, **kwargs) -> np.ndarray:
        """Extract EBE prototypes using PickleHandler."""
        pickle_file = kwargs.get('pickle_file_prototypes')
        if not pickle_file:
            raise ValueError("pickle_file_prototypes parameter is required for EBE")
        
        try:
            prototypes = self.pickle_handler.read(pickle_file)
            if hasattr(prototypes, 'numpy'):
                prototypes = prototypes.numpy()
            return prototypes
        except FileOperationError as e:
            self.logger.error(f"Failed to load prototypes from {pickle_file}: {e}")
            raise

    @classmethod
    def prepare_evaluation_params(
        cls, 
        config: DictConfig, 
        dataset: Any
    ) -> Dict[str, Any]:
        """Prepare EBE-specific parameters."""
        return {
            'directory': config.output_directory,
            'label_legends': config.dataset.labels,
            'dataset': dataset,
            'weight_path': f"{config.input_directory}/best_model.weights.h5",
            'pickle_file_prototypes': f"{config.input_directory}/prototypes.pkl",
        }

# ==================== FACTORY ====================

class EvaluatorFactory:
    """Factory for creating evaluators using Factory pattern."""
    
    _evaluators: Dict[str, Type[BaseEvaluator]] = {
        'map': MAPEvaluator,
        'msp': MSPEvaluator,
        'ebe': EBEEvaluator,
    }
    
    @classmethod
    def create_evaluator(cls, method: str, config: EvaluationConfig) -> BaseEvaluator:
        """
        Create evaluator instance for specified method.
        
        Args:
            method: Method name
            config: Evaluation configuration
            
        Returns:
            Appropriate evaluator instance
            
        Raises:
            ValueError: If method is not supported
        """
        method = method.lower()
        if method not in cls._evaluators:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(cls._evaluators.keys())}"
            )
        
        return cls._evaluators[method](config)
    
    @classmethod
    def get_supported_methods(cls) -> list:
        """Get list of supported methods."""
        return list(cls._evaluators.keys())


# ==================== MAIN ORCHESTRATOR ====================

class MethodEvaluator:
    """Main class for orchestrating method evaluation."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the method evaluator.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.eval_config = EvaluationConfig(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['method', 'dataset', 'input_directory', 'output_directory']
        
        for field in required_fields:
            if not hasattr(self.config, field):
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate method
        method_name = self.config.method.name.lower()
        supported_methods = EvaluatorFactory.get_supported_methods()
        
        if method_name not in supported_methods:
            raise ValueError(
                f"Method '{method_name}' not recognized. "
                f"Valid methods: {supported_methods}"
            )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation based on the configured method.
        
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            Exception: If evaluation fails
        """
        method = self.config.method.name.lower()
        
        # Create the appropriate evaluator using the factory
        evaluator = EvaluatorFactory.create_evaluator(method, self.eval_config)
        
        # Create dataset
        dataset = DatasetFactory.create_dataset(self.config.dataset.name)
        
        # Method-specific evaluation parameters
        evaluation_params = evaluator.prepare_evaluation_params(self.config, dataset)
        
        self.logger.info(f"Starting evaluation for method: {method}")
        return evaluator.evaluate(**evaluation_params)
    
    def _get_evaluation_params(self, dataset: Any) -> Dict[str, Any]:
        """
        Get method-specific evaluation parameters.
        
        Args:
            dataset: Dataset instance
            
        Returns:
            Dictionary of evaluation parameters
        """
        method = self.config.method.name.lower()
        
        base_params = {
            'directory': self.config.output_directory,
            'label_legends': self.config.dataset.labels,
            'dataset': dataset,
        }
        
        method_params = {
            'ebe': self._get_ebe_params,
            'msp': self._get_msp_params,
            'map': self._get_map_params,
        }
        
        if method in method_params:
            specific_params = method_params[method]()
            return {**base_params, **specific_params}
        
        raise ValueError(f"Unsupported method: {method}")