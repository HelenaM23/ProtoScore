"""
Training modules for prototype-based explainable AI methods.

This module implements the Template Method Pattern and Factory Pattern
for a clean, extensible training architecture.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any,  Optional

import random
import numpy as np
import tensorflow as tf

import numpy as np
from omegaconf import DictConfig

from ..core.model import FCNBlock
from ..methods.ebe.ebe_explainer import get_ebe_explanations
from ..methods.ebe.train_classifier import train_classifier_ebe
from ..methods.map.map import get_map_explanations, train_classifier
from ..methods.msp.msp import get_msp_explanations

logger = logging.getLogger(__name__)


# =====================================================
# EXCEPTIONS
# =====================================================

class TrainingError(Exception):
    """Custom exception for training errors."""
    pass


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


# =====================================================
# CORE TRAINING ARCHITECTURE
# =====================================================

class BaseModelRunner(ABC):
    """
    Abstract base class for model runners implementing Template Method pattern.
    
    This class defines the common training workflow while allowing
    subclasses to implement method-specific training details.
    """
    
    def run(
        self,
        epochs: int,
        directory: str,
        data: Any,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Template method for model training execution.
        
        Args:
            epochs: Number of training epochs
            directory: Output directory
            data: Training data (BaseDataset instance)
            n_concepts: Number of concepts
            n_classes: Number of classes
            hp_json_path: Path to hyperparameters
            **kwargs: Additional parameters
        """
        try:
            if random_state is not None:
                self._set_random_seeds(random_state)

            logger.info(f"Starting {self.__class__.__name__} training...")
            
            # 1. Prepare data using dataset's built-in preprocessing
            train, val, test = data.convert_to_namespace()
            
            # 2. Ensure one-hot encoding using dataset's preprocessing
            num_classes = self._determine_num_classes(train, n_classes)
            train = data.ensure_one_hot_labels(train, num_classes)
            val = data.ensure_one_hot_labels(val, num_classes)
            test = data.ensure_one_hot_labels(test, num_classes)
            
            # 3. Validate prepared data
            self._validate_prepared_data(train, val, test)
            
            # 4. Execute specific training
            self._execute_training(
                train=train,
                val=val,
                test=test,
                epochs=epochs,
                directory=Path(directory),
                n_concepts=n_concepts,
                n_classes=num_classes,
                hp_json_path=hp_json_path,
                **kwargs
            )
            
            logger.info(f"{self.__class__.__name__} training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} training: {e}")
            raise TrainingError(f"Training failed: {e}") from e
        
    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def _determine_num_classes(self, train: SimpleNamespace, n_classes: int) -> int:
        """Determine final number of classes."""
        if hasattr(train, "y") and train.y is not None:
            if len(train.y.shape) == 1:
                detected_classes = len(np.unique(train.y))
            elif len(train.y.shape) == 2:
                detected_classes = train.y.shape[1]
            else:
                detected_classes = n_classes
            
            # Use maximum to ensure compatibility
            return max(detected_classes, n_classes)
        
        return n_classes
    
    def _validate_prepared_data(
        self, 
        train: SimpleNamespace, 
        val: SimpleNamespace, 
        test: SimpleNamespace
    ) -> None:
        """Validate prepared data before training."""
        required_attrs = ['X', 'y']
        
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            for attr in required_attrs:
                if not hasattr(split_data, attr) or getattr(split_data, attr) is None:
                    raise TrainingError(f"Missing {attr} in {split_name} data")
            
            if len(split_data.X) != len(split_data.y):
                raise TrainingError(f"Shape mismatch in {split_name} data")
    
    @abstractmethod
    def _execute_training(
        self,
        train: SimpleNamespace,
        val: SimpleNamespace,
        test: SimpleNamespace,
        epochs: int,
        directory: Path,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        **kwargs
    ) -> None:
        """Execute method-specific training. Must be implemented by subclasses."""
        pass


# =====================================================
# METHOD-SPECIFIC RUNNERS
# =====================================================

class EBERunner(BaseModelRunner):
    """Runner for Evidence-Based Explanations (EBE) method."""
    
    def _execute_training(
        self,
        train: SimpleNamespace,
        val: SimpleNamespace,
        test: SimpleNamespace,
        epochs: int,
        directory: Path,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        **kwargs
    ) -> None:
        """Execute EBE-specific training workflow."""
        logger.info("Training EBE classifier...")
        
        try:
            # Train classifier
            model = train_classifier_ebe(
                model=FCNBlock(n_classes),
                train=train,
                valid=val,
                test=test,
                num_classes=n_classes,
                output_dir=directory,
                filepath=Path(""),
                epochs=epochs,
                hp_filepath=hp_json_path,
            )
            
            logger.info("Generating EBE explanations...")
            
            # Generate explanations
            get_ebe_explanations(
                model=model,
                train=train,
                test=test,
                output_dir=directory,
                n_concepts=n_concepts,
            )
            
        except Exception as e:
            raise TrainingError(f"EBE training failed: {e}") from e


class MSPRunner(BaseModelRunner):
    """Runner for Most Similar Prototype (MSP) method."""
    
    def _execute_training(
        self,
        train: SimpleNamespace,
        val: SimpleNamespace,
        test: SimpleNamespace,
        epochs: int,
        directory: Path,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        **kwargs
    ) -> None:
        """Execute MSP-specific training workflow."""
        logger.info("Generating MSP explanations...")
        
        try:
            get_msp_explanations(
                train=train,
                test=test,
                output_dir=directory,
                n_concepts=n_concepts,
                epochs=epochs,
                n_classes=n_classes,
                filepath=Path(""),
                hp_filepath=hp_json_path,
            )
        except Exception as e:
            raise TrainingError(f"MSP training failed: {e}") from e


class MAPRunner(BaseModelRunner):
    """Runner for Maximally Activated Patches (MAP) method."""
    
    def _execute_training(
        self,
        train: SimpleNamespace,
        val: SimpleNamespace,
        test: SimpleNamespace,
        epochs: int,
        directory: Path,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        **kwargs
    ) -> None:
        """Execute MAP-specific training workflow."""
        logger.info("Training MAP classifier...")
        
        try:
            # Train classifier
            model = train_classifier(
                model=FCNBlock(num_classes=n_classes),
                train=train,
                valid=val,
                test=test,
                num_classes=n_classes,
                output_dir=directory,
                filepath=Path(""),
                hp_path=hp_json_path,
            )
            
            logger.info("Generating MAP explanations...")
            
            # Generate explanations
            get_map_explanations(
                model=model,
                train=train,
                test=test,
                output_dir=directory,
                explainer_name="explainer",
                n_concepts=n_concepts,
                epochs=epochs,
            )
            
        except Exception as e:
            raise TrainingError(f"MAP training failed: {e}") from e


# =====================================================
# TRAINER FACADE LAYER
# =====================================================

class BaseTrainer(ABC):
    """Abstract base class for trainers implementing Facade pattern."""
    
    def __init__(self, runner: BaseModelRunner):
        """Initialize trainer with specific runner."""
        if not isinstance(runner, BaseModelRunner):
            raise ValueError("Runner must be an instance of BaseModelRunner")
        self._runner = runner
    
    def run(
        self,
        epochs: int,
        directory: str,
        data: Any,
        n_concepts: int,
        n_classes: int,
        hp_json_path: str,
        **kwargs
    ) -> None:
        """Execute training using the runner."""
        self._runner.run(
            epochs=epochs,
            directory=directory,
            data=data,
            n_concepts=n_concepts,
            n_classes=n_classes,
            hp_json_path=hp_json_path,
            **kwargs
        )


class EBETrainer(BaseTrainer):
    """Trainer for EBE method."""
    
    def __init__(self):
        super().__init__(EBERunner())


class MSPTrainer(BaseTrainer):
    """Trainer for MSP method."""
    
    def __init__(self):
        super().__init__(MSPRunner())


class MAPTrainer(BaseTrainer):
    """Trainer for MAP method."""
    
    def __init__(self):
        super().__init__(MAPRunner())


# =====================================================
# FACTORY AND VALIDATION
# =====================================================

class TrainerFactory:
    """Factory for creating trainer instances using Factory pattern."""
    
    _trainers = {
        'ebe': EBETrainer,
        'msp': MSPTrainer,
        'map': MAPTrainer
    }
    
    @classmethod
    def create_trainer(cls, method_name: str) -> BaseTrainer:
        """Create trainer based on method name."""
        trainer_class = cls._trainers.get(method_name.lower())
        if not trainer_class:
            available = list(cls._trainers.keys())
            raise ValueError(
                f"Unknown method: {method_name}. "
                f"Available methods: {available}"
            )
        
        return trainer_class()
    
    @classmethod
    def register_trainer(cls, name: str, trainer_class: type) -> None:
        """Register new trainer class."""
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError("Trainer must inherit from BaseTrainer")
        cls._trainers[name.lower()] = trainer_class
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available training methods."""
        return list(cls._trainers.keys())


class ConfigValidator:
    """Validator for training configurations."""
    
    REQUIRED_FIELDS = {
        'method': ['name', 'epochs'],
        'dataset': ['name', 'n_classes'],
        'general': ['directory', 'n_concepts', 'train_hyperparameters_path']
    }
    
    VALID_METHODS = ['ebe', 'msp', 'map']
    VALID_DATASETS = [
        'ECG200', 'Sawsine', 'FordA', 'FordB', 
        'StarLightCurve', 'Wafer', 'Blink'
    ]
    
    @classmethod
    def validate_config(cls, config: DictConfig) -> None:
        """Validate configuration."""
        cls._validate_required_fields(config)
        cls._validate_method(config.method.name)
        cls._validate_dataset(config.dataset.name)
        cls._validate_numeric_fields(config)
    
    @classmethod
    def _validate_required_fields(cls, config: DictConfig) -> None:
        """Validate required fields are present."""
        missing_fields = []
        
        # Check method fields
        if hasattr(config, 'method'):
            for field in cls.REQUIRED_FIELDS['method']:
                if not hasattr(config.method, field):
                    missing_fields.append(f'method.{field}')
        else:
            missing_fields.extend([f'method.{f}' for f in cls.REQUIRED_FIELDS['method']])
        
        # Check dataset fields
        if hasattr(config, 'dataset'):
            for field in cls.REQUIRED_FIELDS['dataset']:
                if not hasattr(config.dataset, field):
                    missing_fields.append(f'dataset.{field}')
        else:
            missing_fields.extend([f'dataset.{f}' for f in cls.REQUIRED_FIELDS['dataset']])
        
        # Check general fields
        for field in cls.REQUIRED_FIELDS['general']:
            if not hasattr(config, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigValidationError(
                f"Missing configuration fields: {missing_fields}"
            )
    
    @classmethod
    def _validate_method(cls, method_name: str) -> None:
        """Validate method name."""
        if method_name.lower() not in cls.VALID_METHODS:
            raise ConfigValidationError(
                f"Invalid method: {method_name}. "
                f"Valid methods: {cls.VALID_METHODS}"
            )
    
    @classmethod
    def _validate_dataset(cls, dataset_name: str) -> None:
        """Validate dataset name."""
        if dataset_name not in cls.VALID_DATASETS:
            raise ConfigValidationError(
                f"Invalid dataset: {dataset_name}. "
                f"Valid datasets: {cls.VALID_DATASETS}"
            )
    
    @classmethod
    def _validate_numeric_fields(cls, config: DictConfig) -> None:
        """Validate numeric fields."""
        if config.method.epochs <= 0:
            raise ConfigValidationError("Epochs must be greater than 0")
        
        if config.dataset.n_classes <= 0:
            raise ConfigValidationError("n_classes must be greater than 0")
        
        if config.n_concepts <= 0:
            raise ConfigValidationError("n_concepts must be greater than 0")
        
        if hasattr(config, 'test_size'):
            if not 0 < config.test_size < 1:
                raise ConfigValidationError("test_size must be between 0 and 1")
        
        if hasattr(config, 'validation_size') and config.validation_size is not None:
            if not 0 < config.validation_size < 1:
                raise ConfigValidationError("validation_size must be between 0 and 1")
            
            # Ensure test_size + validation_size < 1
            if hasattr(config, 'test_size'):
                if config.test_size + config.validation_size >= 1:
                    raise ConfigValidationError(
                        f"test_size ({config.test_size}) + validation_size "
                        f"({config.validation_size}) must be less than 1"
                    )
        
        if hasattr(config, 'ensure_min_samples'):
            if config.ensure_min_samples <= 0:
                raise ConfigValidationError("ensure_min_samples must be greater than 0")
        
# =====================================================