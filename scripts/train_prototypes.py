"""
Main training script for prototype-based explainable AI methods.

This script coordinates the training of different prototype-based methods
(EBE, MSP, MAP) using Hydra configuration management.
"""

import logging
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig

from src.core.data import DatasetFactory
from src.training.train_models import TrainerFactory, ConfigValidator, ConfigValidationError

# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the training process for prototype-based explainable AI models.
    
    This class coordinates dataset creation, trainer setup, and training execution
    while providing comprehensive error handling and logging.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the training orchestrator.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.dataset_factory = DatasetFactory()
        self.trainer_factory = TrainerFactory()
    
    def validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        ConfigValidator.validate_config(self.config)
    
    def run_training(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary containing training results and metadata
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            # Create dataset
            logger.info("Creating dataset...")
            dataset = self._create_dataset()
            
            # Create trainer
            logger.info("Initializing trainer...")
            trainer = self._create_trainer()
            
            # Prepare training parameters
            training_params = self._prepare_training_parameters(dataset)
            
            # Execute primary model training
            logger.info("Starting primary model training...")
            trainer.run(**training_params)
            
            # Train replica models if configured
            n_replicas = getattr(self.config, 'n_replicas', 0)
            replica_results = []
            
            if n_replicas > 0:
                logger.info(f"Training {n_replicas} replica model(s)...")
                replica_results = self._train_replicas(
                    trainer=trainer,
                    n_replicas=n_replicas
                )
            
            # Return results
            results = {
                'status': 'completed',
                'method': self.config.method.name,
                'dataset': self.config.dataset.name,
                'epochs': self.config.method.epochs,
                'output_directory': self.config.directory,
                'primary_random_state': self.config.random_state,
                'replicas_trained': len(replica_results),
                'replica_info': replica_results
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Training execution failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def _train_replicas(
        self,
        trainer,
        n_replicas: int
    ) -> list:
        """
        Train replica models with the same hyperparameters as the primary model 
        but different random states.
        
        Args:
            trainer: Trainer instance to use
            dataset: Dataset instance
            n_replicas: Number of replica models to train
            
        Returns:
            List of replica output directories
        """
        from pathlib import Path
        
        replica_results = []
        base_dir = Path(self.config.directory)
        replica_base_dir = base_dir / "replica"
        base_random_state = getattr(self.config, 'random_state', 42)
        
        # Create replica base directory
        replica_base_dir.mkdir(parents=True, exist_ok=True)
        
        for replica_idx in range(1, n_replicas + 1):
            try:
                # Generate different random state for each replica
                replica_random_state = base_random_state + (1000 * replica_idx)
                
                logger.info(f"Training replica model {replica_idx}/{n_replicas} "
                        f"(random_state={replica_random_state})...")
                
                # Create replica-specific directory
                replica_dir = replica_base_dir / f"replica_{replica_idx}"
                replica_dir.mkdir(parents=True, exist_ok=True)
                
                # Create new dataset with different random state
                dataset_params = self._extract_dataset_parameters()
                dataset_params['random_state'] = replica_random_state
                
                replica_dataset = self.dataset_factory.create_dataset(
                    name=self.config.dataset.name,
                    **dataset_params
                )
                
                # Prepare training parameters for replica
                replica_params = self._prepare_training_parameters(
                    replica_dataset,
                    random_state=replica_random_state
                )
                replica_params['directory'] = str(replica_dir)
                
                # Train replica model
                trainer.run(**replica_params)
                
                replica_info = {
                    'directory': str(replica_dir),
                    'random_state': replica_random_state,
                    'replica_id': replica_idx
                }
                replica_results.append(replica_info)
                
                logger.info(f"Replica model {replica_idx} completed. "
                        f"Saved to: {replica_dir}")
                
            except Exception as e:
                logger.error(f"Failed to train replica model {replica_idx}: {str(e)}")
                logger.warning(f"Continuing with remaining replicas...")
                continue
        
        logger.info(f"Completed training {len(replica_results)}/{n_replicas} replica models")
        return replica_results
    
    def _create_dataset(self):
        """Create dataset instance based on configuration."""
        dataset_params = self._extract_dataset_parameters()
        return self.dataset_factory.create_dataset(
            name=self.config.dataset.name,
            **dataset_params
        )
    
    def _create_trainer(self):
        """Create trainer instance based on configuration."""
        return self.trainer_factory.create_trainer(self.config.method.name)
    
    def _extract_dataset_parameters(self) -> Dict[str, Any]:
        """Extract dataset-specific parameters from configuration."""
        params = {}
        # split parameters
        if hasattr(self.config, 'test_size'):
            params['test_size'] = self.config.test_size
        
        if hasattr(self.config, 'validation_size'):
            params['validation_size'] = self.config.validation_size
        
        if hasattr(self.config, 'stratified'):
            params['stratified'] = self.config.stratified
        
        if hasattr(self.config, 'ensure_min_samples'):
            params['ensure_min_samples'] = self.config.ensure_min_samples
        
        if hasattr(self.config, 'random_state'):
            params['random_state'] = self.config.random_state

        # Outlier noise parameters
        if hasattr(self.config, 'outlier_fraction'):
            params['outlier_fraction'] = self.config.outlier_fraction
        
        if hasattr(self.config, 'noise_seed'):
            params['noise_seed'] = self.config.noise_seed
        
        if hasattr(self.config, 'apply_outlier_noise'):
            params['apply_outlier_noise'] = self.config.apply_outlier_noise
        
        if hasattr(self.config, 'use_gev'):
            params['use_gev'] = self.config.use_gev
        
        if hasattr(self.config, 'gev_params'):
            params['gev_params'] = tuple(self.config.gev_params)
        
        return params
    
    def _prepare_training_parameters(self, dataset, random_state: Optional[int]=None) -> Dict[str, Any]:
        """Prepare parameters for trainer execution."""
        params = {
            'epochs': self.config.method.epochs,
            'directory': self.config.directory,
            'data': dataset,
            'n_concepts': self.config.n_concepts,
            'n_classes': self.config.dataset.n_classes,
            'hp_json_path': self.config.train_hyperparameters_path,
        }
        if random_state is not None:
            params['random_state'] = random_state
        
        return params


@hydra.main(config_path="../config", config_name="train_hyperparameter", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training function with improved error handling and logging.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        orchestrator = TrainingOrchestrator(cfg)
        
        # Validate configuration
        orchestrator.validate_config()
        logger.info("Configuration validated successfully")
        
        # Run training
        results = orchestrator.run_training()
        
        # Log results
        _log(cfg, results)
        
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        logger.error("Please check your configuration file and try again")
        raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception("Full traceback:")
        raise



def _log(cfg: DictConfig, results: Dict[str, Any]) -> None:
    """
    Log training results.
    
    Args:
        results: Dictionary containing training results
    """
    if not results:
        logger.warning("No results returned from training")
        return
    
    replicas_trained = results.get('replicas_trained', 0)
    replica_info = results.get('replica_info', [])

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Epochs: {cfg.method.epochs}")

    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Classes: {cfg.dataset.n_classes}")
    logger.info(f"Concepts: {cfg.n_concepts}")

    logger.info(f"Test size: {cfg.test_size}")
    if hasattr(cfg, 'validation_size') and cfg.validation_size:
        logger.info(f"Validation size: {cfg.validation_size}")
    else:
        logger.info("Validation size: None (no validation split)")
    logger.info(f"Stratified: {cfg.stratified}")
    if hasattr(cfg, 'ensure_min_samples'):
        logger.info(f"Min samples per class: {cfg.ensure_min_samples}")
    logger.info(f"Hyperparameters path: {cfg.train_hyperparameters_path}")

    # Log optional parameters if present
    if hasattr(cfg, 'apply_outlier_noise') and cfg.apply_outlier_noise:
        logger.info(f"Outlier noise: {cfg.outlier_fraction} fraction")
        logger.info(f"Noise seed: {cfg.noise_seed}")
        logger.info(f"Use GEV: {cfg.use_gev}")
        

    logger.info(f"Primary model:")
    logger.info(f"  - Random state: {cfg.random_state}")
    logger.info(f"  - Saved to: {cfg.directory}")
    
    
    if replicas_trained > 0:
        logger.info(f"\nReplica models trained: {replicas_trained}")
        logger.info("Replica details:")
        for info in replica_info:
            logger.info(f"  - Replica {info['replica_id']}:")
            logger.info(f"    Random state: {info['random_state']}")
            logger.info(f"    Directory: {info['directory']}")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()