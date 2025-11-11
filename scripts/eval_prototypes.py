"""
Main evaluation script for prototype-based explainable AI methods.

This script coordinates the evaluation of different prototype-based methods
(EBE, MSP, MAP, TapNet) using Hydra configuration management.
"""

import logging
from typing import Any, Dict

import hydra
from omegaconf import DictConfig

from src.evaluation.method_evaluators import MethodEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="eval_hyperparameter", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function with improved error handling and logging.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        evaluator = MethodEvaluator(cfg)
        
        # Validate configuration
        evaluator.validate_config()
        logger.info("Configuration validated successfully")
        
        # Log key parameters
        _log_configuration(cfg, evaluator)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Log results
        _log_results(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


def _log_configuration(cfg: DictConfig, evaluator: MethodEvaluator) -> None:
    """
    Log key configuration parameters.
    
    Args:
        cfg: Configuration object
        evaluator: Method evaluator instance
    """
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Distance metric: {evaluator.eval_config.distance_metric}")
    logger.info(f"Visualization: {evaluator.eval_config.visualize}")
    logger.info(f"Input directory: {cfg.input_directory}")
    logger.info(f"Output directory: {cfg.output_directory}")


def _log_results(results: Dict[str, Any]) -> None:
    """
    Log evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
    """
    if not results:
        logger.warning("No results returned from evaluation")
        return
    
    final_score = results.get('final_score', 'N/A')
    logger.info(f"Evaluation completed successfully!")
    logger.info(f"Final score: {final_score}")
    
    excluded_keys = {'final_score', 'output_clustering', 'prototype_mapping'}
    
    # Log individual metrics
    metric_count = 0
    for metric, value in results.items():
        if metric in excluded_keys:
            continue
        try:
            numeric_value = float(value)
            logger.info(f"{metric}: {numeric_value:.4f}")
            metric_count += 1
        except (TypeError, ValueError):
            pass
    
    logger.info(f"Total metrics evaluated: {metric_count}")


if __name__ == "__main__":
    main()