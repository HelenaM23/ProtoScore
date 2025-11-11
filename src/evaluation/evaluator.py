import csv
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

import numpy as np

from ..core.wrapper import ClassifierWrapper, DecoderWrapper, EncoderWrapper

from .cluster import (BasicClustering, ClusterMerger, HierarchicalClustering,
                      PrototypeMapper, ClusteringConfig)
from .metrics import (CohesionOfLatentSpaceMetric, CompactnessMetric,
                      ConfidenceMetric, ContinuityMetric, ContrastivityMetric,
                      CorrectnessMetric, CovariateComplexityMetric,
                      InputCompletenessMetric, ConsistencyMetric)
from .score import ScoreCalculator


class PrototypeEvaluator:
    """Main evaluation system for prototype-based models."""
    
    def __init__(self, distance_metric: str = "euclidean", 
                 metric_weights: Optional[Dict[str, float]] = None,
                 clustering_config: Optional[Dict[str, Any]] = None):
        self.distance_metric = distance_metric
        self.metric_weights = metric_weights or self._default_weights()
        
        # Create ClusteringConfig instance from dict
        if clustering_config:
            # Convert dict to object-like structure
            class ConfigObj:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            config_obj = ConfigObj(clustering_config)
            self.clustering_config = ClusteringConfig(config_obj)
        else:
            self.clustering_config = ClusteringConfig()

    def evaluate(
        self,
        encoder: EncoderWrapper,
        decoder: DecoderWrapper,
        classifier: ClassifierWrapper,
        data: SimpleNamespace,
        prototypes: np.ndarray,
        save_path: str,
        criteria: Optional[Dict[str, bool]] = None,
        # Additional parameters for consistency metric
        additional_decoders: Optional[List[DecoderWrapper]] = None,
        additional_prototypes: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        
        if criteria is None:
            criteria = {k: True for k in self.metric_weights.keys()}
        
        results: Dict[str, Any] = {}
        
        embeddings = np.array(encoder.call(data.x))
        n_clusters = len(np.unique(data.y))
        
        # Get cluster range from configuration
        cluster_range = self.clustering_config.cluster_range
        
        # Clustering dependent on distance metric
        if self.distance_metric.upper() == "DTW":
            output_clustering = HierarchicalClustering(
                distance_metric=self.distance_metric, 
                decoder=decoder,
                clustering_config=self.clustering_config
            ).compute_multilevel_clustering(
                embeddings=embeddings,
                n_main_clusters=n_clusters,
                labels=data.y
            )
            # Merge clustering results
            cluster_merger = ClusterMerger()
            merged_embeddings, merged_labels = cluster_merger.merge_embeddings_and_labels(output_clustering)
        else:
            # Use BasicClustering for Euclidean
            output_clustering = BasicClustering(
                distance_metric=self.distance_metric,
                decoder=decoder,
                use_optimal_clusters=True,
                clustering_config=self.clustering_config
            ).compute_multilevel_clustering(
                embeddings=embeddings,
                n_main_clusters=n_clusters,
                labels=data.y,
                cluster_range=cluster_range
            )
            
            cluster_merger = ClusterMerger()
            merged_embeddings, merged_labels = cluster_merger.merge_embeddings_and_labels(output_clustering)
        # Mapping prototypes to clusters if needed
        needs_mapping = (
            criteria.get('input_completeness', False) or 
            criteria.get('covariate_complexity', False)
        )
        prototype_mapping = None
        centroid_info = None
        
        if needs_mapping:
            mapper = PrototypeMapper(
                distance_metric=self.distance_metric,
                decoder=decoder if self.distance_metric.upper() == "DTW" else None
            )
            prototype_mapping, centroid_info = mapper.compute_mapping(
                prototypes, merged_embeddings, merged_labels
            )

        # Metric Computations
        if criteria.get('cohesion_of_latent_space', False):
            metric = CohesionOfLatentSpaceMetric(self.distance_metric, decoder)
            results['cohesion_of_latent_space'] = metric.compute(
                embeddings=merged_embeddings,
                cluster_labels=merged_labels
            )
        
        if criteria.get('correctness', False):
            metric = CorrectnessMetric(self.distance_metric, decoder)
            corr = metric.compute(
                encoder=encoder,
                decoder=decoder,
                prototypes=prototypes,
                data_samples=data.x,
                classifier=classifier
            )
            results['correctness'] = corr['accuracy']
        
        if criteria.get('compactness', False):
            metric = CompactnessMetric(self.distance_metric, decoder)
            results['compactness'] = metric.compute(prototypes=prototypes)
        
        if criteria.get('continuity', False):
            metric = ContinuityMetric(self.distance_metric, decoder)
            results['continuity'] = metric.compute(
                encoder=encoder,
                prototypes=prototypes,
                data=data.x
            )
        
        if criteria.get('contrastivity', False):
            metric = ContrastivityMetric(self.distance_metric, decoder)
            results['contrastivity'] = metric.compute(prototypes=prototypes)
        
        if criteria.get('confidence', False):
            metric = ConfidenceMetric(self.distance_metric, decoder)
            results['confidence'] = metric.compute(
                prototypes=prototypes, 
                embeddings=embeddings
            )
        
        if criteria.get('covariate_complexity', False):
            metric = CovariateComplexityMetric(self.distance_metric, decoder)
            results['covariate_complexity'] = metric.compute(
                prototype_mapping=prototype_mapping,
                centroid_info=centroid_info,
                prototypes=prototypes,
                embeddings=merged_embeddings,
                cluster_labels=merged_labels
            )
            
        if criteria.get('input_completeness', False):
            metric = InputCompletenessMetric(self.distance_metric, decoder)
            results['input_completeness'] = metric.compute(
                prototype_mapping=prototype_mapping,
                centroid_info=centroid_info
            )
        if criteria.get('consistency', False):
            if additional_decoders is None or additional_prototypes is None:
                print("Warning: Consistency metric requires additional_decoders and additional_prototypes. Skipping.")
            elif len(additional_decoders) == 0 or len(additional_prototypes) == 0:
                print("Warning: No additional models provided for consistency metric. Skipping.")
            else:
                metric = ConsistencyMetric(self.distance_metric, decoder)
                results['consistency'] = metric.compute(
                    primary_encoder=encoder,
                    primary_prototypes=prototypes,
                    decoders=additional_decoders,
                    list_of_prototypes=additional_prototypes,
                    primary_decoder=decoder if self.distance_metric.upper() == "DTW" else None
                )

        # Final Score
        results['final_score'] = ScoreCalculator.compute_final_score(results, self.metric_weights)
        results['output_clustering'] = output_clustering
        if prototype_mapping is not None:
            results['prototype_mapping'] = prototype_mapping
        
        # save results
        self._save_results(results, save_path)
        return results
    
    def _save_results(self, results: Dict[str, Any], save_path: str):
        """Save results to CSV file."""
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "results.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in results.items():
                if key not in ['output_clustering', 'prototype_mapping']:
                    writer.writerow([key, value])
        
        print(f"Results saved to: {csv_path}")