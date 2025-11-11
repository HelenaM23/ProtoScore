from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, silhouette_score)

from ..utils.distance import DistanceCalculatorFactory


class BaseEvaluationMetric(ABC):
    """Base class for evaluation metrics with standardized interface."""
    
    def __init__(self, distance_metric: str = "euclidean", decoder: Optional = None) -> None:
        self.distance_metric = distance_metric
        self.distance_calculator = DistanceCalculatorFactory.create(distance_metric, decoder)
    
    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the evaluation metric. Returns value between 0 and 1."""
        pass
    
    def _validate_inputs(self, **kwargs) -> None:
        """Validate common input parameters. Override in subclasses as needed."""
        pass



class ConsistencyMetric(BaseEvaluationMetric):
    """Measures consistency across multiple model runs."""
    
    def compute(
        self,
        primary_encoder,
        primary_prototypes: np.ndarray,
        decoders: List,
        list_of_prototypes: List[np.ndarray],
        primary_decoder: Optional = None
    ) -> Tuple[float, float]:
        """
        Compute consistency metric and its error.
        
        Returns:
            Tuple of (consistency_metric, error)
        """
        self._validate_consistency_inputs(primary_prototypes, decoders, list_of_prototypes)
        
        model_distances = self._compute_model_distances(
            primary_encoder, primary_prototypes, decoders, 
            list_of_prototypes, primary_decoder
        )

        return np.exp(-np.mean(model_distances))
    
    def _validate_consistency_inputs(
        self, 
        primary_prototypes: np.ndarray, 
        decoders: List, 
        list_of_prototypes: List[np.ndarray]
    ) -> None:
        """Validate inputs for consistency computation."""
        if len(decoders) != len(list_of_prototypes):
            raise ValueError("Number of decoders must match number of prototype lists")
        if len(primary_prototypes) == 0:
            raise ValueError("Primary prototypes cannot be empty")
    
    def _compute_model_distances(
        self,
        primary_encoder,
        primary_prototypes: np.ndarray,
        decoders: List,
        list_of_prototypes: List[np.ndarray],
        primary_decoder: Optional
    ) -> List[float]:
        """Compute distances between models."""
        model_distances = []
        
        for decoder, prototypes in zip(decoders, list_of_prototypes):
            decoded_prototypes = decoder(prototypes)
            reencoded_prototypes = primary_encoder(decoded_prototypes)
            
            if self.distance_metric == "DTW" and primary_decoder:
                primary_decoded = primary_decoder(primary_prototypes)
                distances = self.distance_calculator.compute_distance(
                    prototypes, primary_prototypes
                )
            else:
                distances = self.distance_calculator.compute_distance(
                    reencoded_prototypes, primary_prototypes
                )
            
            model_distances.append(np.mean(np.min(distances, axis=1)))
        
        return model_distances


class ContinuityMetric(BaseEvaluationMetric):
    """Measures model stability under input noise."""
    
    DEFAULT_NOISE_SCALE = 0.05
    
    def compute(
        self, 
        encoder, 
        prototypes: np.ndarray, 
        data: np.ndarray, 
        noise_scale: float = DEFAULT_NOISE_SCALE
    ) -> float:
        """Compute continuity metric."""
        self._validate_continuity_inputs(data, prototypes, noise_scale)
        
        noised_data = self._add_noise(data, noise_scale)
        
        encoded_data = encoder.call(data)
        encoded_noised_data = encoder.call(noised_data)
        
        closest_prototypes = self._get_closest_prototypes(encoded_data, prototypes)
        closest_noised_prototypes = self._get_closest_prototypes(encoded_noised_data, prototypes)
        
        differences = self.distance_calculator.compute_distance(closest_prototypes, closest_noised_prototypes)
        return np.exp(-np.mean(differences))
    
    def _validate_continuity_inputs(
        self, 
        data: np.ndarray, 
        prototypes: np.ndarray, 
        noise_scale: float
    ) -> None:
        """Validate inputs for continuity computation."""
        if noise_scale < 0:
            raise ValueError("Noise scale must be non-negative")
        if len(data) == 0 or len(prototypes) == 0:
            raise ValueError("Data and prototypes cannot be empty")
    
    def _get_closest_prototypes(self, encoded_data: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Get closest prototypes for each encoded data point."""
        distances = self.distance_calculator.compute_distance(encoded_data, prototypes)
        closest_indices = np.argmin(distances, axis=1)
        return prototypes[closest_indices]
    
    def _add_noise(self, data: np.ndarray, noise_scale: float) -> np.ndarray:
        """Add scaled noise to data."""
        sample_ranges = np.ptp(data, axis=tuple(range(1, data.ndim)), keepdims=True)
        noise_std = np.mean(sample_ranges) * noise_scale
        return data + np.random.normal(0.0, noise_std, data.shape)


class ContrastivityMetric(BaseEvaluationMetric):
    """Measures distances between prototypes."""
    
    def compute(self, prototypes: np.ndarray) -> float:
        """Compute contrastivity metric."""
        if len(prototypes) < 2:
            return 1.0  # Perfect contrastivity for single prototype
        
        distances = self.distance_calculator.compute_distance(prototypes, prototypes)
        mask = ~np.eye(len(prototypes), dtype=bool)
        return np.exp(-distances[mask].mean())


class CorrectnessMetric(BaseEvaluationMetric):
    """Measures classification accuracy using prototypes."""
    
    def compute(
        self, 
        encoder, 
        decoder, 
        prototypes: np.ndarray, 
        data_samples: np.ndarray, 
        classifier
    ) -> Dict[str, float]:
        """Compute correctness metrics."""
        self._validate_correctness_inputs(prototypes, data_samples)
        
        sample_labels = self._get_original_predictions(classifier, data_samples)
        predicted_labels = self._get_prototype_predictions(
            encoder, decoder, classifier, prototypes, data_samples
        )
        
        return self._compute_classification_metrics(sample_labels, predicted_labels)
    
    def _validate_correctness_inputs(self, prototypes: np.ndarray, data_samples: np.ndarray) -> None:
        """Validate inputs for correctness computation."""
        if len(prototypes) == 0 or len(data_samples) == 0:
            raise ValueError("Prototypes and data samples cannot be empty")
    
    def _get_original_predictions(self, classifier, data_samples: np.ndarray) -> np.ndarray:
        """Get original classifier predictions."""
        original_predictions = classifier.call(data_samples)
        return np.argmax(original_predictions, axis=1)
    
    def _get_prototype_predictions(
        self, 
        encoder, 
        decoder, 
        classifier, 
        prototypes: np.ndarray, 
        data_samples: np.ndarray
    ) -> np.ndarray:
        """Get predictions using closest prototypes."""
        encoded_samples = encoder.call(data_samples)
        distances = self.distance_calculator.compute_distance(encoded_samples, prototypes)
        closest_indices = np.argmin(distances, axis=1)
        
        decoded_prototypes = decoder.call(prototypes[closest_indices])
        prototype_predictions = classifier.call(decoded_prototypes)
        return np.argmax(prototype_predictions, axis=1)
    
    def _compute_classification_metrics(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        return {
            'precision': precision_score(true_labels, predicted_labels, average='macro', zero_division=0),
            'recall': recall_score(true_labels, predicted_labels, average='macro', zero_division=0),
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'f1': f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        }


class CompactnessMetric(BaseEvaluationMetric):
    """Measures prototype set compactness."""
    
    COMPACTNESS_FACTOR = 0.08
    
    def compute(self, prototypes: np.ndarray) -> float:
        """Compute compactness metric based on prototype count."""
        return np.exp((-prototypes.shape[0] + 1) * self.COMPACTNESS_FACTOR)


class ConfidenceMetric(BaseEvaluationMetric):
    """Measures confidence based on closest prototype distances."""
    
    def compute(self, prototypes: np.ndarray, embeddings: np.ndarray) -> float:
        """Compute confidence metric."""
        if len(prototypes) == 0 or len(embeddings) == 0:
            return 0.0
        
        distances = self.distance_calculator.compute_distance(embeddings, prototypes)
        min_distances = np.min(distances, axis=1)
        return np.exp(-np.mean(min_distances))


class CohesionOfLatentSpaceMetric(BaseEvaluationMetric):
    """Measures cohesion of latent space using silhouette score."""
    
    def compute(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute cohesion metric."""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        if not self._can_compute_silhouette(n_clusters, len(embeddings)):
            return 0.0
        
        silhouette_score_value = self._compute_silhouette_score(embeddings, cluster_labels)
        return self._normalize_silhouette_score(silhouette_score_value)
    
    def _can_compute_silhouette(self, n_clusters: int, n_samples: int) -> bool:
        """Check if silhouette score can be computed."""
        return n_clusters > 1 and n_samples > n_clusters
    
    def _compute_silhouette_score(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute silhouette score based on distance metric."""
        if self.distance_metric == "euclidean":
            return silhouette_score(embeddings, cluster_labels)
        elif self.distance_metric == "DTW":
            distance_matrix = self.distance_calculator.compute_distance(embeddings, embeddings)
            return silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _normalize_silhouette_score(self, score: float) -> float:
        """Normalize silhouette score from [-1, 1] to [0, 1]."""
        return (score + 1) / 2


class InputCompletenessMetric(BaseEvaluationMetric):
    """Measures input completeness using pre-computed mapping."""
    
    def compute(
        self,
        prototype_mapping: Dict[int, Dict[str, Any]],
        centroid_info: Dict[int, Dict[str, Any]],
        **kwargs
    ) -> float:
        """Compute input completeness score."""
        if not centroid_info:
            return 0.0
        
        matched_centroids = self._identify_matched_centroids(prototype_mapping, centroid_info)
        return sum(matched_centroids) / len(matched_centroids)
    
    def _identify_matched_centroids(
        self,
        prototype_mapping: Dict[int, Dict[str, Any]],
        centroid_info: Dict[int, Dict[str, Any]]
    ) -> List[bool]:
        """Identify which centroids have representative prototypes."""
        matched_centroids = [False] * len(centroid_info)
        
        for proto_info in prototype_mapping.values():
            centroid_idx = proto_info.get("centroid")
            if self._is_valid_centroid_match(centroid_idx, proto_info, centroid_info, matched_centroids):
                matched_centroids[centroid_idx] = True
        
        return matched_centroids
    
    def _is_valid_centroid_match(
        self,
        centroid_idx: Optional[int],
        proto_info: Dict[str, Any],
        centroid_info: Dict[int, Dict[str, Any]],
        matched_centroids: List[bool]
    ) -> bool:
        """Check if prototype represents its centroid well."""
        if centroid_idx is None or centroid_idx >= len(matched_centroids):
            return False
        
        avg_distance = centroid_info[centroid_idx]["average_distance"]
        proto_distance = proto_info["distance"]
        
        return (avg_distance is not None and 
                proto_distance is not None and 
                proto_distance < avg_distance)


class CovariateComplexityMetric(BaseEvaluationMetric):
    """Measures covariate complexity using silhouette scores for prototypes."""
    
    def compute(
        self, 
        prototypes: np.ndarray, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray,
        **kwargs
    ) -> float:
        """Compute covariate complexity score."""
        silhouette_scores = self._compute_prototype_silhouette_scores(
            prototypes, embeddings, cluster_labels
        )
        return self._normalize_complexity_score(silhouette_scores)
    
    def _compute_prototype_silhouette_scores(
        self, 
        prototypes: np.ndarray, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> List[float]:
        """Compute silhouette scores for prototypes."""
        centroids = self._compute_centroids(embeddings, cluster_labels)
        assigned_labels = self._assign_prototypes_to_centroids(prototypes, centroids, cluster_labels)
        
        return [
            self._compute_sample_silhouette_score(proto, label, embeddings, cluster_labels)
            for proto, label in zip(prototypes, assigned_labels)
        ]
    
    def _compute_centroids(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids."""
        unique_labels = np.unique(cluster_labels)
        return np.array([
            embeddings[cluster_labels == label].mean(axis=0) 
            for label in unique_labels
        ])
    
    def _assign_prototypes_to_centroids(
        self, 
        prototypes: np.ndarray, 
        centroids: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Assign prototypes to nearest centroids."""
        unique_labels = np.unique(cluster_labels)
        distances = self.distance_calculator.compute_distance(prototypes, centroids)
        closest_centroid_indices = np.argmin(distances, axis=1)
        return unique_labels[closest_centroid_indices]
    
    def _compute_sample_silhouette_score(
        self, 
        sample: np.ndarray, 
        sample_label: Any, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Compute silhouette score for a single sample."""
        same_cluster_mask = (labels == sample_label)
        
        if np.sum(same_cluster_mask) <= 1:
            return 0.0
        
        distances = self.distance_calculator.compute_distance(
            sample.reshape(1, -1), embeddings
        ).flatten()
        
        intra_distance = self._compute_intra_cluster_distance(distances, same_cluster_mask)
        inter_distance = self._compute_inter_cluster_distance(distances, labels, sample_label)
        
        return self._compute_silhouette_value(intra_distance, inter_distance)
    
    def _compute_intra_cluster_distance(
        self, 
        distances: np.ndarray, 
        same_cluster_mask: np.ndarray
    ) -> float:
        """Compute average intra-cluster distance."""
        same_distances = distances[same_cluster_mask]
        non_zero_distances = same_distances[same_distances > 0]
        return np.mean(non_zero_distances) if len(non_zero_distances) > 0 else 0.0
    
    def _compute_inter_cluster_distance(
        self, 
        distances: np.ndarray, 
        labels: np.ndarray, 
        sample_label: Any
    ) -> float:
        """Compute minimum average inter-cluster distance."""
        different_cluster_mask = (labels != sample_label)
        
        if not np.sum(different_cluster_mask):
            return 0.0
        
        unique_different_labels = np.unique(labels[different_cluster_mask])
        cluster_distances = [
            np.mean(distances[labels == label]) 
            for label in unique_different_labels
        ]
        
        return np.min(cluster_distances)
    
    def _compute_silhouette_value(self, intra_distance: float, inter_distance: float) -> float:
        """Compute silhouette value from intra and inter distances."""
        max_distance = max(intra_distance, inter_distance)
        if max_distance == 0:
            return 0.0
        return (inter_distance - intra_distance) / max_distance
    
    def _normalize_complexity_score(self, silhouette_scores: List[float]) -> float:
        """Normalize complexity score from [-1, 1] to [0, 1]."""
        mean_score = np.mean(silhouette_scores)
        return (mean_score + 1) / 2