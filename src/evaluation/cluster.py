# cluster.py
"""Optimized clustering utilities for prototype evaluation."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..utils.distance import DistanceCalculatorFactory


# Configuration and Constants
class ClusteringConfig:
    """Centralized configuration for clustering operations."""
    
    def __init__(self, cfg: Any = None):
        """
        Initialize configuration from Hydra config.
        
        Args:
            cfg: Hydra configuration object or None for defaults
        """
        if cfg is None:
            cfg = {}
            
        # Core clustering parameters
        self.default_random_state = getattr(cfg, 'default_random_state', 42)
        self.scale_data = getattr(cfg, 'scale_data_for_clustering', True)
        self.reduce_dimensions = getattr(cfg, 'reduce_dimensions_for_clustering', False)
        self.reduction_method = getattr(cfg, 'dimensionality_reduction_method', 'pca')
        self.default_n_init = getattr(cfg, 'default_n_init', 10)
        self.min_clusters = getattr(cfg, 'min_clusters', 2)
        self.max_clusters = getattr(cfg, 'max_clusters', 15)
        self.min_silhouette_samples = getattr(cfg, 'min_silhouette_samples', 10)
        self.default_n_components = getattr(cfg, 'default_n_components', 2)
        
        # Derived parameters
        self.cluster_range = (self.min_clusters, self.max_clusters)
        
        


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ClusteringValidator:
    """Centralized validation for clustering operations."""
    
    @staticmethod
    def validate_data(data: np.ndarray, min_samples: int = 1) -> None:
        """Validate input data."""
        if len(data) == 0:
            raise ValidationError("Input data cannot be empty")
        if len(data) < min_samples:
            raise ValidationError(f"Insufficient data: need at least {min_samples} samples, got {len(data)}")
    
    @staticmethod
    def validate_cluster_params(n_clusters: int, data_size: int) -> None:
        """Validate clustering parameters."""
        if n_clusters <= 0:
            raise ValidationError("Number of clusters must be positive")
        if n_clusters > data_size:
            raise ValidationError(f"Number of clusters ({n_clusters}) cannot exceed data points ({data_size})")
    
    @staticmethod
    def validate_cluster_range(cluster_range: Tuple[int, int], data_size: int, config: ClusteringConfig) -> Tuple[int, int]:
        """Validate and adjust cluster range."""
        min_k, max_k = cluster_range
        
        if min_k < config.min_clusters:
            min_k = config.min_clusters
        
        max_k = min(max_k, data_size - 1)
        
        if min_k > max_k:
            raise ValidationError(f"Invalid cluster range: min={min_k}, max={max_k}, data_size={data_size}")
        
        return min_k, max_k


class DimensionalityReducer:
    """Optimized dimensionality reduction with registration pattern."""
    
    _REDUCERS: Dict[str, Callable] = {}
    
    def __init__(self, method: str = "pca", config: Optional[ClusteringConfig] = None):
        self.method = method.lower()
        self.config = config or ClusteringConfig()
        self._validate_method()
    
    @classmethod
    def register_reducer(cls, name: str, reducer_func: Callable):
        """Register a new dimensionality reduction method."""
        cls._REDUCERS[name.lower()] = reducer_func
    
    def reduce_data(self, data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Reduce dimensionality of data."""
        if n_components is None:
            n_components = self.config.default_n_components
        
        if n_components <= 0 or n_components > data.shape[1]:
            raise ValidationError(f"Invalid n_components: {n_components}")
        
        if self.method in self._REDUCERS:
            return self._REDUCERS[self.method](data, n_components)
        
        return self._get_default_reducer()(data, n_components)
    
    def _validate_method(self):
        """Validate the reduction method."""
        valid_methods = ["pca", "tsne", "umap"] + list(self._REDUCERS.keys())
        if self.method not in valid_methods:
            raise ValidationError(f"Invalid reduction method: {self.method}")
    
    def _get_default_reducer(self) -> Callable:
        """Get default reducer function based on method."""
        reducers = {
            "pca": self._reduce_with_pca,
            "tsne": self._reduce_with_tsne,
            "umap": self._reduce_with_umap
        }
        return reducers[self.method]
    
    def _reduce_with_pca(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        pca = PCA(n_components=n_components, random_state=self.config.default_random_state)
        return pca.fit_transform(data)
    
    def _reduce_with_tsne(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce dimensionality using t-SNE."""
        tsne = TSNE(n_components=n_components, random_state=self.config.default_random_state)
        return tsne.fit_transform(data)
    
    def _reduce_with_umap(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce dimensionality using UMAP."""
        reducer = umap.UMAP(n_components=n_components, random_state=self.config.default_random_state)
        return reducer.fit_transform(data)


class DataPreprocessor:
    """Optimized data preprocessing pipeline."""
    
    def __init__(self, scale_data: bool = True, reduce_dimensions: bool = False, 
                 reduction_method: str = "pca", config: Optional[ClusteringConfig] = None):
        self.scale_data = scale_data
        self.reduce_dimensions = reduce_dimensions
        self.config = config or ClusteringConfig()
        self.reducer = DimensionalityReducer(reduction_method, self.config) if reduce_dimensions else None
    
    def preprocess(self, data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Preprocess data."""
        processed_data = data.copy()
        
        if self.scale_data:
            scaler = StandardScaler()
            processed_data = scaler.fit_transform(processed_data)
        
        if self.reduce_dimensions and self.reducer:
            if n_components is None:
                n_components = self.config.default_n_components
            processed_data = self.reducer.reduce_data(processed_data, n_components)
        
        return processed_data


class ClusterOptimizer:
    """Optimized cluster optimization without caching."""
    
    def __init__(self, distance_metric: str = "euclidean", decoder=None, config: Optional[ClusteringConfig] = None):
        self.distance_metric = distance_metric
        self.decoder = decoder
        self.config = config or ClusteringConfig()
        self.distance_calculator = DistanceCalculatorFactory.create(distance_metric, decoder)
        self.preprocessor = DataPreprocessor(self.config.scale_data,
            self.config.reduce_dimensions,
            self.config.reduction_method,
            config=self.config)

    def find_optimal_clusters(
        self,
        data: np.ndarray,
        cluster_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """Find optimal clusters."""
        ClusteringValidator.validate_data(data, self.config.min_silhouette_samples)
        min_k, max_k = ClusteringValidator.validate_cluster_range(
            cluster_range or self.config.cluster_range, len(data), self.config
        )
        
        try:
            if self.distance_metric.upper() == "DTW":
                result = self._optimize_dtw_clustering(data, (min_k, max_k))
            else:
                result = self._optimize_euclidean_clustering(data, (min_k, max_k))
            
            return result
            
        except Exception as e:
            logging.warning(f"Cluster optimization failed: {e}")
            return None, -1.0
    
    def _optimize_euclidean_clustering(
        self,
        data: np.ndarray,
        cluster_range: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], float]:
        """Optimize clustering for Euclidean distance."""
        processed_data = self.preprocessor.preprocess(data)
        best_score = -1
        best_labels = None
        
        min_k, max_k = cluster_range
        
        for n_clusters in range(min_k, max_k + 1):
            try:
                labels = self._perform_kmeans_clustering(processed_data, n_clusters)
                
                if not self._is_valid_clustering(labels):
                    continue
                
                score = silhouette_score(processed_data, labels, metric="euclidean")
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    
            except Exception as e:
                logging.debug(f"Failed clustering with {n_clusters} clusters: {e}")
                continue
        
        return best_labels, best_score
    
    def _optimize_dtw_clustering(
        self,
        data: np.ndarray,
        cluster_range: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], float]:
        """Optimize clustering for DTW distance."""
        if self.decoder is None:
            raise ValidationError("Decoder required for DTW clustering")
        processed_data = self.preprocessor.preprocess(data)
        try:
            distance_matrix = self.distance_calculator.compute_distance(processed_data, processed_data)
        except Exception as e:
            raise ValidationError(f"Failed to compute DTW distance matrix: {e}")
        
        best_score = -1
        best_labels = None
        
        min_k, max_k = cluster_range
        
        for n_clusters in range(min_k, max_k + 1):
            try:
                labels = self._perform_agglomerative_clustering(distance_matrix, n_clusters)
                
                if not self._is_valid_clustering(labels):
                    continue
                
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    
            except Exception as e:
                logging.debug(f"Failed DTW clustering with {n_clusters} clusters: {e}")
                continue
        
        return best_labels, best_score
    
    def _perform_kmeans_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=self.config.default_n_init,
            random_state=self.config.default_random_state
        )
        return kmeans.fit_predict(data)
    
    def _perform_agglomerative_clustering(
        self,
        distance_matrix: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """Perform agglomerative clustering on distance matrix."""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
        return clustering.fit_predict(distance_matrix)
    
    def _is_valid_clustering(self, labels: np.ndarray) -> bool:
        """Check if clustering result is valid."""
        return len(np.unique(labels)) >= self.config.min_clusters


# Abstract base class for clustering strategies
class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    def __init__(self, distance_metric: str = "euclidean", decoder=None, clustering_config: Optional[ClusteringConfig] = None):
        self.distance_metric = distance_metric
        self.decoder = decoder
        self.config = clustering_config or ClusteringConfig()
        self.optimizer = ClusterOptimizer(distance_metric, decoder, self.config)
        self.preprocessor = DataPreprocessor(self.config.scale_data,
            self.config.reduce_dimensions,
            self.config.reduction_method,
            config=self.config)
    
    @abstractmethod
    def compute_multilevel_clustering(
        self,
        embeddings: np.ndarray,
        n_main_clusters: Optional[int] = None,
        labels: Optional[np.ndarray] = None,
        cluster_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute multilevel clustering."""
        pass
    
    def _get_main_cluster_labels(
        self, 
        embeddings: np.ndarray, 
        n_main_clusters: Optional[int], 
        labels: Optional[np.ndarray]
    ) -> np.ndarray:
        """Get main cluster labels (shared logic)."""
        if labels is not None:
            return labels
        
        if n_main_clusters is None:
            raise ValidationError("Either provide labels or n_main_clusters")
        
        return self._perform_main_clustering(embeddings, n_main_clusters)
    
    def _perform_main_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform main clustering (K-means with scaling)."""
        processed_data = self.preprocessor.preprocess(embeddings)
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=self.config.default_n_init,
            random_state=self.config.default_random_state
        )
        return kmeans.fit_predict(processed_data)
    
    def _optimal_subclustering(
        self, 
        class_embeddings: np.ndarray, 
        cluster_range: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Perform optimal subclustering for a single main cluster."""
        if cluster_range is None:
            cluster_range = self.config.cluster_range
        
        # Handle edge cases
        if len(class_embeddings) < self.config.min_clusters:
            return np.zeros(len(class_embeddings), dtype=int)
        
        try:
            min_k, max_k = ClusteringValidator.validate_cluster_range(cluster_range, len(class_embeddings), self.config)
            subcluster_labels, _ = self.optimizer.find_optimal_clusters(
                class_embeddings, (min_k, max_k)
            )
            
            if subcluster_labels is None:
                return np.zeros(len(class_embeddings), dtype=int)
            
            return subcluster_labels
            
        except ValidationError:
            return np.zeros(len(class_embeddings), dtype=int)


class BasicClustering(ClusteringStrategy):
    """Optimized basic K-means clustering operations."""
    
    def __init__(self, distance_metric: str = "euclidean", decoder=None, use_optimal_clusters: bool = False, clustering_config: Optional[ClusteringConfig] = None):
        super().__init__(distance_metric, decoder, clustering_config)
        self.use_optimal_clusters = use_optimal_clusters
    
    def compute_multilevel_clustering(
        self,
        embeddings: np.ndarray,
        n_main_clusters: Optional[int] = None,
        labels: Optional[np.ndarray] = None,
        cluster_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute multilevel clustering with Euclidean distance."""
        # Get main cluster labels
        main_cluster_labels = self._get_main_cluster_labels(embeddings, n_main_clusters, labels)
        
        # Perform subclustering for each main cluster
        result = {}
        unique_clusters = np.unique(main_cluster_labels)
        
        for idx, main_cluster in enumerate(unique_clusters):
            indices = np.where(main_cluster_labels == main_cluster)[0]
            class_embeddings = embeddings[indices]
            
            subcluster_labels = self._optimal_subclustering(class_embeddings, cluster_range)
            
            result[f'class_cluster{idx}'] = {
                'embeddings': class_embeddings,
                'cluster_labels': subcluster_labels
            }
        
        return result


class HierarchicalClustering(ClusteringStrategy):
    """Optimized hierarchical clustering operations."""
    
    def __init__(self, distance_metric: str = "DTW", decoder=None, clustering_config: Optional[ClusteringConfig] = None):
        super().__init__(distance_metric, decoder, clustering_config)
    
    def compute_multilevel_clustering(
        self,
        embeddings: np.ndarray,
        n_main_clusters: Optional[int] = None,
        labels: Optional[np.ndarray] = None,
        cluster_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute multilevel hierarchical clustering."""
        # Get main cluster labels
        main_cluster_labels = self._get_main_cluster_labels(embeddings, n_main_clusters, labels)
        
        # Perform subclustering for each main cluster
        result = {}
        unique_clusters = np.unique(main_cluster_labels)
        for idx, main_cluster in enumerate(unique_clusters):
            indices = np.where(main_cluster_labels == main_cluster)[0]
            class_embeddings = embeddings[indices]

            # Use cluster range up to max_clusters or data size
            if cluster_range is None:
                max_clusters = min(self.config.max_clusters, len(class_embeddings))
                cluster_range = (self.config.min_clusters, max_clusters)
                
            subcluster_labels = self._optimal_subclustering(class_embeddings, cluster_range)
            
            result[f'class_cluster{idx}'] = {
                'embeddings': class_embeddings,
                'cluster_labels': subcluster_labels
            }
        
        return result


class ClusterMetricsCalculator:
    """Optimized cluster metrics calculation."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
    
    @staticmethod
    def compute_centroids(embeddings: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids efficiently."""
        if len(embeddings) == 0:
            return np.empty((0, embeddings.shape[1] if len(embeddings.shape) > 1 else 0))
        
        unique_clusters = np.unique(cluster_labels)
        n_features = embeddings.shape[1]
        centroids = np.zeros((len(unique_clusters), n_features))
        
        # Vectorized centroid computation
        for idx, cluster in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster
            centroids[idx] = embeddings[cluster_mask].mean(axis=0)
        
        return centroids
    
    def compute_cluster_metrics_and_centroids(
        self, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive cluster metrics efficiently."""
        
        results = {}
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Compute centroids
        centroids = self.compute_centroids(embeddings, cluster_labels)
        
        # Compute centroid-to-point distances
        centroid_to_point_distances = {}
        for idx, cluster in enumerate(unique_clusters):
            cluster_points = embeddings[cluster_labels == cluster]
            distances = np.linalg.norm(cluster_points - centroids[idx], axis=1)
            centroid_to_point_distances[cluster] = float(distances.mean())
        
        # Compute inter-centroid distances
        if n_clusters > 1:
            centroid_distances = squareform(pdist(centroids, metric='euclidean'))
        else:
            centroid_distances = np.array([[0.0]])
        
        # Compute silhouette score
        s_score = None
        if n_clusters > 1 and len(embeddings) > n_clusters:
            try:
                s_score = silhouette_score(embeddings, cluster_labels)
            except Exception as e:
                logging.warning(f"Failed to compute silhouette score: {e}")
        
        results.update({
            'centroid_to_cluster_point_distances': centroid_to_point_distances,
            'distance_between_centroids': centroid_distances,
            'silhouette_score': s_score,
            'centroids': centroids
        })
        
        return results


class ClusterMerger:
    """Optimized cluster merging operations."""
    
    @staticmethod
    def merge_embeddings_and_labels(cluster_results: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Efficiently merge embeddings and labels from hierarchical clustering results."""
        if not cluster_results:
            return np.empty((0, 0)), np.empty(0, dtype=int)
        
        # Pre-allocate lists for better performance
        merged_embeddings_list = []
        merged_labels_list = []
        current_max_label = -1
        
        for cluster_name, data in cluster_results.items():
            embeddings = data["embeddings"]
            labels = data["cluster_labels"]
            
            # Shift labels to ensure uniqueness across clusters
            shift_value = current_max_label + 1
            shifted_labels = labels + shift_value
            
            merged_embeddings_list.append(embeddings)
            merged_labels_list.append(shifted_labels)
            
            # Update max label more efficiently
            current_max_label = np.max(shifted_labels)
        
        # Use vstack and concatenate for efficiency
        merged_embeddings = np.vstack(merged_embeddings_list)
        merged_labels = np.concatenate(merged_labels_list)
        
        return merged_embeddings, merged_labels


class PrototypeMapper:
    """Optimized prototype mapping without caching."""
    
    def __init__(self, distance_metric: str = "euclidean", decoder=None, config: Optional[ClusteringConfig] = None):
        self.distance_calculator = DistanceCalculatorFactory.create(distance_metric, decoder)
        self.config = config or ClusteringConfig()
    
    def compute_mapping(
        self,
        prototypes: np.ndarray,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Compute mapping between prototypes and centroids."""
        # Compute centroids and their information
        centroid_info = self._compute_centroid_info(embeddings, cluster_labels)
        centroids = self._extract_centroids(centroid_info)
        
        # Map prototypes to centroids
        prototype_mapping = self._map_prototypes_to_centroids(prototypes, centroids, centroid_info)
        
        return prototype_mapping, centroid_info
    
    def _compute_centroid_info(
        self, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """Compute centroids and their average distances to cluster points."""
        unique_labels = np.unique(cluster_labels)
        centroid_info = {}
        
        for idx, label in enumerate(unique_labels):
            cluster_mask = cluster_labels == label
            cluster_embs = embeddings[cluster_mask]
            centroid = cluster_embs.mean(axis=0)
            
            # Compute average distance more efficiently
            avg_distance = None
            if cluster_embs.shape[0] > 0:
                try:
                    distances = self.distance_calculator.compute_distance(
                        cluster_embs, 
                        centroid.reshape(1, -1)
                    )
                    avg_distance = float(np.mean(distances))
                except Exception as e:
                    logging.warning(f"Failed to compute distance for cluster {label}: {e}")
            
            centroid_info[idx] = {
                "label": label,
                "centroid": centroid,
                "average_distance": avg_distance,
                "cluster_size": len(cluster_embs)
            }
        
        return centroid_info
    
    def _extract_centroids(self, centroid_info: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Extract centroid positions from centroid info."""
        if not centroid_info:
            return np.empty((0, 0))
        
        centroids = [info["centroid"] for info in centroid_info.values()]
        return np.vstack(centroids)
    
    def _map_prototypes_to_centroids(
        self,
        prototypes: np.ndarray,
        centroids: np.ndarray,
        centroid_info: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Map each prototype to its nearest centroid efficiently."""
        mapping = {}
        
        if prototypes.shape[0] == 0 or centroids.shape[0] == 0:
            return {i: {"centroid": None, "distance": None, "label": None} 
                   for i in range(prototypes.shape[0])}
        
        try:
            distance_matrix = self.distance_calculator.compute_distance(prototypes, centroids)
            unique_labels = [info["label"] for info in centroid_info.values()]
            
            # Vectorized nearest centroid computation
            nearest_centroid_indices = np.argmin(distance_matrix, axis=1)
            
            for i in range(prototypes.shape[0]):
                nearest_idx = nearest_centroid_indices[i]
                
                mapping[i] = {
                    "centroid": int(nearest_idx),
                    "distance": float(distance_matrix[i, nearest_idx]),
                    "label": unique_labels[nearest_idx],
                    "centroid_info": centroid_info[nearest_idx]
                }
                
        except Exception as e:
            logging.error(f"Failed to compute prototype mapping: {e}")
            mapping = {i: {"centroid": None, "distance": None, "label": None, "centroid_info": None} 
                      for i in range(prototypes.shape[0])}
        
        return mapping


class ClusteringFactory:
    """Factory for creating clustering strategies."""
    
    @staticmethod
    def create_clustering_strategy(
        distance_metric: str, 
        decoder=None, 
        use_optimal_clusters: bool = False,
        clustering_config: Optional[ClusteringConfig] = None
    ) -> ClusteringStrategy:
        """Create appropriate clustering strategy based on distance metric."""
        
        if distance_metric.upper() == "DTW":
            return HierarchicalClustering(distance_metric, decoder, clustering_config)
        else:
            return BasicClustering(distance_metric, decoder, use_optimal_clusters, clustering_config)