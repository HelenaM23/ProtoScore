"""
Visualization module for clustering results with improved structure and clean code.

This module implements visualization for prototype-based clustering results
using various dimensionality reduction techniques.
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install umap-learn to use UMAP reduction.")


class PlotConfig:
    """Configuration for visualization plots."""
    
    def __init__(self, cfg: Any = None):
        """
        Initialize configuration from Hydra config.
        
        Args:
            cfg: Hydra configuration object or None for defaults
        """
        if cfg is None:
            cfg = {}
        
        # Plot dimensions and sizing
        self.figure_size = tuple(getattr(cfg, 'figure_size', [12, 10]))
        self.dpi = getattr(cfg, 'dpi', 300)
        
        # Marker sizes
        self.marker_size_embeddings = getattr(cfg, 'marker_size_embeddings', 50)
        self.marker_size_centroids = getattr(cfg, 'marker_size_centroids', 200)
        self.marker_size_prototypes = getattr(cfg, 'marker_size_prototypes', 300)
        
        # Line and edge styling
        self.edge_linewidth = getattr(cfg, 'edge_linewidth', 2)
        self.line_alpha = getattr(cfg, 'line_alpha', 0.7)
        
        # Legend configuration
        self.legend_position = getattr(cfg, 'legend_position', 'center left')
        self.legend_bbox_to_anchor = tuple(getattr(cfg, 'legend_bbox_to_anchor', [1, 0.5]))
        self.legend_fontsize = getattr(cfg, 'legend_fontsize', 10)
        self.legend_title_fontsize = getattr(cfg, 'legend_title_fontsize', 12)
        
        # Axis and label styling
        self.xlabel_fontsize = getattr(cfg, 'xlabel_fontsize', 14)
        self.ylabel_fontsize = getattr(cfg, 'ylabel_fontsize', 14)
        self.tick_labelsize = getattr(cfg, 'tick_labelsize', 12)
        
        # Dimensionality reduction
        self.default_n_components = getattr(cfg, 'default_n_components', 2)
        self.default_random_state = getattr(cfg, 'default_random_state', 42)
        
        # Save format
        self.save_format = getattr(cfg, 'save_format', 'pdf')


@dataclass
class VisualizationData:
    """Container for visualization data."""
    embeddings_2d: np.ndarray
    cluster_labels: np.ndarray
    centroids_2d: Optional[np.ndarray] = None
    prototypes_2d: Optional[np.ndarray] = None
    class_label: Optional[str] = None


class DimensionReducer(ABC):
    """Abstract base class for dimensionality reduction strategies."""
    
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform data."""
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted reducer."""
        pass


class PCAReducer(DimensionReducer):
    """PCA dimensionality reducer."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._fitted_reducer = None
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data."""
        self._fitted_reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        return self._fitted_reducer.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        if self._fitted_reducer is None:
            raise RuntimeError("Reducer not fitted. Call fit_transform first.")
        return self._fitted_reducer.transform(data)


class TSNEReducer(DimensionReducer):
    """t-SNE dimensionality reducer."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._fitted_reducer = None
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit t-SNE and transform data."""
        self._fitted_reducer = TSNE(
            n_components=self.n_components, 
            random_state=self.random_state
        )
        return self._fitted_reducer.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted t-SNE."""
        warnings.warn("t-SNE doesn't support transform method. Using fit_transform.")
        return self.fit_transform(data)


class UMAPReducer(DimensionReducer):
    """UMAP dimensionality reducer."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install umap-learn package.")
        
        self.n_components = n_components
        self.random_state = random_state
        self._fitted_reducer = None
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit UMAP and transform data."""
        self._fitted_reducer = umap.UMAP(
            n_components=self.n_components, 
            random_state=self.random_state
        )
        return self._fitted_reducer.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted UMAP."""
        if self._fitted_reducer is None:
            raise RuntimeError("Reducer not fitted. Call fit_transform first.")
        return self._fitted_reducer.transform(data)


class ReducerFactory:
    """Factory for creating dimensionality reducers."""
    
    _reducers = {
        'pca': PCAReducer,
        'tsne': TSNEReducer,
        'umap': UMAPReducer,
    }
    
    @classmethod
    def create_reducer(cls, method: str, random_state: int = 42, **kwargs) -> DimensionReducer:
        """Create a reducer instance."""
        method_lower = method.lower()
        if method_lower not in cls._reducers:
            available = list(cls._reducers.keys())
            raise ValueError(f"Method '{method}' not supported. Available: {available}")
        
        # Pass random_state to all reducers
        kwargs['random_state'] = random_state
        return cls._reducers[method_lower](**kwargs)


class PlotRenderer:
    """Handles the actual plotting operations."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        # Color and marker management
        self.marker_shapes = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X", "d"]
        self.colors = plt.get_cmap("tab10", 10).colors
        self.class_colors = {}
        self.cluster_markers = {}
        self.class_label_to_idx = {}
    
    def get_class_color(self, class_label: str, idx: int) -> Tuple[float, float, float]:
        """Get color for a class."""
        if class_label not in self.class_colors:
            color = self.colors[idx % len(self.colors)]
            self.class_colors[class_label] = color
            self.class_label_to_idx[class_label] = idx
        return self.class_colors[class_label]
    
    def get_cluster_marker(self, class_label: str, cluster: int) -> str:
        """Get marker for a specific cluster."""
        cluster_key = (class_label, cluster)
        if cluster_key not in self.cluster_markers:
            cluster_idx = len([k for k in self.cluster_markers.keys() if k[0] == class_label])
            marker = self.marker_shapes[cluster_idx % len(self.marker_shapes)]
            self.cluster_markers[cluster_key] = marker
        return self.cluster_markers[cluster_key]
    
    def plot_embeddings(
        self, 
        ax: plt.Axes, 
        vis_data: VisualizationData, 
        class_idx: int,
        label_legend: List[str]
    ) -> None:
        """Plot embedding points."""
        color = self.get_class_color(vis_data.class_label, class_idx)
        unique_clusters = np.unique(vis_data.cluster_labels)
        
        for cluster in unique_clusters:
            cluster_indices = vis_data.cluster_labels == cluster
            marker = self.get_cluster_marker(vis_data.class_label, cluster)
            
            ax.scatter(
                vis_data.embeddings_2d[cluster_indices, 0],
                vis_data.embeddings_2d[cluster_indices, 1],
                marker=marker,
                color=color,
                s=self.config.marker_size_embeddings,
                label=f"Class {label_legend[class_idx]}, Cluster {cluster}",
                zorder=1,
            )
    
    def plot_centroids(
        self, 
        ax: plt.Axes, 
        vis_data: VisualizationData, 
        class_idx: int,
        label_legend: List[str]
    ) -> None:
        """Plot centroid points."""
        if vis_data.centroids_2d is None:
            return
            
        color = self.get_class_color(vis_data.class_label, class_idx)
        
        ax.scatter(
            vis_data.centroids_2d[:, 0],
            vis_data.centroids_2d[:, 1],
            marker="s",
            facecolor=color,
            s=self.config.marker_size_centroids,
            edgecolors="black",
            linewidths=self.config.edge_linewidth,
            label=f"Centroids of Class {label_legend[class_idx]}",
            zorder=3,
        )
    
    def plot_prototypes(
        self, 
        ax: plt.Axes, 
        prototypes_2d: np.ndarray,
        mapping: Dict[int, Dict[str, Any]],
        centroid_idx_to_class_label: Dict[int, str]
    ) -> None:
        """Plot prototype points."""
        for proto_idx in range(len(prototypes_2d)):
            x, y = prototypes_2d[proto_idx]
            
            # Determine color based on mapping
            color = self._get_prototype_color(proto_idx, mapping, centroid_idx_to_class_label)
            
            ax.scatter(
                x, y,
                marker="o",
                facecolor=color,
                s=self.config.marker_size_prototypes,
                edgecolors="black",
                linewidths=self.config.edge_linewidth,
                label="Prototype" if proto_idx == 0 else "",
                zorder=4,
            )
    
    def plot_connections(
        self, 
        ax: plt.Axes, 
        prototypes_2d: np.ndarray,
        all_centroids_2d: np.ndarray,
        mapping: Dict[int, Dict[str, Any]]
    ) -> None:
        """Plot lines connecting prototypes to centroids."""
        if all_centroids_2d.size == 0:
            return
            
        for proto_idx, info in mapping.items():
            if "centroid" not in info:
                continue
                
            centroid_idx = info["centroid"]
            if centroid_idx >= len(all_centroids_2d):
                continue
                
            proto_point = prototypes_2d[proto_idx]
            centroid_point = all_centroids_2d[centroid_idx]
            
            ax.plot(
                [proto_point[0], centroid_point[0]],
                [proto_point[1], centroid_point[1]],
                "k--",
                alpha=self.config.line_alpha,
                zorder=2,
            )
    
    def add_legend(
        self, 
        ax: plt.Axes,
        label_legend: List[str]
    ) -> None:
        """Add legend to the plot."""
        legend_elements = self._create_legend_elements(label_legend)
        
        ax.legend(
            handles=legend_elements,
            loc=self.config.legend_position,
            bbox_to_anchor=self.config.legend_bbox_to_anchor,
            fontsize=self.config.legend_fontsize,
            title_fontsize=self.config.legend_title_fontsize,
        )
    
    def _create_legend_elements(self, label_legend: List[str]) -> List[Line2D]:
        """Create legend elements."""
        legend_elements = []
        
        # Add class colors
        for class_label, color in self.class_colors.items():
            idx = self.class_label_to_idx[class_label]
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=f"Class {label_legend[idx]}",
                    markerfacecolor=color,
                    markersize=10,
                )
            )
        
        # Add centroids and prototypes
        legend_elements.extend([
            Line2D(
                [0], [0],
                marker="s",
                color="w",
                label="Centroids (colored by class)",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label="Prototypes (colored by class)",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
            )
        ])
        
        return legend_elements
    
    def _get_prototype_color(
        self, 
        proto_idx: int, 
        mapping: Dict[int, Dict[str, Any]],
        centroid_idx_to_class_label: Dict[int, str]
    ) -> Union[Tuple[float, float, float], str]:
        """Get color for prototype based on mapping."""
        if proto_idx in mapping and "centroid" in mapping[proto_idx]:
            centroid_idx = mapping[proto_idx]["centroid"]
            if centroid_idx in centroid_idx_to_class_label:
                class_label = centroid_idx_to_class_label[centroid_idx]
                return self.class_colors.get(class_label, 'gray')
        
        # Fallback color
        colors = list(self.class_colors.values())
        return colors[0] if colors else 'gray'


class ClusteringVisualizer:
    """Main visualizer class for clustering results."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.plot_renderer = PlotRenderer(self.config)
    
    def visualize_clustering(
        self,
        output_clustering: Dict[str, Dict[str, np.ndarray]],
        all_labels_metrics_centroids: Optional[Dict[str, Dict[str, np.ndarray]]],
        prototypes: np.ndarray,
        mapping: Optional[Dict[int, Dict[str, Any]]],
        label_legend: List[str],
        method: str = "pca",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Main visualization method.
        
        Args:
            output_clustering: Clustering results by class
            all_labels_metrics_centroids: Centroid information by class
            prototypes: Prototype arrays
            mapping: Prototype to centroid mapping
            label_legend: Labels for legend
            method: Dimensionality reduction method
            save_path: Path to save plots
        """
        # Handle None parameters gracefully
        all_labels_metrics_centroids = self._handle_none_centroids(
            output_clustering, all_labels_metrics_centroids
        )
        mapping = mapping or self._create_dummy_mapping(len(prototypes))
        
        # Create dimensionality reducer with config
        reducer = ReducerFactory.create_reducer(
            method, 
            random_state=self.config.default_random_state,
            n_components=self.config.default_n_components
        )
        
        # Initialize plotting
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Process and plot data
        all_centroids_2d, centroid_idx_to_class_label = self._process_and_plot_classes(
            ax, output_clustering, all_labels_metrics_centroids, 
            reducer, label_legend
        )
        
        # Process and plot prototypes
        prototypes_2d = self._process_prototypes(prototypes, reducer)
        self.plot_renderer.plot_prototypes(
            ax, prototypes_2d, mapping, centroid_idx_to_class_label
        )
        
        # Plot connections
        self.plot_renderer.plot_connections(
            ax, prototypes_2d, all_centroids_2d, mapping
        )
        
        # Configure plot
        self._configure_plot(ax)
        
        # Add legend to plot
        self.plot_renderer.add_legend(ax, label_legend)
        
        # Save plot
        if save_path:
            self._save_plot(save_path)
    
    def _handle_none_centroids(
        self, 
        output_clustering: Dict[str, Dict[str, np.ndarray]],
        all_labels_metrics_centroids: Optional[Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Handle None centroids by computing from clustering data."""
        if all_labels_metrics_centroids is not None:
            return all_labels_metrics_centroids
        
        print("Warning: all_labels_metrics_centroids is None. Computing from clustering data.")
        centroids = {}
        
        for class_label, data in output_clustering.items():
            embeddings = data["embeddings"]
            cluster_labels = data["cluster_labels"]
            
            class_centroids = []
            for cluster in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster
                centroid = np.mean(embeddings[cluster_mask], axis=0)
                class_centroids.append(centroid)
            
            centroids[class_label] = {"centroids": np.array(class_centroids)}
        
        return centroids
    
    def _create_dummy_mapping(self, n_prototypes: int) -> Dict[int, Dict[str, int]]:
        """Create dummy mapping when none provided."""
        print("Warning: mapping is None. Creating dummy mapping.")
        return {i: {"centroid": i} for i in range(n_prototypes)}
    
    def _process_and_plot_classes(
        self,
        ax: plt.Axes,
        output_clustering: Dict[str, Dict[str, np.ndarray]],
        all_labels_metrics_centroids: Dict[str, Dict[str, np.ndarray]],
        reducer: DimensionReducer,
        label_legend: List[str]
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Process and plot all classes."""
        all_centroids_2d = []
        centroid_idx_to_class_label = {}
        current_offset = 0
        
        for idx, (class_label, data) in enumerate(output_clustering.items()):
            # Prepare visualization data
            vis_data = self._prepare_class_data(
                class_label, data, all_labels_metrics_centroids, reducer
            )
            
            # Plot embeddings and centroids
            self.plot_renderer.plot_embeddings(ax, vis_data, idx, label_legend)
            self.plot_renderer.plot_centroids(ax, vis_data, idx, label_legend)
            
            # Track centroids for connections
            if vis_data.centroids_2d is not None:
                all_centroids_2d.append(vis_data.centroids_2d)
                
                # Map centroid indices to class labels
                for i in range(len(vis_data.centroids_2d)):
                    centroid_idx_to_class_label[current_offset + i] = class_label
                current_offset += len(vis_data.centroids_2d)
        
        # Concatenate all centroids
        all_centroids_2d = np.vstack(all_centroids_2d) if all_centroids_2d else np.array([])
        
        return all_centroids_2d, centroid_idx_to_class_label
    
    def _prepare_class_data(
        self,
        class_label: str,
        data: Dict[str, np.ndarray],
        all_labels_metrics_centroids: Dict[str, Dict[str, np.ndarray]],
        reducer: DimensionReducer
    ) -> VisualizationData:
        """Prepare visualization data for a class."""
        embeddings = data["embeddings"]
        cluster_labels = data["cluster_labels"]
        
        # Reduce dimensionality of embeddings
        if embeddings.shape[1] > 2:
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # Process centroids if available
        centroids_2d = None
        if class_label in all_labels_metrics_centroids:
            centroids = all_labels_metrics_centroids[class_label]["centroids"]
            if centroids.shape[1] > 2:
                centroids_2d = reducer.transform(centroids)
            else:
                centroids_2d = centroids
        
        return VisualizationData(
            embeddings_2d=embeddings_2d,
            cluster_labels=cluster_labels,
            centroids_2d=centroids_2d,
            class_label=class_label
        )
    
    def _process_prototypes(
        self, 
        prototypes: np.ndarray, 
        reducer: DimensionReducer
    ) -> np.ndarray:
        """Process prototypes for visualization."""
        if prototypes.shape[1] > 2:
            return reducer.transform(prototypes)
        return prototypes
    
    def _configure_plot(self, ax: plt.Axes) -> None:
        """Configure plot appearance."""
        ax.set_xlabel("Dimension 1", fontsize=self.config.xlabel_fontsize)
        ax.set_ylabel("Dimension 2", fontsize=self.config.ylabel_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.config.tick_labelsize)
        plt.tight_layout()
    
    def _save_plot(self, save_path: str) -> None:
        """Save the main plot."""
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f"clustering_plot.{self.config.save_format}")
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")