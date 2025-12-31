# src/__init__.py - Re-export all functions for easy imports
from .config import Config
from .preprocessing import clean_and_normalize
from .dimensionality_reduction import run_pca, Autoencoder, train_autoencoder
from .clustering import (
    clustering_analysis,
    run_variance_weighted_agglomerative
)
from .visualization import (
    plot_umap_for_all_clusterings,
    plot_knee_detection,
    plot_agglomerative_dendrogram,
    plot_weighted_umap)

__all__ = [
    'Config', 'clean_and_normalize', 'run_pca', 
    'Autoencoder', 'train_autoencoder','clustering_analysis', 'run_variance_weighted_agglomerative',
    'plot_umap_for_all_clusterings', 'plot_knee_detection', 
    'plot_agglomerative_dendrogram','plot_weighted_umap'
]
