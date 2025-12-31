from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scipy.spatial.distance import squareform 
from scipy.cluster.hierarchy import linkage, dendrogram 
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def plot_umap_for_all_clusterings(
    df_clean: pd.DataFrame,
    results_raw: Dict[str, dict],
    df_pca: Optional[pd.DataFrame] = None,
    results_pca: Optional[Dict[str, dict]] = None,
    df_ae: Optional[pd.DataFrame] = None,
    results_ae: Optional[Dict[str, dict]] = None,
    random_state: int = 42
) -> plt.Figure:

    """
    Comprehensive UMAP visualization for ALL clustering results across data spaces.
    Creates 3 subplots per data space (raw/PCA/AE) showing KMeans, Spectral, Agglomerative.

    Parameters
    ---------- 
    df_clean : pd.DataFrame
        Cleaned DataFrame with omics features.
    results_raw : Dict[str, dict]
        Clustering results on raw data.
    df_pca : Optional[pd.DataFrame], optional
        PCA reduced DataFrame, by default None.
    results_pca : Optional[Dict[str, dict]], optional
        Clustering results on PCA data, by default None.
    df_ae : Optional[pd.DataFrame], optional
        Autoencoder features DataFrame, by default None.
    results_ae : Optional[Dict[str, dict]], optional
        Clustering results on AE data, by default None.
    random_state : int
        Random seed for reproducibility, default is 42.
    
    Returns
    -------
    plt.Figure
        Figure containing UMAP plots for all clustering results.

    """
    # Prepare datasets and results for plotting
    datasets = [
        ("Raw Data", df_clean, results_raw),
    ]
    
    if df_pca is not None and results_pca is not None:
        datasets.append(("PCA Reduced", df_pca, results_pca))
    if df_ae is not None and results_ae is not None:
        datasets.append(("Autoencoder", df_ae, results_ae))
    
    # Create subplots for each dataset
    n_rows = len(datasets)
    fig, axs = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    for row_idx, (data_name, X_data, results) in enumerate(datasets):
        # Extract the 3 clustering methods (assumes consistent keys)
        clustering_methods = ['kmeans', 'spectral', 'agglo']
        
        label_list = [results[method]['labels'] for method in clustering_methods]
        score_list = [results[method]['silhouette_score'] for method in clustering_methods]
        
        reducer = umap.UMAP(random_state=random_state)                              # UMAP reducer
        X_umap = reducer.fit_transform(X_data)
        
        for i, (method, labels, score) in enumerate(zip(clustering_methods, label_list, score_list)):
            ax = axs[row_idx, i]
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=labels,
                palette='tab10',
                s=30,
                ax=ax,
                legend=False
            )
            ax.set_title(f"{data_name}: {method.title()}\nSilhouette: {score:.4f}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
    
    plt.tight_layout()
    return fig

def plot_knee_detection(k_values, silhouette_scores, optimal_k, ax=None):
    """
    Plot silhouette score vs k (for agglomorative clustering) with optimal point highlighted.
    
    Parameters
    ---------- 
    k_values : List[int]
        List of k values tested.
    silhouette_scores : List[float]
        Corresponding silhouette scores.
    optimal_k : int
        Optimal k determined by knee detection.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes to plot on, by default None.
    Returns
    -------
    plt.Axes
        Axes containing the silhouette score plot.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=6)
    ax.axvline(x=optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}', linewidth=2)
    ax.set_title("Silhouette Score vs Number of Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_agglomerative_dendrogram(dist_matrix, sample_ids=None, truncate_p=100, ax=None):
    """
    Plot dendrogram for agglomerative clustering using variance-weighted cosine distance.
    
    Parameters
    ---------- 
    dist_matrix : np.ndarray
        Precomputed distance matrix.
    sample_ids : list, optional
        List of sample IDs for labeling.
    truncate_p : int, optional
        Number of samples to show in dendrogram.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes to plot on, by default None.

    Returns
    -------
    plt.Axes
        Axes containing the dendrogram plot.        
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    condensed_dist = squareform(dist_matrix, checks=False)
    linked = linkage(condensed_dist, method='complete')
    dendrogram(linked, ax=ax, truncate_mode='lastp', p=truncate_p, 
               leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Agglomerative Clustering Dendrogram (Variance-Weighted Cosine)")
    ax.set_xlabel("Sample index or cluster size")
    ax.set_ylabel("Cosine Distance")
    return ax

def plot_weighted_umap(X_weighted, labels, title="UMAP (Variance-Weighted Cosine)", ax=None):
    """
    UMAP with cosine metric on weighted data.
    
    Parameters
    ---------- 
    X_weighted : np.ndarray
        Variance-weighted feature matrix.
    labels : np.ndarray
        Cluster labels.
    title : str, optional
        Title for the plot.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes to plot on, by default None.

    Returns
    -------
    plt.Axes
        Axes containing the UMAP plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    reducer = umap.UMAP(metric='cosine', random_state=42)
    X_umap = reducer.fit_transform(X_weighted)
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, 
                   palette='tab10', s=40, ax=ax, legend='brief')
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return ax
