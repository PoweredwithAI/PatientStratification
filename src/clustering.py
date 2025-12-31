from __future__ import annotations
import pandas as pd
from typing import Dict, Any
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def clustering_analysis(
    df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generalized version of 'Clustering Analysis Function' cell.
    Run KMeans, Spectral, and Agglomerative and return labels and scores.
    Run for all types of data: raw, PCA, AE.
        
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with omics features.
    n_clusters : int
        Number of clusters for clustering algorithms.
    random_state : int
        Random seed for reproducibility, default is 42.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing silhouette scores, labels, and label counts for each clustering method.    
    """
    results = {}
    
    # Section : KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)                       # Scikit-learn's KMeans function
    labels_kmeans = kmeans.fit_predict(df)                          # Fit and predict cluster labels
    results['kmeans'] = {
        'silhouette_score': silhouette_score(df, labels_kmeans),
        'labels': labels_kmeans,
        'label_counts': dict(zip(*np.unique(labels_kmeans, return_counts=True)))
    }
    
    # Section : Spectral Clustering with automatic gamma tuning
    #dists = pairwise_distances(X, metric='cosine', n_jobs=-1) # cosine or correlation are ideal choices instead of euclidean, but were not used here due to system limitations
    #median_dist = np.median(dists)
    #gamma = 1/(2*median_dist**2) if median_dist > 0 else 1.0
    gamma=0.0005 # Set gamma to a small value for better clustering - preference is to use the above code with a cosine metric, but it was not possible due to system limitations
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', 
                                gamma=gamma, random_state=random_state)                            # Scikit-learn's Spectral Clustering function
    labels_spectral = spectral.fit_predict(df)
   
    results['spectral'] = {
        'silhouette_score': silhouette_score(df, labels_spectral),
        'labels': labels_spectral,
        'label_counts': dict(zip(*np.unique(labels_spectral, return_counts=True)))
    }
    
    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')                # Scikit-learn's Agglomerative Clustering function
    labels_agglo = agglo.fit_predict(df)
    results['agglo'] = {
        'silhouette_score': silhouette_score(df, labels_agglo),
        'labels': labels_agglo,
        'label_counts': dict(zip(*np.unique(labels_agglo, return_counts=True)))
    }
    
    return results

def run_variance_weighted_agglomerative(X_data, sample_ids=None, random_state=42):
    """
    Agglomerative Clustering using Variance-Weighted Cosine Distance.
    Weights features based on their variance to compute a weighted cosine distance matrix.
        
    Parameters
    ----------
    X_data : np.ndarray
        Input data matrix with omics features.
    sample_ids : list, optional
        List of sample IDs corresponding to rows in X_data.
    random_state : int
        Random seed for reproducibility, default is 42.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing clustering labels, silhouette score, optimal k, distance matrix, and weights.    
    """
    
    # Weights based on variance
    feature_variances = np.var(X_data, axis=0)
    var_weights = feature_variances / np.sum(feature_variances)
    X_weighted = X_data * var_weights
    
    # Cosine distance matrix
    dist_matrix = 1 - cosine_similarity(X_weighted)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 2)
    
    # Find optimal k via silhouette scores
    silhouette_scores = []
    k_values = list(range(3, 21))
    for k in k_values:
        clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(dist_matrix)
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)
    
    optimal_k = k_values[np.argmin(silhouette_scores)]
    
    # Final clustering at optimal k
    clustering_optimal = AgglomerativeClustering(n_clusters=optimal_k, metric='precomputed', linkage='complete')
    labels_optimal = clustering_optimal.fit_predict(dist_matrix)
    score_optimal = silhouette_score(dist_matrix, labels_optimal, metric='precomputed')
    
    results = {
        'method': 'Agglomerative (Variance-Weighted Cosine)',
        'labels': labels_optimal,
        'silhouette_score': score_optimal,
        'optimal_k': optimal_k,
        'k_values': k_values,
        'silhouette_scores': silhouette_scores,
        'dist_matrix': dist_matrix,
        'X_weighted': X_weighted,
        'var_weights': var_weights,
        'sample_ids': sample_ids
    }
    
    return results