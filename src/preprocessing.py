from __future__ import annotations
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def drop_sparse_and_zero_variance_features(df: pd.DataFrame, sparsity_threshold: float =0.985, feature_var_threshold: float = 0.015):
    
    """
    Initial cleaning step to drop sparse and zero-variance Omic features 
    Conservative thresholds by and dropping features which overlap in both categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with omics features.
    sparsity_threshold : float
        Threshold for sparsity (e.g., default is 0.985 for 98.5%)
    feature_var_threshold : float
        Threshold for feature variance (e.g., default is 0.015 for 1.5%)

    Returns
    -------
    X_cleaned : pd.DataFrame
        Cleaned DataFrame with sparse and low-variance features removed.
    sparse_features : List[str]
        List of sparse features removed.
    low_variance_features : List[str]
        List of low-variance features removed.
    common_features : List[str]
        List of common features removed.
    """
        
    # Checking data for sparcity (percentage of zeros in each feature)
    X = df.drop(columns=["index"])
    zero_fraction = (X == 0).sum(axis=0) / X.shape[0]                                           # Fraction of zeros per feature
    sparse_features = zero_fraction[zero_fraction > sparsity_threshold].index                   # Features exceeding sparsity threshold
    print(f"Number of sparse features (>{sparsity_threshold}% zeros): {len(sparse_features)}")

    # Checking data for zero-variance features (features which do not vary across samples)

    feature_variance = X.var(axis=0)                                                             # Variance per feature   
    low_variance_features = feature_variance[feature_variance < feature_var_threshold].index     # Features below variance threshold  
    print(f"Number of low variance features (<{feature_var_threshold*100}% variance): {len(low_variance_features)}")    

    # Common features (sparce and low variance)
    common_features = sparse_features.intersection(low_variance_features)           # Features that are both sparse and low variance  
    print(f"Common features (sparse + low variance): {len(common_features)}")

    # Removing common features
    X_cleaned = X.drop(columns=common_features)                                     # Drop common features from data
    print(f"Shape after removing common features: {X_cleaned.shape}")
    return X_cleaned, sparse_features, low_variance_features, common_features


def log_transform_and_check_skewness(df: pd.DataFrame):
    """
    Function to log transform the data and check skewness distribution before and after log transformation.
    Current implementation uses log1p to handle zeros only, recommend exploring other transformations as needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Cleaned DataFrame with omics features.
    
    Returns
    -------
    X_log : pd.DataFrame
        Log transformed DataFrame.
    fig : matplotlib.figure.Figure
        Histogram of skewness before log transformation.
    fig_log : matplotlib.figure.Figure
        Histogram of skewness after log transformation
    """
    feature_skewness = pd.Series(skew(df, axis=0))

    plt.figure(figsize=(12, 4))
    sns.histplot(feature_skewness, bins=50)
    plt.title("Feature Skewness Distribution")
    plt.xlabel("Skewness")
    plt.ylabel("Number of Features")
    fig=plt.gcf()


    print(f"Highly skewed features (|skew| > 2): {(np.abs(feature_skewness) > 2).sum()}")


    # Since >90% of the features are highly skewed, we will apply log transformation to reduce skewness.
    # Log transformation to reduce skewness
    X_log = np.log1p(df) 

    feature_skewness_log = pd.Series(skew(X_log, axis=0))
    plt.figure(figsize=(12, 4))
    sns.histplot(feature_skewness_log, bins=50)
    plt.title("Feature Skewness Distribution post Log Transformation")
    plt.xlabel("Skewness")
    plt.ylabel("Number of Features")
    fig_log=plt.gcf()

    return X_log, fig, fig_log


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to scale data using RobustScaler (Transforms data to have a mean of zero and a standard deviation of one).    
    
    Parameters
    ----------
    df : pd.DataFrame
        Input transformed DataFrame.
    
    Returns
    -------
    df_scaled : pd.DataFrame
        Scaled DataFrame.
    """
        
    scalerr = RobustScaler()
    scaled_array = scalerr.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
    return df_scaled


def remove_outliers(df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42):
    """
    Function to remove outliers using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Input scaled and transformed DataFrame with omics features.
    contamination : float
        The proportion of outliers in the data set.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    X_robust_cleaned : pd.DataFrame
        DataFrame with outliers removed.
    mask.sum() : int
        Sum of boolean masks indicating which rows were removed as outliers -> number of outliers removed.
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    labels = iso.fit_predict(df.values)
    mask = (labels == -1)
    X_robust_cleaned = df[~mask]
    print(f"Removed {mask.sum()} outliers. New shape: {X_robust_cleaned.shape}")
    return X_robust_cleaned, mask.sum()


def clean_and_normalize(df: pd.DataFrame, sparsity_threshold: float = 0.985, feature_var_threshold: float = 0.015, contamination: float = 0.01) -> pd.DataFrame:
    """
    Full preprocessing pipeline combining sparsity filtering, log transform,
    scaling, and outlier removal.

    Parameters
    ---------- 
    df : pd.DataFrame
        Input raw DataFrame with omics features.
    sparsity_threshold : float
        Threshold for sparsity (e.g., default is 0.985 for 98.5%)
    feature_var_threshold : float
        Threshold for feature variance (e.g., default is 0.015 for 1.5%)
    contamination : float
        The proportion of outliers in the data set.
    
    Returns
    -------
    df : pd.DataFrame
        Cleaned and normalized DataFrame.
    fig : matplotlib.figure.Figure
        Histogram of skewness before log transformation.    
    fig_log : matplotlib.figure.Figure
        Histogram of skewness after log transformation.
    sparse_features : List[str]
        List of sparse features removed.
    low_variance_features : List[str]
        List of low-variance features removed.
    common_features : List[str]
        List of common features removed.
    outliers : int
        Number of outliers removed.
    """
    df,sparse_features, low_variance_features, common_features = drop_sparse_and_zero_variance_features(df, sparsity_threshold=sparsity_threshold, feature_var_threshold=feature_var_threshold)
    df , fig, fig_log = log_transform_and_check_skewness(df)
    df = scale_data(df)
    df,outliers = remove_outliers(df, contamination=contamination)
    return df, fig, fig_log, sparse_features, low_variance_features, common_features, outliers
