import streamlit as st
import yaml
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Core imports from your package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now your imports work exactly as written:
from src.config import Config
from src import (  # ← src/__init__.py makes this work!
    clean_and_normalize, run_pca, Autoencoder, train_autoencoder,clustering_analysis,run_variance_weighted_agglomerative,
    plot_umap_for_all_clusterings,plot_knee_detection, plot_agglomerative_dendrogram, plot_weighted_umap
)

st.set_page_config(
    page_title="Patient Clustering Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_streamlit_config():
    """Load Streamlit-specific config."""
    with open("streamlit/config.yaml", "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        data_path=Path(raw["data_path"]),
        output_dir=Path(raw["output_dir"]),
        random_seed=raw["random_seed"],
        n_clusters=raw["n_clusters"],
        ae_latent_dim=raw["ae_latent_dim"],
    )

def show_clustering_metrics(results_ae):
    """One-liner for metrics display."""

    st.subheader("📊 Clustering Metrics")    
    for method, data in results_ae.items():
        st.write(f"{method.upper()}: Silhouette = {data['silhouette_score']:.4f}")
        st.caption(f"Distribution: {data['label_counts']}")

def show_results_ui():
    """Display the results section after pipeline run."""
    if not st.session_state.get("pipeline_ran", False):
        return

    df_clean = st.session_state.df_clean
    df_pca = st.session_state.df_pca
    df_ae = st.session_state.df_ae
    results_raw = st.session_state.results_raw
    results_pca = st.session_state.results_pca
    results_ae = st.session_state.results_ae
    agg_results = st.session_state.get("agg_results", None)

    labels_k, score_k, dendro_k = None, None, None
    # Visualization status + global UMAP
    with st.status("5. Generating visualization section...", expanded=True):
        st.header("📊 Visualizations for Clustering Results")
        fig_umap = plot_umap_for_all_clusterings(
            df_clean, results_raw,
            df_pca=df_pca, results_pca=results_pca,
            df_ae=df_ae, results_ae=results_ae
        )
        st.pyplot(fig_umap)

        tab1, tab2, tab3 = st.tabs(["All Methods", "Raw Data Only", "Download"])

        with tab1:
            st.subheader("Comprehensive Clustering Comparison")
            results_df = pd.DataFrame({
                "Method": ["KMeans (Raw)", "Spectral (Raw)", "Agglomerative (Raw)",
                           "KMeans (PCA)", "KMeans (AE)"],
                "Data Shape": [df_clean.shape, df_clean.shape, df_clean.shape,
                               df_pca.shape, df_ae.shape]
            })
            st.dataframe(results_df)

        with tab2:
            fig_raw = plot_umap_for_all_clusterings(df_clean, results_raw)
            st.pyplot(fig_raw)

        with tab3:
            st.download_button(
                "💾 Download all results",
                data="Results saved to artifacts/",
                file_name="clustering_results.zip"
            )

    if agg_results is not None:
        st.header("📊 Visualizations for Variance-Weighted Agglomorative Clustering Results")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📈 Knee Detection (Silhouette Analysis)")
            fig_knee, ax_knee = plt.subplots(figsize=(10, 4))
            plot_knee_detection(
                agg_results['k_values'],
                agg_results['silhouette_scores'],
                agg_results['optimal_k'],
                ax=ax_knee
            )
            st.pyplot(fig_knee)

        with col2:
            st.subheader("🎚️ Select k for dendrogram (local)")

            # initialize default once from current pipeline k
            if "dendro_k" not in st.session_state:
                # use the k that was used in the last pipeline run
                st.session_state.dendro_k = st.session_state.get("last_pipeline_k", 8)

            dendro_k = st.slider(
                "Choose k:",
                3, 20, st.session_state.dendro_k, 1,
                key="dendro_k"
            )

            clustering_k = AgglomerativeClustering(
                n_clusters=dendro_k, metric='precomputed', linkage='complete'
            )
            labels_k = clustering_k.fit_predict(agg_results['dist_matrix'])
            score_k = silhouette_score(
                agg_results['dist_matrix'],
                labels_k,
                metric='precomputed'
            )
            st.info(f"Silhouette Score at k={dendro_k}: {score_k:.4f}")


    # Dendrogram & UMAP tabs
    if agg_results is not None and labels_k is not None:
        tab_agg1, tab_agg2, tab_agg3 = st.tabs(["Dendrogram", "UMAP", "Annotations"])

        with tab_agg1:
            fig_dendro, ax_dendro = plt.subplots(figsize=(14, 6))
            plot_agglomerative_dendrogram(
                agg_results['dist_matrix'],
                ax=ax_dendro
            )
            st.pyplot(fig_dendro)

        with tab_agg2:
            fig_umap, ax_umap = plt.subplots(figsize=(10, 7))
            plot_weighted_umap(
                agg_results['X_weighted'],
                labels_k,
                f"UMAP (k={dendro_k}, Silhouette: {score_k:.4f})",
                ax=ax_umap
            )
            st.pyplot(fig_umap)

        with tab_agg3:
            annotations = pd.DataFrame({
                "Sample_id": agg_results.get('sample_ids', range(len(labels_k))),
                "Cluster_id": labels_k,
                "Silhouette_Score": score_k
            })
            st.dataframe(annotations.head(10))
            csv = annotations.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Cluster Annotations",
                data=csv,
                file_name=f"weighted_cosine_clusters_k{dendro_k}.csv",
                mime="text/csv"
            )





def main():
    """Main Streamlit app function."""

    st.title("🔬 Patient Clustering Dashboard")
    st.markdown("**Clustering analysis with PCA, Autoencoder, and multiple algorithms for RNA-seq data**")

    # Sidebar controls
    st.sidebar.header("📊 Configuration")
    cfg = load_streamlit_config()
    
    data_file = st.sidebar.file_uploader(
        "Upload dataset (CSV)",
        type="csv",
        help="Upload your high-dimensional dataset"
    )
    
    st.sidebar.markdown("---")

    # --- 1. Feature Quality Controls ---
    st.sidebar.markdown("### Feature Quality Controls")

    sparsity_thresh = st.sidebar.slider(
        "Sparsity threshold",
        0.5, 0.99, 0.985, 0.005
    )  

    feature_var_cutoff = st.sidebar.slider(
        "Feature variance cutoff",
        0.0, 1.0, 0.015, 0.005
    )  

    use_log_transform = st.sidebar.checkbox(
        "Skew control (log transform)",
        value=True
    )  

    st.sidebar.markdown("---")

    # --- 2. Normalization & Outlier Removal Controls ---
    st.sidebar.markdown("### Normalization & Outlier Removal Controls")

    use_robust_scaler = st.sidebar.checkbox(
        "Standardization (RobustScaler)",
        value=True
    )  

    contamination = st.sidebar.slider(
        "Outlier contamination",
        0.01, 0.10, 0.01, 0.01
    )  

    st.sidebar.markdown("---")
    # --- 3. Clustering Controls ---
    st.sidebar.markdown("### Clustering Controls")

    n_clusters = st.sidebar.slider(
        "Number of clusters (k)",
        2, 20, cfg.n_clusters
    )  

    random_seed = st.sidebar.slider(
        "Random seed",
        0, 100, cfg.random_seed
    )  
    st.sidebar.markdown("---")
    
    # Run button
    if st.sidebar.button("🚀 Run Full Pipeline", type="primary"):
        with st.spinner("Running unsupervised analysis..."):
            run_full_pipeline(
                data_file, n_clusters, random_seed, 
                sparsity_thresh, feature_var_cutoff,contamination
            )

def run_full_pipeline(data_file, n_clusters, random_seed, sparsity_thresh, feature_var_cutoff, contamination):
    if data_file is not None:
        # Save uploaded file
        df_raw = pd.read_csv(data_file)
        st.session_state.df_raw = df_raw
        
        st.header("📈 Raw Data Overview")
        st.dataframe(df_raw.head())
        st.write("Input data:", f"Patient Samples: {df_raw.shape[0]:,} × Omics data points: {df_raw.shape[1]-1:,}")
        
        
        st.header("Preprocessing Progress")
        
        # Step 1: Clean & normalize
        with st.status("1. Removing sparse/zero-variance features...", expanded=True):
            df_clean, skewed_figure,skewed_figure_log, sparse_features, low_variance_features, common_features,outliers = clean_and_normalize(
                df_raw, sparsity_threshold=sparsity_thresh, feature_var_threshold=feature_var_cutoff, contamination=contamination
            )
            st.write("Input data:", f"Patient Samples: {df_raw.shape[0]:,} × Omics Features: {df_raw.shape[1]-1:,}")
            st.write(f"Omics Features which are {sparsity_thresh*100} % or more empty: {len(sparse_features)}")
            st.caption(f"Features : {sparse_features}")
            st.write(f"Omics features which have variance <{feature_var_cutoff*100}% across samples: {len(low_variance_features)}")
            st.caption(f"Features : {low_variance_features}")
            st.write(f"Common features (sparse + low variance): {len(common_features)} are being dropped.")
            st.caption(f"Features : {common_features}")
            st.write(f"Removed {outliers} outliers")
            st.write(f"Skewness in unscaled data.")
            st.pyplot(skewed_figure)
            st.success(f"✅ Clean data: {df_clean.shape}")
            st.write(f"Skewness in log transformed data.")
            st.pyplot(skewed_figure_log)

            st.session_state.df_clean = df_clean
        
        # Step 2: Clustering on original space
        with st.status("2. Clustering on cleaned data...", expanded=True):
            results_raw = clustering_analysis(
                df_clean, n_clusters, random_seed
            )
            st.session_state.results_raw = results_raw
            show_clustering_metrics(results_raw)

        # Step 3: Dimensionality reduction
        
        with st.status("3a. PCA reduction...", expanded=True):
            df_pca, pca_model = run_pca(df_clean, n_components=10)
            st.success(f"✅ Dataset reduced to PCA dataset with 10 dimensions")
        st.session_state.df_pca = df_pca
        
        with st.status("3b. Clustering on PCA...", expanded=True):
            results_pca = clustering_analysis(df_pca, n_clusters, random_seed)
            st.session_state.results_pca = results_pca
            show_clustering_metrics(results_pca)
        

        with st.status("3c. Training Autoencoder...", expanded=True):
            try:
                df_ae, history = train_autoencoder(df_clean, epochs=50, batch_size=256)
                st.session_state.df_ae = df_ae
                st.metric("Final Val Loss", f"{history[-1]['val_loss']:.4f}")
                st.success("✅ Autoencoder training completed!")
                st.success(f"✅ Dataset reduced to AE dataset with 10 dimensions")
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.session_state.df_ae = None
                st.stop()

        # Clustering only runs if AE succeeded
        if st.session_state.df_ae is not None:
            with st.status("3d. Clustering on Autoencoder features...", expanded=True):
                results_ae = clustering_analysis(
                    st.session_state.df_ae, n_clusters, random_seed
                )
                st.session_state.results_ae = results_ae
                st.success("✅ Clustering completed!")
                show_clustering_metrics(results_ae)
        else:
            st.warning("⚠️ Skipping clustering - Autoencoder failed")


        with st.status("4. Variance-Weighted Agglomerative Clustering...", expanded=True):
            # Run on PCA data (best for this method)
            if 'df_pca' in st.session_state:
                agg_results = run_variance_weighted_agglomerative(
                    st.session_state.df_pca, 
                    sample_ids=df_clean.index.tolist() if hasattr(df_clean, 'index') else None
                )
                st.session_state.agg_results = agg_results
                st.success(f"✅ Optimal k={agg_results['optimal_k']}, Silhouette: {agg_results['silhouette_score']:.4f}")

        st.session_state.pipeline_ran = True

        

if __name__ == "__main__":
    main()
    if st.session_state.get("pipeline_ran", False):
        show_results_ui()
