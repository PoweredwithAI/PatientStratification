from __future__ import annotations
from pathlib import Path

from unsup_clustering.config import load_config
from unsup_clustering.data_io import load_raw_data
from unsup_clustering.preprocessing import clean_and_normalize
from unsup_clustering.dimensionality_reduction import run_pca, AutoencoderReducer
from unsup_clustering.clustering import (
    run_clustering_on_data,
    run_clustering_on_pca,
    run_clustering_on_autoencoder,
)
from unsup_clustering.visualization import plot_umap_for_all_clusterings
from unsup_clustering.evaluation import cosine_similarity_analysis


def main(config_path: str = "examples/example_config.yaml") -> None:
    cfg = load_config(config_path)

    df_raw = load_raw_data(cfg.data_path)
    df_clean = clean_and_normalize(df_raw)

    # Original space clustering
    results_raw = run_clustering_on_data(df_clean, n_clusters=cfg.n_clusters, random_state=cfg.random_seed)

    # PCA
    df_pca, _ = run_pca(df_clean, n_components=2)  # or the n_components used in notebook
    results_pca = run_clustering_on_pca(df_pca, n_clusters=cfg.n_clusters, random_state=cfg.random_seed)

    # Autoencoder
    ae = AutoencoderReducer(input_dim=df_clean.shape[1], latent_dim=cfg.ae_latent_dim)
    ae.fit(df_clean, epochs=50, batch_size=32)  # replace with notebook values
    df_ae = ae.transform(df_clean)
    results_ae = run_clustering_on_autoencoder(df_ae, n_clusters=cfg.n_clusters, random_state=cfg.random_seed)

    # Visualizations and similarity analysis
    plot_umap_for_all_clusterings(df_clean, results_raw)
    sim_df = cosine_similarity_analysis(df_clean)
    out_sim_path = Path(cfg.output_dir) / "cosine_similarity.csv"
    out_sim_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_sim_path)


if __name__ == "__main__":
    main()
