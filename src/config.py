from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    data_path: Path
    output_dir: Path
    random_seed: int
    # number of clusters chosen around elbow point (e.g., k=8)
    n_clusters: int = 8
    # autoencoder latent dimensions (e.g., 10)
    ae_latent_dim: int = 10


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        data_path=Path(raw["data_path"]),
        output_dir=Path(raw["output_dir"]),
        random_seed=int(raw.get("random_seed", 42)),
        n_clusters=int(raw.get("n_clusters", 8)),
        ae_latent_dim=int(raw.get("ae_latent_dim", 10)),
    )
