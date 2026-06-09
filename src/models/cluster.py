"""Unsupervised clustering: UMAP dimensionality reduction + KMeans/HDBSCAN."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

PROCESSED_DIR = Path("data/processed")


def reduce_and_cluster(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    umap_dims: int = 2,
    use_hdbscan: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (umap_2d, cluster_labels)."""
    reducer = umap.UMAP(n_components=umap_dims, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)

    if use_hdbscan and HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5)
        labels = clusterer.fit_predict(reduced)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = clusterer.fit_predict(reduced)

    return reduced, labels


def run():
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    with open(PROCESSED_DIR / "posts_filtered.json") as f:
        posts = json.load(f)

    reduced_2d, labels = reduce_and_cluster(embeddings, n_clusters=8)
    reduced_3d, _ = reduce_and_cluster(embeddings, umap_dims=3)

    df = pd.DataFrame(posts)
    df["cluster"] = labels
    df["umap_x"] = reduced_2d[:, 0]
    df["umap_y"] = reduced_2d[:, 1]
    df["umap_z"] = reduced_3d[:, 2]

    df.to_csv(PROCESSED_DIR / "clustered.csv", index=False)
    np.save(PROCESSED_DIR / "umap_2d.npy", reduced_2d)
    np.save(PROCESSED_DIR / "umap_3d.npy", reduced_3d)
    print(f"Clusters: {df['cluster'].value_counts().to_dict()}")


if __name__ == "__main__":
    run()
