"""Unsupervised clustering: UMAP dimensionality reduction + KMeans."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 8


def run():
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    with open(PROCESSED_DIR / "posts_filtered.json") as f:
        posts = json.load(f)

    print("Fitting UMAP (3D)...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_3d = reducer.fit_transform(embeddings)

    print("Fitting KMeans...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(reduced_3d)

    df = pd.DataFrame(posts)
    df["cluster"] = labels
    df["umap_x"] = reduced_3d[:, 0]
    df["umap_y"] = reduced_3d[:, 1]
    df["umap_z"] = reduced_3d[:, 2]

    df.to_csv(PROCESSED_DIR / "clustered.csv", index=False)
    np.save(PROCESSED_DIR / "umap_3d.npy", reduced_3d)
    np.save(PROCESSED_DIR / "embeddings.npy", embeddings)

    # Save models so the app can project new images
    joblib.dump(reducer, MODELS_DIR / "umap_reducer.pkl")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.pkl")

    print(f"Clusters: {df['cluster'].value_counts().sort_index().to_dict()}")
    print(f"Saved umap_reducer.pkl and kmeans.pkl to {MODELS_DIR}")


if __name__ == "__main__":
    run()
