"""Streamlit-specific glue — imported by both Score_Photos.py and pages/.

The actual scoring pipeline (CLIP embeddings, regression/classifier scoring,
clustering, similarity search) lives in src/scoring/scorer.py, which has no
Streamlit dependency and is shared with the FastAPI service in src/api/main.py.
This module just wires that pipeline into Streamlit's resource caching and
re-exports the UI helpers pages/ import.
"""

from __future__ import annotations

import streamlit as st

from src.scoring.scorer import (
    ATTRIBUTE_IMPACT,
    CLUSTER_DESCRIPTIONS,
    CLUSTER_NAMES,
    PHOTO_GENRES,
    TAG_COLORS,
    VISUAL_ATTRIBUTES,
    Resources,
    _analyse_image_clip,
    _cluster_attrs,
    _dominant_colors,
    attr_tags_html,
    load_resources,
    run_pipeline,
    score_label,
    shooting_tips,
)

__all__ = [
    "ATTRIBUTE_IMPACT", "CLUSTER_DESCRIPTIONS", "CLUSTER_NAMES", "PHOTO_GENRES", "TAG_COLORS",
    "VISUAL_ATTRIBUTES", "_analyse_image_clip", "_cluster_attrs", "_dominant_colors",
    "_score_image", "attr_tags_html", "load_all", "score_label", "shooting_tips",
]


@st.cache_resource
def load_all():
    """Load models/data once per Streamlit session and return the flat tuple the pages expect."""
    r = load_resources()
    return (r.clip, r.processor, r.scorer, r.classifier, r.device,
            r.training_preds, r.embeddings, r.df, r.umap_3d, r.cluster_centroids)


def _score_image(img, clip, processor, scorer, classifier, device,
                 training_preds, embeddings, df, umap_3d, cluster_centroids):
    """Backward-compatible wrapper around scorer.run_pipeline for the flat-tuple interface."""
    resources = Resources(
        clip=clip, processor=processor, scorer=scorer, classifier=classifier, device=device,
        training_preds=training_preds, embeddings=embeddings, df=df,
        umap_3d=umap_3d, cluster_centroids=cluster_centroids,
    )
    return run_pipeline(img, resources)
