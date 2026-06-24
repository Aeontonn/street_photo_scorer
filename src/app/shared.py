"""Shared ML logic — imported by both streamlit_app.py and pages/."""

from __future__ import annotations
from pathlib import Path
from scipy import stats
import gc
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.analysis.photo_analysis import safe_analyze

MODELS_DIR    = Path("data/models")
PROCESSED_DIR = Path("data/processed")
MODEL_NAME    = "openai/clip-vit-base-patch32"

CLUSTER_NAMES: dict[int, str] = {
    0: "Style group A", 1: "Style group B", 2: "Style group C", 3: "Style group D",
    4: "Style group E", 5: "Style group F", 6: "Style group G", 7: "Style group H",
}
CLUSTER_DESCRIPTIONS: dict[int, str] = {}

VISUAL_ATTRIBUTES = [
    ("a photo with people and human subjects",                    "People present"),
    ("a photo without people, only architecture or landscape",    "No people"),
    ("a black and white photo",                                   "Black & white"),
    ("a colour photo",                                            "Colour"),
    ("a photo taken at night or in low light",                    "Low light / night"),
    ("a photo taken in bright daylight",                          "Bright daylight"),
    ("strong contrast and shadows",                               "High contrast"),
    ("soft light and muted tones",                                "Soft / muted"),
    ("motion blur and movement",                                  "Motion blur"),
    ("sharp and in focus",                                        "Sharp focus"),
    ("a wide landscape or street scene",                          "Wide scene"),
    ("a tight portrait or close-up",                              "Close-up / portrait"),
]

ATTRIBUTE_IMPACT = {
    "People present":      ("positive", "Community strongly prefers photos with human subjects"),
    "No people":           ("negative", "Architecture/street scenes without people score lower on this subreddit"),
    "Black & white":       ("positive", "B&W is a popular and well-received style in street photography"),
    "Colour":              ("neutral",  "Colour photos are common — impact depends on palette and mood"),
    "Low light / night":   ("positive", "Dramatic low-light scenes tend to perform well"),
    "Bright daylight":     ("neutral",  "Daylight is common — strong shadows help"),
    "High contrast":       ("positive", "Strong contrast and shadows are a hallmark of street photography"),
    "Soft / muted":        ("neutral",  "Soft tones can work well but are less distinctive"),
    "Motion blur":         ("positive", "Motion blur adds energy and a sense of life to the scene"),
    "Sharp focus":         ("neutral",  "Sharp focus is standard — decisive moment matters more"),
    "Wide scene":          ("neutral",  "Wide shots work well when the environment tells a story"),
    "Close-up / portrait": ("positive", "Tight candid portraits tend to get strong engagement"),
}

TAG_COLORS = {
    "positive": ("1a1e2e", "e8c87a"),
    "negative": ("1a1e2e", "e89090"),
    "neutral":  ("f5f0e0", "353a55"),
}

PHOTO_GENRES = [
    ("Candid street photography — people in decisive moments",  "Candid / decisive moment"),
    ("Fine art black and white photography",                    "Fine art B&W"),
    ("Urban landscape and architectural photography",           "Urban landscape"),
    ("Documentary and photojournalism photography",             "Documentary"),
    ("Long exposure and light trail photography",               "Long exposure"),
    ("Moody atmospheric night photography",                     "Night / atmosphere"),
    ("Minimalist photography with negative space",              "Minimalist"),
    ("Colourful vibrant street photography",                    "Colourful & vibrant"),
]

_TIPS: list[tuple] = [
    (
        lambda a, p: "No people" in a and p < 60,
        "💡 **Add a human element.** Photos with people consistently score higher on r/streetphotography — even a silhouette or a passing figure transforms an architectural shot."
    ),
    (
        lambda a, p: "Colour" in a and "High contrast" not in a and p < 60,
        "💡 **Push your contrast.** Colour street photos benefit from strong shadows and punchy tones — try increasing contrast in post or shooting during golden hour for harsher light."
    ),
    (
        lambda a, p: "Black & white" in a and "High contrast" not in a,
        "💡 **Lean into the shadows.** Your B&W image reads as soft — classic street photography B&W relies on deep blacks and bright highlights. Try pushing the contrast further in editing."
    ),
    (
        lambda a, p: "Bright daylight" in a and "High contrast" not in a and p < 50,
        "💡 **Use harsh midday light to your advantage.** Flat daylight is the enemy of street photography — look for strong shadows cast by buildings, or shoot subjects backlit against bright sky."
    ),
    (
        lambda a, p: "Low light / night" in a and "Sharp focus" not in a,
        "💡 **Nail the sharpness.** Low-light shots are popular but lose impact when they're soft — try a wider aperture, bump ISO, or brace against a wall to keep your subject crisp."
    ),
    (
        lambda a, p: "Wide scene" in a and "People present" not in a and p < 50,
        "💡 **Give your wide shot a focal point.** Empty streets read as landscapes — wait for a person to enter the frame, or find a strong geometric element to anchor the composition."
    ),
    (
        lambda a, p: "Motion blur" in a and p < 40,
        "💡 **Make the blur intentional.** Motion blur works best when one element is sharp and another blurred — try panning with a moving subject while keeping the background static."
    ),
    (
        lambda a, p: "Close-up / portrait" in a and p < 50,
        "💡 **Capture the decisive moment.** Tight portraits need a strong expression or gesture — the technical quality is there, but the most upvoted candids freeze a fleeting, unrepeatable instant."
    ),
    (
        lambda a, p: p >= 75,
        "✨ **Strong shot.** This photo matches the visual style of the community's top-performing images — composition, light, and subject all working together."
    ),
    (
        lambda a, p: 40 <= p < 60,
        "💡 **Almost there.** The shot has potential but something is holding it back — look at whether the subject is clearly defined and whether the light creates enough drama."
    ),
]


def shooting_tips(attributes: list, percentile: float) -> list[str]:
    active = {label for label, _ in attributes}
    return [tip for cond, tip in _TIPS if cond(active, percentile)]


def score_label(percentile: float) -> tuple[str, str, str]:
    if percentile >= 90: return "Exceptional",   "🏆", "#e8c87a"
    if percentile >= 75: return "Strong",         "⭐", "#a8d8c0"
    if percentile >= 50: return "Above average",  "👍", "#a0b8e8"
    if percentile >= 25: return "Below average",  "📷", "#c8c4d8"
    return               "Low match",             "🔍", "#9a98c0"


def attr_tags_html(attributes: list) -> str:
    parts = []
    for attr_label, _ in attributes:
        impact, _ = ATTRIBUTE_IMPACT.get(attr_label, ("neutral", ""))
        fg, bg = TAG_COLORS[impact]
        parts.append(
            f'<span class="tag" style="color:#{fg};background:#{bg};">{attr_label}</span>'
        )
    return " ".join(parts)


def _dominant_colors(img_pil: Image.Image, n: int = 6) -> list[tuple[int, int, int]]:
    small = img_pil.convert("RGB").resize((100, 100), Image.LANCZOS)
    quantized = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:n * 3]
    counts: dict[int, int] = {}
    for px in quantized.getdata():
        counts[px] = counts.get(px, 0) + 1
    sorted_idx = sorted(counts, key=lambda k: counts[k], reverse=True)[:n]
    return [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) for i in sorted_idx]


def _analyse_image_clip(img, clip, processor, device):
    attr_texts  = [t for t, _ in VISUAL_ATTRIBUTES]
    attr_labels = [l for _, l in VISUAL_ATTRIBUTES]
    genre_texts  = [t for t, _ in PHOTO_GENRES]
    genre_labels = [l for _, l in PHOTO_GENRES]

    all_texts   = attr_texts + genre_texts
    text_inputs = processor(text=all_texts, return_tensors="pt",
                            padding=True, truncation=True).to(device)
    img_inputs  = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        img_feats  = clip.get_image_features(**img_inputs)
        if not isinstance(img_feats,  torch.Tensor): img_feats  = img_feats.pooler_output
        if not isinstance(text_feats, torch.Tensor): text_feats = text_feats.pooler_output
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats  = img_feats  / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()

    embedding = img_feats.cpu().numpy()
    attr_sims = sims[:len(attr_texts)]
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    attributes = []
    for i, j in pairs:
        attributes.append(
            (attr_labels[i], float(attr_sims[i])) if attr_sims[i] > attr_sims[j]
            else (attr_labels[j], float(attr_sims[j]))
        )
    attributes.sort(key=lambda x: x[1], reverse=True)

    genre_sims = sims[len(attr_texts):]
    exp_g = np.exp(genre_sims * 20)
    genre_probs = exp_g / exp_g.sum()
    genre_scores = sorted(zip(genre_labels, genre_probs.tolist()),
                          key=lambda x: x[1], reverse=True)

    return embedding, attributes, genre_scores


def _cluster_attrs(cluster_centroid_np, clip, processor, device):
    texts  = [t for t, _ in VISUAL_ATTRIBUTES]
    labels = [l for _, l in VISUAL_ATTRIBUTES]
    text_inputs = processor(text=texts, return_tensors="pt",
                            padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        if not isinstance(text_feats, torch.Tensor): text_feats = text_feats.pooler_output
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    img_feats = torch.tensor(cluster_centroid_np, dtype=torch.float32).to(device)
    sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    results = []
    for i, j in pairs:
        results.append((labels[i], float(sims[i])) if sims[i] > sims[j]
                       else (labels[j], float(sims[j])))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


@st.cache_resource
def load_all():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    clip      = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()

    scorer         = joblib.load(MODELS_DIR / "scorer.pkl")
    classifier     = joblib.load(MODELS_DIR / "classifier.pkl")
    training_preds = np.load(MODELS_DIR / "training_predictions.npy")

    embeddings = np.load(PROCESSED_DIR / "embeddings.npy").astype(np.float16)
    df         = pd.read_csv(PROCESSED_DIR / "clustered.csv")
    umap_3d    = np.load(PROCESSED_DIR / "umap_3d.npy").astype(np.float16)

    n_clusters = int(df["cluster"].max()) + 1
    cluster_centroids = np.stack([
        embeddings[df["cluster"].values == c].mean(axis=0)
        for c in range(n_clusters)
    ]).astype(np.float32)
    cluster_centroids /= np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-8

    return (clip, processor, scorer, classifier, device,
            training_preds, embeddings, df, umap_3d, cluster_centroids)


def _score_image(img, clip, processor, scorer, classifier, device,
                 training_preds, embeddings, df, umap_3d, cluster_centroids):
    embedding, attributes, genre_scores = _analyse_image_clip(img, clip, processor, device)

    raw_score       = scorer.predict(embedding)[0]
    percentile      = float(stats.percentileofscore(training_preds, raw_score))
    aesthetic_score = round(percentile / 10, 1)
    score_lbl, score_icon, score_color = score_label(percentile)
    clf_pred  = classifier.predict(embedding)[0]
    clf_proba = classifier.predict_proba(embedding)[0][1]

    emb_f32  = embedding.squeeze().astype(np.float32)
    emb_f32 /= np.linalg.norm(emb_f32) + 1e-8
    chunk    = 2000
    cos_sims = np.empty(len(embeddings), dtype=np.float32)
    for start in range(0, len(embeddings), chunk):
        block = embeddings[start:start+chunk].astype(np.float32)
        cos_sims[start:start+chunk] = block @ emb_f32
    top_idx         = np.argsort(cos_sims)[::-1][:19]
    similar_indices = top_idx[1:]

    emb_norm     = embedding / (np.linalg.norm(embedding) + 1e-8)
    sims_c       = (cluster_centroids @ emb_norm.T).squeeze()
    cluster_id   = int(np.argmax(sims_c))
    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Style group {cluster_id}")

    nearest_idx      = int(top_idx[0])
    umap_coords      = umap_3d[nearest_idx]
    cluster_centroid = cluster_centroids[cluster_id : cluster_id + 1]
    cluster_attrs    = _cluster_attrs(cluster_centroid, clip, processor, device)

    gc.collect()
    img_thumb = img.copy()
    img_thumb.thumbnail((800, 800), Image.LANCZOS)
    tech            = safe_analyze(img_thumb)
    dominant_colors = _dominant_colors(img_thumb)

    return dict(
        embedding=embedding, attributes=attributes, genre_scores=genre_scores,
        raw_score=raw_score, percentile=percentile, aesthetic_score=aesthetic_score,
        score_lbl=score_lbl, score_icon=score_icon, score_color=score_color,
        clf_pred=clf_pred, clf_proba=clf_proba,
        similar_indices=similar_indices, cos_sims=cos_sims,
        cluster_id=cluster_id, cluster_name=cluster_name,
        umap_coords=umap_coords, cluster_attrs=cluster_attrs,
        tech=tech, dominant_colors=dominant_colors,
    )
