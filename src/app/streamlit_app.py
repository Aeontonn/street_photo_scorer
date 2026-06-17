"""Streamlit demo: upload a street photo and get an aesthetic score."""

from pathlib import Path
from scipy import stats

import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

MODELS_DIR = Path("data/models")
MODEL_NAME = "openai/clip-vit-base-patch32"

VISUAL_ATTRIBUTES = [
    ("a photo with people and human subjects", "People present"),
    ("a photo without people, only architecture or landscape", "No people"),
    ("a black and white photo", "Black & white"),
    ("a colour photo", "Colour"),
    ("a photo taken at night or in low light", "Low light / night"),
    ("a photo taken in bright daylight", "Bright daylight"),
    ("strong contrast and shadows", "High contrast"),
    ("soft light and muted tones", "Soft / muted"),
    ("motion blur and movement", "Motion blur"),
    ("sharp and in focus", "Sharp focus"),
    ("a wide landscape or street scene", "Wide scene"),
    ("a tight portrait or close-up", "Close-up / portrait"),
]

st.set_page_config(page_title="Street Photo Scorer", layout="wide")

st.title("Street Photo Scorer")
st.markdown(
    "Trained on **1434 photos** from r/streetphotography. "
    "Score reflects how similar your photo is to highly upvoted community posts — "
    "not universal aesthetic quality."
)
st.divider()


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    scorer = joblib.load(MODELS_DIR / "scorer.pkl")
    training_preds = np.load(MODELS_DIR / "training_predictions.npy")
    return clip, processor, scorer, device, training_preds


def get_visual_attributes(img: Image.Image, clip, processor, device) -> list[tuple[str, float]]:
    texts = [t for t, _ in VISUAL_ATTRIBUTES]
    labels = [l for _, l in VISUAL_ATTRIBUTES]

    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    img_inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        img_feats = clip.get_image_features(**img_inputs)
        if not isinstance(img_feats, torch.Tensor):
            img_feats = img_feats.pooler_output
        if not isinstance(text_feats, torch.Tensor):
            text_feats = text_feats.pooler_output

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        similarities = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()

    # Pair up opposing attributes and keep only the strongest from each pair
    pairs = [
        (0, 1),   # people vs no people
        (2, 3),   # b&w vs colour
        (4, 5),   # night vs daylight
        (6, 7),   # high contrast vs soft
        (8, 9),   # motion blur vs sharp
        (10, 11), # wide vs close-up
    ]
    results = []
    for i, j in pairs:
        if similarities[i] > similarities[j]:
            results.append((labels[i], float(similarities[i])))
        else:
            results.append((labels[j], float(similarities[j])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_label(percentile: float) -> tuple[str, str]:
    if percentile >= 90:
        return "Exceptional", "🏆"
    elif percentile >= 75:
        return "Strong", "⭐"
    elif percentile >= 50:
        return "Above average", "👍"
    elif percentile >= 25:
        return "Below average", "📷"
    else:
        return "Low match", "🔍"


col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Upload your photo")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded:
        img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        st.image(img, use_container_width=True)

with col_result:
    st.subheader("Score")
    if not uploaded:
        st.info("Upload a photo on the left to get your score.")
    else:
        with st.spinner("Analysing with CLIP..."):
            clip, processor, scorer, device, training_preds = load_models()

            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = clip.get_image_features(**inputs)
                if not isinstance(feats, torch.Tensor):
                    feats = feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embedding = feats.cpu().numpy()

            raw_score = scorer.predict(embedding)[0]
            percentile = float(stats.percentileofscore(training_preds, raw_score))
            aesthetic_score = round(percentile / 10, 1)
            label, icon = score_label(percentile)

            attributes = get_visual_attributes(img, clip, processor, device)

        st.metric("Aesthetic Score", f"{aesthetic_score:.1f} / 10", label)
        st.progress(percentile / 100)
        st.markdown(f"**{icon} {label}** — better than **{percentile:.0f}%** of {len(training_preds)} training photos.")

        st.divider()
        st.markdown("#### What CLIP sees in your photo")

        # Based on training data: what this community tends to upvote
        ATTRIBUTE_IMPACT = {
            "People present":       ("positive", "Community strongly prefers photos with human subjects"),
            "No people":            ("negative", "Architecture/street scenes without people score lower on this subreddit"),
            "Black & white":        ("positive", "B&W is a popular and well-received style in street photography"),
            "Colour":               ("neutral",  "Colour photos are common — impact depends on palette and mood"),
            "Low light / night":    ("positive", "Dramatic low-light scenes tend to perform well"),
            "Bright daylight":      ("neutral",  "Daylight is common — strong shadows help"),
            "High contrast":        ("positive", "Strong contrast and shadows are a hallmark of street photography"),
            "Soft / muted":         ("neutral",  "Soft tones can work well but are less distinctive"),
            "Motion blur":          ("positive", "Motion blur adds energy and a sense of life to the scene"),
            "Sharp focus":          ("neutral",  "Sharp focus is standard — decisive moment matters more"),
            "Wide scene":           ("neutral",  "Wide shots work well when the environment tells a story"),
            "Close-up / portrait":  ("positive", "Tight candid portraits tend to get strong engagement"),
        }

        icons = {"positive": "✅", "negative": "⚠️", "neutral": "➖"}

        for attr_label, sim in attributes:
            impact, reason = ATTRIBUTE_IMPACT.get(attr_label, ("neutral", ""))
            icon = icons[impact]
            st.markdown(f"{icon} **{attr_label}** — {reason}")

        st.divider()
        st.markdown("#### How it works")
        st.markdown(
            "1. Your photo is processed by **CLIP** (OpenAI) into a 512-dimensional embedding\n"
            "2. A **Gradient Boosting** model predicts an aesthetic score based on patterns learned from Reddit upvotes\n"
            "3. The score is **percentile-ranked** against all training photos for a fair comparison"
        )

        st.divider()
        st.markdown("#### Known limitations")
        st.markdown(
            "- Photos **with people** score higher on average — the community prefers candid shots\n"
            "- Famous historical photos may score low — the model reflects 2021–2024 Reddit taste, not universal aesthetics\n"
            "- Score ≠ quality. It measures similarity to what this community upvotes."
        )
