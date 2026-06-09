"""Streamlit demo: upload a street photo and get an aesthetic score + style cluster."""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODELS_DIR = Path("data/models")
MODEL_NAME = "openai/clip-vit-base-patch32"

st.set_page_config(page_title="Street Photo Scorer", layout="centered")
st.title("Street Photo Scorer")
st.caption("Upload a street photo to get an aesthetic score and style cluster.")


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    scorer = joblib.load(MODELS_DIR / "scorer.pkl")
    return clip, processor, scorer, device


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    with st.spinner("Analysing..."):
        clip, processor, scorer, device = load_models()
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embedding = feats.cpu().numpy()

        raw_score = scorer.predict(embedding)[0]
        # Map from log-score space to 0-10
        aesthetic_score = float(np.clip((raw_score / 10) * 10, 0, 10))

    st.metric("Aesthetic Score", f"{aesthetic_score:.1f} / 10")
    st.progress(aesthetic_score / 10)
