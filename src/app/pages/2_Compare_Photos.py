"""Compare two street photos side by side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from scipy import stats
import gc
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.app.shared import (
    load_all,
    _score_image,
    _analyse_image_clip,
    attr_tags_html,
    shooting_tips,
    ATTRIBUTE_IMPACT,
    CLUSTER_NAMES,
    TAG_COLORS,
)

st.set_page_config(page_title="Compare Photos", layout="wide", page_icon="⚖️")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .big-score   { font-size: 3rem; font-weight: 800; line-height: 1; }
    .score-label { font-size: 1rem; font-weight: 600; color: #f5f0e0; margin-top: 0.2rem; }
    .tag { display: inline-block; padding: 4px 12px; border-radius: 20px;
           font-size: 0.84rem; font-weight: 600; margin: 3px 3px 3px 0; }
    .info-card { background: #252a40; border-radius: 12px; padding: 1rem 1.2rem;
                 border-left: 5px solid #e8c87a; margin-bottom: 0.5rem; }
    .info-card * { color: #f5f0e0 !important; }

    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #252a40 !important;
        border-radius: 12px !important;
        border: 2px dashed #e8c87a !important;
    }
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] * { color: #f5f0e0 !important; }
    [data-testid="stFileUploader"] button {
        background-color: #e8c87a !important; color: #1a1e2e !important;
        border: none !important; border-radius: 6px !important; font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button * { color: #1a1e2e !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #252a40 !important;
        border: 1px solid #353a55 !important; border-radius: 12px !important;
    }
    hr { border-color: #353a55 !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Compare Photos")
st.markdown("Upload two photos to see which one scores higher and what makes them different.")

col1, col2 = st.columns(2, gap="large")
with col1:
    uploaded_a = st.file_uploader("Photo A", type=["jpg", "jpeg", "png", "webp"], key="cmp_a")
with col2:
    uploaded_b = st.file_uploader("Photo B", type=["jpg", "jpeg", "png", "webp"], key="cmp_b")

if not uploaded_a or not uploaded_b:
    st.info("Upload both photos above to start the comparison.")
    st.stop()


def _open(upload):
    try:
        img = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
    except Exception:
        st.error("Felaktig filtyp — ladda upp en JPEG, PNG eller WEBP.")
        st.stop()
    if max(img.size) > 1200:
        img.thumbnail((1200, 1200), Image.LANCZOS)
    return img


img_a = _open(uploaded_a)
img_b = _open(uploaded_b)

try:
    with st.status("Analysing both photos…", expanded=True) as status:
        st.write("Loading AI models…")
        models = load_all()
        st.write("Scoring Photo A…")
        r_a = _score_image(img_a, *models)
        st.write("Scoring Photo B…")
        r_b = _score_image(img_b, *models)
        status.update(label="Done ✓", state="complete", expanded=False)
except Exception as exc:
    st.error(f"Something went wrong: `{type(exc).__name__}: {exc}`")
    st.stop()

# ── Winner banner ─────────────────────────────────────────────────────────────

diff   = abs(r_a["aesthetic_score"] - r_b["aesthetic_score"])
winner = "A" if r_a["percentile"] >= r_b["percentile"] else "B"

st.markdown("---")
if diff < 0.3:
    st.info("**Too close to call** — both photos score almost identically.")
else:
    win_r = r_a if winner == "A" else r_b
    st.success(
        f"**Photo {winner} wins** — {diff:.1f} points higher "
        f"({win_r['score_lbl']}, better than {win_r['percentile']:.0f}% of the dataset)."
    )

# ── Side-by-side results ──────────────────────────────────────────────────────

col_a, col_b = st.columns(2, gap="large")

for col, res, photo, lbl in [(col_a, r_a, img_a, "A"), (col_b, r_b, img_b, "B")]:
    with col:
        is_winner = winner == lbl and diff >= 0.3
        border = "3px solid #e8c87a" if is_winner else "1px solid #353a55"
        st.markdown(
            f'<div style="border:{border};border-radius:12px;padding:4px;">',
            unsafe_allow_html=True,
        )
        st.image(photo, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="big-score" style="color:{res["score_color"]};">'
            f'{res["aesthetic_score"]}</div>'
            f'<div class="score-label">out of 10 · {res["score_lbl"]}</div>',
            unsafe_allow_html=True,
        )
        st.progress(res["percentile"] / 100)
        st.caption(f"Better than {res['percentile']:.0f}% of training photos")

        verdict = "High quality" if res["clf_pred"] == 1 else "Low quality"
        conf = res["clf_proba"] if res["clf_pred"] == 1 else 1 - res["clf_proba"]
        verdict_color = "#a8d8c0" if res["clf_pred"] == 1 else "#e8a0a0"
        st.markdown(
            f'**AI verdict:** <span style="color:{verdict_color};font-weight:600;">'
            f'{verdict}</span> ({conf*100:.0f}% confident)',
            unsafe_allow_html=True,
        )

        st.markdown("**What the AI sees:**")
        st.markdown(attr_tags_html(res["attributes"]), unsafe_allow_html=True)

        tips = shooting_tips(res["attributes"], res["percentile"])
        if tips:
            st.markdown("**📸 Shoot this:**")
            for tip in tips[:2]:
                st.markdown(
                    f'<div class="info-card" style="font-size:0.88rem;">{tip}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f"**Style:** {res['cluster_name']}")

# ── Attribute diff ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### What's different between the two photos")

attrs_a = {lbl for lbl, _ in r_a["attributes"]}
attrs_b = {lbl for lbl, _ in r_b["attributes"]}
only_a  = attrs_a - attrs_b
only_b  = attrs_b - attrs_a
shared  = attrs_a & attrs_b

impact_icon = {"positive": "✅", "negative": "⚠️", "neutral": "➖"}

d1, d2, d3 = st.columns(3)
with d1:
    st.markdown("**Only in Photo A**")
    for a in sorted(only_a):
        impact, _ = ATTRIBUTE_IMPACT.get(a, ("neutral", ""))
        st.markdown(f"{impact_icon[impact]} {a}")
    if not only_a:
        st.caption("—")
with d2:
    st.markdown("**Only in Photo B**")
    for a in sorted(only_b):
        impact, _ = ATTRIBUTE_IMPACT.get(a, ("neutral", ""))
        st.markdown(f"{impact_icon[impact]} {a}")
    if not only_b:
        st.caption("—")
with d3:
    st.markdown("**Shared traits**")
    for a in sorted(shared):
        st.markdown(f"➖ {a}")
    if not shared:
        st.caption("—")
