"""Street Photo Scorer — full Streamlit app."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from scipy import stats

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPModel, CLIPProcessor

from src.analysis.photo_analysis import analyze, STORYTELLING_PROMPTS

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "openai/clip-vit-base-patch32"

CLUSTER_NAMES = {
    0: "Style group A", 1: "Style group B", 2: "Style group C", 3: "Style group D",
    4: "Style group E", 5: "Style group F", 6: "Style group G", 7: "Style group H",
}

# Fill these in after inspecting notebook 02 — shown under cluster name in the UI.
CLUSTER_DESCRIPTIONS: dict[int, str] = {}

VISUAL_ATTRIBUTES = [
    ("a photo with people and human subjects",          "People present"),
    ("a photo without people, only architecture or landscape", "No people"),
    ("a black and white photo",                        "Black & white"),
    ("a colour photo",                                 "Colour"),
    ("a photo taken at night or in low light",         "Low light / night"),
    ("a photo taken in bright daylight",               "Bright daylight"),
    ("strong contrast and shadows",                    "High contrast"),
    ("soft light and muted tones",                     "Soft / muted"),
    ("motion blur and movement",                       "Motion blur"),
    ("sharp and in focus",                             "Sharp focus"),
    ("a wide landscape or street scene",               "Wide scene"),
    ("a tight portrait or close-up",                   "Close-up / portrait"),
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
    "positive": ("1a7a4a", "d4edda"),
    "negative": ("842029", "f8d7da"),
    "neutral":  ("495057", "e9ecef"),
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Street Photo Scorer",
    layout="wide",
    page_icon="📷",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .big-score { font-size: 4rem; font-weight: 800; line-height: 1; }
    .score-label { font-size: 1.1rem; font-weight: 600; color: #666; margin-top: 0.2rem; }
    .tag { display: inline-block; padding: 3px 10px; border-radius: 20px;
           font-size: 0.82rem; font-weight: 500; margin: 3px 3px 3px 0; }
    .info-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem 1.4rem;
                 border-left: 4px solid #6c757d; margin-bottom: 0.5rem; }
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6;
              color: #999; font-size: 0.8rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_label(percentile: float) -> tuple[str, str, str]:
    """Returns (label, emoji, color)."""
    if percentile >= 90: return "Exceptional",   "🏆", "#d4a017"
    if percentile >= 75: return "Strong",         "⭐", "#2e7d32"
    if percentile >= 50: return "Above average",  "👍", "#1565c0"
    if percentile >= 25: return "Below average",  "📷", "#6c757d"
    return               "Low match",             "🔍", "#9e9e9e"


def attr_tags_html(attributes: list) -> str:
    parts = []
    for attr_label, _ in attributes:
        impact, _ = ATTRIBUTE_IMPACT.get(attr_label, ("neutral", ""))
        fg, bg = TAG_COLORS[impact]
        parts.append(
            f'<span class="tag" style="color:#{fg};background:#{bg};">'
            f'{attr_label}</span>'
        )
    return " ".join(parts)


def _clip_storytelling(img, clip, processor, device) -> list[tuple[str, float]]:
    texts  = [t for t, _ in STORYTELLING_PROMPTS]
    labels = [l for _, l in STORYTELLING_PROMPTS]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    img_inputs  = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        img_feats  = clip.get_image_features(**img_inputs)
        if not isinstance(img_feats,  torch.Tensor): img_feats  = img_feats.pooler_output
        if not isinstance(text_feats, torch.Tensor): text_feats = text_feats.pooler_output
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats  = img_feats  / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
    exp_sims = np.exp(sims * 10)
    probs = exp_sims / exp_sims.sum()
    return list(zip(labels, probs.tolist()))


def _visual_attrs_from_embedding(img_feats_np, clip, processor, device):
    texts  = [t for t, _ in VISUAL_ATTRIBUTES]
    labels = [l for _, l in VISUAL_ATTRIBUTES]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        if not isinstance(text_feats, torch.Tensor): text_feats = text_feats.pooler_output
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    img_feats = torch.tensor(img_feats_np, dtype=torch.float32).to(device)
    sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    results = []
    for i, j in pairs:
        results.append((labels[i], float(sims[i])) if sims[i] > sims[j] else (labels[j], float(sims[j])))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _visual_attrs_from_image(img, clip, processor, device):
    texts  = [t for t, _ in VISUAL_ATTRIBUTES]
    labels = [l for _, l in VISUAL_ATTRIBUTES]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    img_inputs  = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        img_feats  = clip.get_image_features(**img_inputs)
        if not isinstance(img_feats,  torch.Tensor): img_feats  = img_feats.pooler_output
        if not isinstance(text_feats, torch.Tensor): text_feats = text_feats.pooler_output
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats  = img_feats  / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    results = []
    for i, j in pairs:
        results.append((labels[i], float(sims[i])) if sims[i] > sims[j] else (labels[j], float(sims[j])))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def embed_image(img, clip, processor, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor): feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@st.cache_resource
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip      = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    scorer          = joblib.load(MODELS_DIR / "scorer.pkl")
    classifier      = joblib.load(MODELS_DIR / "classifier.pkl")
    training_preds  = np.load(MODELS_DIR / "training_predictions.npy")
    embeddings      = np.load(PROCESSED_DIR / "embeddings.npy")
    df              = pd.read_csv(PROCESSED_DIR / "clustered.csv")
    nn = NearestNeighbors(n_neighbors=7, metric="cosine")
    nn.fit(embeddings)
    umap_reducer = joblib.load(MODELS_DIR / "umap_reducer.pkl")
    kmeans       = joblib.load(MODELS_DIR / "kmeans.pkl")
    return clip, processor, scorer, classifier, device, training_preds, embeddings, df, nn, umap_reducer, kmeans


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📷 Street Photo Scorer")
st.markdown(
    "Upload a street photo to get an **aesthetic score**, find **visually similar photos**, "
    "and get a detailed **technical breakdown** of your image. "
    "Trained on photos from **r/streetphotography**."
)

# ── Landing info cards (shown before upload) ──────────────────────────────────

uploaded = st.file_uploader(
    "Upload a photo", type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("#### What does this app do?")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**🎯 Aesthetic Score**")
            st.markdown(
                "Rates your photo 0–10 based on how it compares to thousands of real "
                "street photos, using the same signals that drive upvotes on Reddit's "
                "street photography community."
            )
    with c2:
        with st.container(border=True):
            st.markdown("**🔍 Find Similar Photos**")
            st.markdown(
                "Finds the most visually similar photos from the training dataset "
                "using AI embeddings — useful for understanding what style your photo belongs to."
            )
    with c3:
        with st.container(border=True):
            st.markdown("**🔬 Technical Breakdown**")
            st.markdown(
                "Measures sharpness, exposure, noise, contrast, rule of thirds, leading lines, "
                "and horizon levelness using computer vision — completely separate from the AI score."
            )

    st.markdown("---")
    st.markdown("""
**How the score works:** The AI (CLIP) converts your photo into a numerical fingerprint.
A machine learning model trained on Reddit upvotes predicts how the street photography
community would likely receive it. The score is then percentile-ranked against all training
photos so 5/10 literally means better than 50% of photos in the dataset.

> ⚠️ The score reflects **community taste**, not universal photographic quality.
> A technically perfect landscape with no people may score lower than a blurry candid
> portrait — because that's what this community upvotes.
""")
    st.stop()


# ── Analysis ──────────────────────────────────────────────────────────────────

img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")

with st.status("Analysing your photo…", expanded=True) as status:
    st.write("Loading AI models…")
    clip, processor, scorer, classifier, device, training_preds, embeddings, df, nn, umap_reducer, kmeans = load_all()

    st.write("Extracting visual features…")
    embedding = embed_image(img, clip, processor, device)
    attributes = _visual_attrs_from_image(img, clip, processor, device)
    story_scores = _clip_storytelling(img, clip, processor, device)

    st.write("Computing aesthetic score…")
    raw_score      = scorer.predict(embedding)[0]
    percentile     = float(stats.percentileofscore(training_preds, raw_score))
    aesthetic_score = round(percentile / 10, 1)
    score_lbl, score_icon, score_color = score_label(percentile)
    clf_pred  = classifier.predict(embedding)[0]
    clf_proba = classifier.predict_proba(embedding)[0][1]

    st.write("Finding similar photos and style cluster…")
    _, indices       = nn.kneighbors(embedding)
    similar_indices  = indices[0][1:]
    umap_coords      = umap_reducer.transform(embedding)
    cluster_id       = int(kmeans.predict(umap_coords)[0])
    cluster_name     = CLUSTER_NAMES.get(cluster_id, f"Style group {cluster_id}")
    cluster_mask_arr = (df["cluster"] == cluster_id).values
    cluster_centroid = embeddings[cluster_mask_arr].mean(axis=0, keepdims=True)
    cluster_centroid /= np.linalg.norm(cluster_centroid) + 1e-8
    cluster_attrs    = _visual_attrs_from_embedding(cluster_centroid, clip, processor, device)

    st.write("Running technical analysis (sharpness, composition…)")
    tech = analyze(img)

    status.update(label="Analysis complete ✓", state="complete", expanded=False)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Score & Style",
    "🔍 Similar Photos",
    "🗺️ Visual Style Map",
    "🔬 Technical Analysis",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Score & Style
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # Top row: photo + score side by side
    col_photo, col_score = st.columns([1, 1], gap="large")

    with col_photo:
        st.image(img, use_container_width=True)

    with col_score:
        # Big score number
        st.markdown(
            f'<div class="big-score" style="color:{score_color};">{aesthetic_score}</div>'
            f'<div class="score-label">out of 10</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {score_icon} {score_lbl}")
        st.progress(percentile / 100)
        st.markdown(
            f"Better than **{percentile:.0f}%** of {len(training_preds):,} training photos."
        )

        st.markdown("---")

        # AI quality verdict (replaces "Classifier")
        verdict = "High quality" if clf_pred == 1 else "Low quality"
        conf    = clf_proba if clf_pred == 1 else 1 - clf_proba
        verdict_color = "#2e7d32" if clf_pred == 1 else "#c62828"
        st.markdown(
            f"**AI quality verdict:** "
            f'<span style="color:{verdict_color};font-weight:600;">{verdict}</span> '
            f"({conf*100:.0f}% confident)",
            unsafe_allow_html=True,
        )
        st.caption(
            "A separate AI classifier trained to distinguish high-upvote from low-upvote photos."
        )

        st.markdown("---")

        with st.expander("How the score is calculated"):
            st.markdown(
                "1. Your photo is converted to a **512-number fingerprint** by OpenAI's CLIP model\n"
                "2. A **Gradient Boosting** model predicts an expected upvote count\n"
                "3. A **Random Forest** classifier independently labels it high or low quality\n"
                "4. The raw score is **percentile-ranked** against all training photos "
                "so 5.0 = better than 50% of the dataset\n"
            )
        with st.expander("Known limitations"):
            st.markdown(
                "- Photos **with people** score higher — this community values candid human moments\n"
                "- Score ≠ artistic quality. It predicts community upvotes, not objective merit\n"
                "- Famous historical photos may score low if they don't match recent Reddit trends\n"
            )

    # ── Full-width: what the AI sees ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### What the AI sees in your photo")
    st.caption(
        "The AI compares your photo against pairs of opposite concepts "
        "(e.g. B&W vs colour, night vs daylight). "
        "Green tags boost your score, red tags lower it, grey tags are neutral."
    )
    st.markdown(attr_tags_html(attributes), unsafe_allow_html=True)

    st.markdown("")
    with st.expander("Detailed breakdown"):
        impact_icons = {"positive": "✅", "negative": "⚠️", "neutral": "➖"}
        for attr_label, _ in attributes:
            impact, reason = ATTRIBUTE_IMPACT.get(attr_label, ("neutral", ""))
            st.markdown(f"{impact_icons[impact]} **{attr_label}** — {reason}")

    # ── Full-width: style cluster ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🎨 Your visual style — {cluster_name}")

    cluster_mask_df = df["cluster"] == cluster_id
    cluster_size    = int(cluster_mask_df.sum())
    cluster_med     = float(df.loc[cluster_mask_df, "score"].median())

    manual_desc = CLUSTER_DESCRIPTIONS.get(cluster_id)
    if manual_desc:
        st.markdown(manual_desc)
    else:
        st.markdown(
            f"Your photo shares visual style with **{cluster_size} other photos** in this group "
            f"(median **{cluster_med:.0f} upvotes**). "
            "Style groups are discovered automatically by the AI — photos in the same group look "
            "similar in terms of lighting, composition, and mood."
        )

    top_tags = " · ".join(f"**{a}**" for a, _ in cluster_attrs[:3])
    st.caption(f"Typical traits of this group: {top_tags}")

    cluster_examples = df[cluster_mask_df].nlargest(3, "score")
    ex_cols = st.columns(3)
    for col, (_, ex_row) in zip(ex_cols, cluster_examples.iterrows()):
        with col:
            try:
                col.image(Image.open(ex_row["local_path"]), use_container_width=True)
                col.caption(f"{int(ex_row['score'])} upvotes")
            except Exception:
                col.caption("Image unavailable")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Similar Photos
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Most visually similar photos from the training dataset")
    st.caption(
        "Similarity is measured by comparing AI fingerprints (CLIP embeddings) "
        "using cosine distance — photos that look alike cluster close together."
    )

    cols = st.columns(3)
    for i, idx in enumerate(similar_indices[:6]):
        row = df.iloc[idx]
        with cols[i % 3]:
            try:
                st.image(Image.open(row["local_path"]), use_container_width=True)
                sim_pct = float(stats.percentileofscore(training_preds, training_preds[idx]))
                title   = str(row.get("title", ""))[:55]
                if title:
                    st.caption(f"**{title}**")
                st.caption(
                    f"Score {round(sim_pct/10,1)}/10 · "
                    f"{int(row['score'])} upvotes · "
                    f"{CLUSTER_NAMES.get(int(row['cluster']), '')}"
                )
            except Exception:
                st.caption("Image unavailable")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Visual Style Map
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Visual style map")
    st.caption(
        "Every dot is a training photo. The AI arranged them in 3D space so that "
        "visually similar photos sit close together. Dots are coloured by style group. "
        "**Your photo** is the large red diamond — rotate and zoom to explore."
    )

    umap_3d = np.load(PROCESSED_DIR / "umap_3d.npy")

    fig = go.Figure()
    colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880"]
    for c in range(8):
        mask = df["cluster"] == c
        fig.add_trace(go.Scatter3d(
            x=umap_3d[mask, 0], y=umap_3d[mask, 1], z=umap_3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=colors[c], opacity=0.6),
            name=CLUSTER_NAMES.get(c, f"Style group {c}"),
            text=df[mask]["title"].str[:50],
            hovertemplate="%{text}<br>%{customdata} upvotes<extra></extra>",
            customdata=df[mask]["score"],
        ))

    fig.add_trace(go.Scatter3d(
        x=[umap_coords[0, 0]], y=[umap_coords[0, 1]], z=[umap_coords[0, 2]],
        mode="markers",
        marker=dict(size=14, color="red", symbol="diamond", opacity=1.0),
        name="⭐ Your photo",
    ))

    fig.update_layout(
        height=620,
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title="", yaxis_title="", zaxis_title="",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Technical Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Technical & compositional analysis")
    st.caption(
        "This analysis uses classical computer vision (OpenCV), completely separate from the AI score. "
        "It measures objective technical properties of the image."
    )

    col_ann, col_metrics = st.columns([1, 1], gap="large")

    with col_ann:
        st.markdown("**Composition overlay**")
        st.image(tech.annotated_image, use_container_width=True)
        st.caption(
            "White grid = rule of thirds · "
            "Green/yellow dot = detected subject · "
            "Orange = leading lines · "
            "Cyan = detected horizon"
        )

    with col_metrics:
        st.markdown(f"**Overall technical score: {tech.overall_technical} / 10**")
        st.progress(tech.overall_technical / 10)
        st.markdown("")

        def _row(title, result):
            icon = "✅" if result.good else "⚠️"
            st.markdown(f"{icon} **{title}** — {result.label}")
            st.progress(result.score / 10)
            st.caption(result.detail)

        _row("Sharpness", tech.sharpness)
        _row("Exposure",  tech.exposure)
        _row("Noise",     tech.noise)
        _row("Contrast",  tech.contrast)

        comp = tech.composition
        st.markdown("---")
        st.markdown("**Composition**")

        rot_icon = "✅" if comp.rule_of_thirds_score >= 6 else "➖"
        subj_type = "face detected" if comp.subject_found else "brightness saliency"
        st.markdown(f"{rot_icon} **Rule of thirds** — {comp.rule_of_thirds_label}")
        st.progress(comp.rule_of_thirds_score / 10)
        st.caption(
            f"Main subject located via {subj_type} at "
            f"{comp.subject_x*100:.0f}% across / {comp.subject_y*100:.0f}% down the frame. "
            "Subjects near the four intersection points score highest."
        )

        ll_icon = "✅" if comp.leading_lines_score >= 5 else "➖"
        st.markdown(f"{ll_icon} **Leading lines** — {comp.leading_lines_count} detected")
        st.progress(comp.leading_lines_score / 10)
        st.caption(
            "Diagonal lines that guide the viewer's eye into the scene. "
            "Street alleys, staircases, and railway tracks are classic examples."
        )

        if comp.horizon_angle is not None:
            h_icon = "✅" if comp.horizon_score >= 7 else "⚠️"
            st.markdown(f"{h_icon} **Horizon level** — {comp.horizon_angle:.1f}° off")
            st.progress(comp.horizon_score / 10)
            st.caption(
                f"The dominant horizontal line is tilted {comp.horizon_angle:.1f}°. "
                + ("A slight tilt can add dynamism, but strong tilts are usually unintentional."
                   if comp.horizon_angle > 3 else "This looks level.")
            )
        else:
            st.markdown("➖ **Horizon** — no clear horizontal line detected")
            st.caption("Common in photos with complex scenes or no clear skyline.")

    # ── Creative assessment (replaces "CLIP storytelling") ───────────────────
    st.markdown("---")
    st.markdown("### Creative assessment")
    st.caption(
        "The AI scores your photo against creative concepts by comparing your photo's fingerprint "
        "to written descriptions. Scores are relative to each other, not absolute percentages."
    )

    positive_labels = [l for l, _ in STORYTELLING_PROMPTS if "(negative)" not in l]
    negative_labels = [l for l, _ in STORYTELLING_PROMPTS if "(negative)" in l]
    story_dict = dict(story_scores)

    pos_scores  = [story_dict.get(l, 0) for l in positive_labels]
    pos_display = [l.replace(" (negative)", "") for l in positive_labels]

    max_val     = max(pos_scores) if pos_scores and max(pos_scores) > 0 else 1
    norm_scores = [s / max_val for s in pos_scores]

    bar_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    # Sort highest → lowest for readability
    paired = sorted(zip(norm_scores, pos_display, bar_colors[:len(pos_display)]),
                    reverse=True)
    sorted_vals, sorted_labels, sorted_colors = zip(*paired) if paired else ([], [], [])

    bar_fig = go.Figure(go.Bar(
        x=list(sorted_vals),
        y=list(sorted_labels),
        orientation="h",
        marker=dict(color=list(sorted_colors)),
        text=[f"{v*100:.0f}%" for v in sorted_vals],
        textposition="outside",
        cliponaxis=False,
    ))
    bar_fig.update_layout(
        height=300,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1.25]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    radar_col, watch_col = st.columns([1, 1], gap="large")

    with radar_col:
        st.markdown("**Positive creative signals**")
        st.plotly_chart(bar_fig, use_container_width=True)

    with watch_col:
        st.markdown("**Signals to watch**")
        st.caption(
            "These concepts have a negative connotation — lower is better here."
        )
        st.markdown("")
        for label in negative_labels:
            clean = label.replace(" (negative)", "")
            prob  = story_dict.get(label, 0)
            icon  = "⚠️" if prob > 0.15 else "✅"
            level = "High — worth reviewing" if prob > 0.15 else "Low — not a concern"
            st.markdown(f"{icon} **{clean}**")
            st.progress(min(prob * 4, 1.0))
            st.caption(level)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">Street Photo Scorer · '
        'Trained on r/streetphotography · '
        'CLIP + Gradient Boosting + OpenCV · '
        'Built as an ML course project</div>',
        unsafe_allow_html=True,
    )
