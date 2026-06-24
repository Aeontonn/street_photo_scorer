"""Street Photo Scorer — main page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from src.app.shared import (
    ATTRIBUTE_IMPACT, CLUSTER_DESCRIPTIONS, CLUSTER_NAMES, TAG_COLORS,
    _analyse_image_clip, _cluster_attrs, _dominant_colors, _score_image,
    attr_tags_html, load_all, score_label, shooting_tips,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Street Photo Scorer",
    layout="wide",
    page_icon="📷",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Layout ── */
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* ── Score display ── */
    .big-score   { font-size: 4rem; font-weight: 800; line-height: 1; }
    .score-label { font-size: 1.1rem; font-weight: 600; color: #f5f0e0; margin-top: 0.2rem; }

    /* ── Tags ── */
    .tag { display: inline-block; padding: 4px 12px; border-radius: 20px;
           font-size: 0.84rem; font-weight: 600; margin: 3px 3px 3px 0; }

    /* ── Info card ── */
    .info-card { background: #252a40; border-radius: 12px; padding: 1.2rem 1.4rem;
                 border-left: 5px solid #e8c87a; margin-bottom: 0.5rem; }
    .info-card * { color: #f5f0e0 !important; }

    /* ── Footer ── */
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #353a55;
              color: #9a98c0; font-size: 0.8rem; text-align: center; }

    /* ── Global text ── */
    .stMarkdown p, .stMarkdown li { color: #f5f0e0 !important; }
    label, .stMetric label { color: #d8d4f0 !important; }
    h1, h2, h3 { color: #f5f0e0 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #252a40 !important;
        border-radius: 12px !important;
        border: 2px dashed #e8c87a !important;
        padding: 0.5rem !important;
    }
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #f5f0e0 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #e8c87a !important;
        color: #1a1e2e !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #1a1e2e !important;
    }

    /* ── Container cards ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #252a40 !important;
        border: 1px solid #353a55 !important;
        border-radius: 12px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #10142a !important;
        border-radius: 10px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid #353a55 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9a98c0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        border-radius: 7px !important;
        transition: background 0.15s, color 0.15s !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #252a40 !important;
        color: #f5f0e0 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8c87a !important;
        color: #1a1e2e !important;
        font-weight: 700 !important;
        border-bottom: none !important;
    }

    /* ── Progress bars ── */
    [data-testid="stProgressBar"] > div {
        background-color: #252a40 !important;
        border-radius: 4px;
    }
    [data-testid="stProgressBar"] > div > div {
        background-color: #e8c87a !important;
        border-radius: 4px;
    }

    /* ── Expanders ── */
    details > summary {
        background-color: #252a40 !important;
        border-radius: 6px !important;
        color: #f5f0e0 !important;
        font-weight: 600 !important;
    }
    details[open] > summary { border-radius: 6px 6px 0 0 !important; }
    details > div { background-color: #252a40 !important; }

    /* ── Status / spinner ── */
    [data-testid="stStatusWidget"],
    [data-testid="stStatus"] {
        background-color: #252a40 !important;
        border: 1px solid #353a55 !important;
        border-radius: 8px !important;
    }

    /* ── Captions ── */
    .stCaption p { color: #9a98c0 !important; }

    /* ── Dividers ── */
    hr { border-color: #353a55 !important; opacity: 1; }

    /* ── Alerts / warnings ── */
    [data-testid="stAlert"] {
        background-color: #252a40 !important;
        border: 1px solid #e8c87a !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] p { color: #f5f0e0 !important; }
</style>
""", unsafe_allow_html=True)


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
                "Measures sharpness, exposure, noise, contrast, and rule of thirds "
                "using classical image analysis — completely separate from the AI score."
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

try:
    img_raw = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except Exception:
    st.error("Felaktig filtyp — ladda upp en JPEG, PNG eller WEBP och försök igen.")
    st.stop()

if max(img_raw.size) > 1200:
    img_raw.thumbnail((1200, 1200), Image.LANCZOS)
img = img_raw

try:
    with st.status("Analysing your photo…", expanded=True) as status:
        st.write("Loading AI models…")
        (clip, processor, scorer, classifier, device,
         training_preds, embeddings, df, umap_3d, cluster_centroids) = load_all()

        models = (clip, processor, scorer, classifier, device,
                  training_preds, embeddings, df, umap_3d, cluster_centroids)

        st.write("Extracting visual features and scoring…")
        r = _score_image(img, *models)

        status.update(label="Analysis complete ✓", state="complete", expanded=False)

except Exception as exc:
    st.error(
        f"**Something went wrong during analysis.**\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "Try uploading a different photo, or restart the app if the problem persists."
    )
    st.stop()

embedding       = r["embedding"]
attributes      = r["attributes"]
genre_scores    = r["genre_scores"]
percentile      = r["percentile"]
aesthetic_score = r["aesthetic_score"]
score_lbl       = r["score_lbl"]
score_icon      = r["score_icon"]
score_color     = r["score_color"]
clf_pred        = r["clf_pred"]
clf_proba       = r["clf_proba"]
similar_indices = r["similar_indices"]
cluster_id      = r["cluster_id"]
cluster_name    = r["cluster_name"]
umap_coords     = r["umap_coords"]
cluster_attrs   = r["cluster_attrs"]
tech            = r["tech"]
dominant_colors = r["dominant_colors"]


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
        verdict_color = "#7ed4a0" if clf_pred == 1 else "#f0a8a8"
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

    # ── Shooting tips ─────────────────────────────────────────────────────────
    tips = shooting_tips(attributes, percentile)
    if tips:
        st.markdown("")
        st.markdown("#### 📸 Shoot this")
        for tip in tips:
            st.markdown(
                f'<div class="info-card">{tip}</div>',
                unsafe_allow_html=True,
            )

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
    for i, idx in enumerate(similar_indices[:18]):
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

    # umap_3d already loaded in load_all() — reuse it

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
        x=[float(umap_coords[0])], y=[float(umap_coords[1])], z=[float(umap_coords[2])],
        mode="markers",
        marker=dict(size=14, color="#f0c060", symbol="diamond", opacity=1.0),
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
        "Classical image analysis (scipy/numpy) — completely separate from the AI score. "
        "Measures objective technical properties of the image."
    )

    if tech is None:
        st.warning(
            "Technical analysis could not be completed for this image. "
            "Try a different photo or a smaller file."
        )
        st.stop()

    col_ann, col_metrics = st.columns([1, 1], gap="large")

    with col_ann:
        st.markdown("**Composition overlay**")
        st.image(tech.annotated_image, use_container_width=True)
        st.caption(
            "White grid = rule of thirds · "
            "Cyan dot = brightest/salient point in the frame"
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
        st.markdown(f"{rot_icon} **Rule of thirds** — {comp.rule_of_thirds_label}")
        st.progress(comp.rule_of_thirds_score / 10)
        st.caption(
            f"Brightest/most salient point estimated at "
            f"{comp.subject_x*100:.0f}% across / {comp.subject_y*100:.0f}% down the frame. "
            "Subjects near the four intersection points score highest."
        )

        st.markdown("➖ **Leading lines** — not measured")
        st.caption(
            "Diagonal lines that guide the viewer's eye into the scene "
            "(staircases, alleys, railway tracks). Detection not available in this build."
        )

        st.markdown("➖ **Horizon** — not measured")
        st.caption("Horizon levelness detection requires the Hough line transform, not available in the current build.")

    # ── Photography style match ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Photography style")
    st.caption(
        "The AI compares your photo's visual fingerprint against descriptions of real photography genres. "
        "The tallest bar is the style your photo most closely resembles."
    )

    genre_labels = [l for l, _ in genre_scores]
    genre_vals   = [v for _, v in genre_scores]
    max_g = max(genre_vals) if max(genre_vals) > 0 else 1
    genre_norm = [v / max_g for v in genre_vals]

    genre_colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880"]

    genre_fig = go.Figure(go.Bar(
        x=genre_labels,
        y=genre_norm,
        marker=dict(color=genre_colors[:len(genre_labels)]),
        text=[f"{v*100:.0f}%" for v in genre_norm],
        textposition="outside",
        cliponaxis=False,
    ))
    genre_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(visible=False, range=[0, 1.3]),
        xaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(genre_fig, use_container_width=True)

    top_genre = genre_labels[0]
    st.info(f"**Best match: {top_genre}** — this is the photography style your image most resembles according to the AI.")

    # ── Colour palette + tonal balance ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visual character")

    pal_col, tone_col = st.columns([1, 1], gap="large")

    with pal_col:
        st.markdown("**Dominant colour palette**")
        st.caption("The most common colours extracted from your photo.")
        swatches_html = ""
        for r, g, b in dominant_colors:
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            # Pick readable text colour based on luminance
            lum = 0.299*r + 0.587*g + 0.114*b
            text_col = "#000000" if lum > 140 else "#ffffff"
            swatches_html += (
                f'<div style="display:inline-block;width:80px;height:60px;'
                f'background:{hex_col};border-radius:8px;margin:4px;'
                f'text-align:center;line-height:60px;font-size:0.75rem;'
                f'font-weight:600;color:{text_col};">{hex_col}</div>'
            )
        st.markdown(swatches_html, unsafe_allow_html=True)

    with tone_col:
        st.markdown("**Tonal balance**")
        st.caption("How the brightness is distributed across the image — from shadows to highlights.")
        img_arr = np.array(img.convert("L"))
        shadows    = float((img_arr < 85).sum()  / img_arr.size)
        midtones   = float(((img_arr >= 85) & (img_arr < 170)).sum() / img_arr.size)
        highlights = float((img_arr >= 170).sum() / img_arr.size)

        tone_fig = go.Figure(go.Bar(
            x=["Shadows", "Midtones", "Highlights"],
            y=[shadows, midtones, highlights],
            marker=dict(color=["#3d5046", "#7ea090", "#c8d8cc"],
                        line=dict(color="#3d5046", width=1)),
            text=[f"{v*100:.0f}%" for v in [shadows, midtones, highlights]],
            textposition="outside",
            cliponaxis=False,
        ))
        tone_fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(visible=False, range=[0, 1.2]),
            xaxis=dict(tickfont=dict(size=13)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(tone_fig, use_container_width=True)

        dominant_tone = max(
            [("shadow-heavy", shadows), ("balanced", midtones), ("high-key", highlights)],
            key=lambda x: x[1]
        )[0]
        tone_notes = {
            "shadow-heavy": "Dark, moody image — heavy in shadows. Typical of night scenes and high-contrast work.",
            "balanced":     "Well-balanced tonal range across shadows, midtones, and highlights.",
            "high-key":     "Bright, light image — dominated by highlights. Can feel airy or overexposed.",
        }
        st.caption(tone_notes[dominant_tone])

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">Street Photo Scorer · '
        'Trained on r/streetphotography · '
        'CLIP + Gradient Boosting + scipy · '
        'Built as an ML course project</div>',
        unsafe_allow_html=True,
    )
