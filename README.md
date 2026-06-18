# 📷 Street Photo Scorer

> Upload a street photo — get an aesthetic score, a technical breakdown, and see where your photo fits among thousands of real community-rated shots.

---

## What is this?

**Street Photo Scorer** is an AI-powered web app that analyses street photography. You upload a photo and it tells you:

- **How it would likely perform** in the r/streetphotography community (scored 0–10)
- **What visual style it has** — black & white, low light, candid portrait, etc.
- **Where it fits** among ~1,400 real street photos on an interactive style map
- **Which photos look most similar** to yours in the training dataset
- **Technical quality details** — sharpness, exposure, noise, rule of thirds, leading lines

The score is trained on real Reddit upvotes, so it reflects genuine community taste rather than a subjective opinion.

---

## Demo

![App screenshot placeholder](docs/screenshot.png)

```
Upload photo → AI analyses it → Score 0–10 + full breakdown
```

---

## How it works

### 1. Data collection
Around 1,400 photos were scraped from [r/streetphotography](https://reddit.com/r/streetphotography) using the [Arctic Shift](https://arctic-shift.photon-reddit.com) Reddit archive — no Reddit account needed. Upvote counts are used as a proxy for aesthetic quality.

### 2. AI visual fingerprinting (CLIP)
Each photo is passed through **CLIP** (OpenAI's vision model) which converts it into a 512-number fingerprint capturing its visual style, lighting, mood, and composition. Photos that look similar end up with similar fingerprints.

### 3. Style clustering (UMAP + KMeans)
The 512-number fingerprints are compressed into 3D space using **UMAP**, then grouped into 8 visual style clusters using **KMeans**. This reveals natural groupings like "night scenes", "candid portraits", "high contrast B&W", etc.

### 4. Score prediction (machine learning)
Three models are trained to predict upvotes from the fingerprints:
- **Ridge Regression** — simple baseline
- **Random Forest** — captures non-linear patterns
- **Gradient Boosting** — best overall performance

A separate **binary classifier** also predicts whether a photo is "high quality" or "low quality" (above/below median upvotes).

### 5. Technical analysis (OpenCV)
Completely separate from the AI score, classical computer vision measures:
- **Sharpness** — Laplacian variance detects blur
- **Exposure** — histogram analysis catches over/underexposure
- **Noise** — estimated from smooth image regions
- **Contrast** — RMS contrast across the image
- **Rule of thirds** — face detection checks if the subject falls on the grid intersections
- **Leading lines** — Hough line transform detects diagonal lines that draw the eye
- **Horizon levelness** — detects tilted horizons

### 6. Web app (Streamlit)
Everything is tied together in an interactive web app with four tabs:
| Tab | What it shows |
|---|---|
| 📊 Score & Style | Big score, AI quality verdict, visual style tags, style cluster with examples |
| 🔍 Similar Photos | 6 most visually similar photos from the dataset |
| 🗺️ Visual Style Map | 3D interactive map of all training photos coloured by cluster |
| 🔬 Technical Analysis | Composition overlay, metric scores, creative assessment chart |

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd street_photo_scorer
pip install -r requirements.txt

# 2. Run the data pipeline (first time only — takes ~20 min)
python -m src.scraping.arctic_shift_scraper    # Download ~1400 photos
python -m src.features.extract_embeddings      # Extract AI fingerprints
python -m src.models.cluster                   # Cluster into style groups
python -m src.models.scorer                    # Train scoring models

# 3. Launch the app
streamlit run src/app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## Project structure

```
street_photo_scorer/
├── data/
│   ├── raw/            # Downloaded images + posts.json       (git-ignored)
│   ├── processed/      # Embeddings, clustered CSV, UMAP data (git-ignored)
│   └── models/         # Trained ML models                    (git-ignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Score distribution, dataset overview
│   ├── 02_clustering.ipynb             # UMAP visualisation, cluster examples
│   ├── 03_scoring_model.ipynb          # Regression model evaluation
│   └── 04_classification_evaluation.ipynb  # Confusion matrix, AUC-ROC
│
├── src/
│   ├── scraping/        # Arctic Shift scraper (no Reddit account needed)
│   ├── features/        # CLIP embedding extraction
│   ├── models/          # Clustering, regression, and classifier training
│   ├── analysis/        # OpenCV technical quality analysis
│   └── app/             # Streamlit web app
│
├── requirements.txt
└── README.md
```

---

## Tech stack

| Area | Tool |
|---|---|
| Data collection | Arctic Shift Reddit archive |
| AI embeddings | CLIP ViT-B/32 (OpenAI via HuggingFace) |
| Dimensionality reduction | UMAP |
| Clustering | KMeans |
| Regression models | Ridge, Random Forest, Gradient Boosting |
| Classification | Logistic Regression, Random Forest, Gradient Boosting |
| Technical analysis | OpenCV |
| Web app | Streamlit + Plotly |
| Language | Python 3.10+ |

---

## Why upvotes as a quality proxy?

Upvotes are imperfect but useful. Here's what the data shows:

- **Photos with people score higher** — the community strongly rewards candid human moments
- **Black & white is popular** — signals intentionality and style
- **Night and low-light scenes perform well** — dramatic and visually distinctive
- **Architecture without people scores lower** — even if technically excellent

This means the score measures **community taste**, not universal photographic quality. A famous Ansel Adams landscape might score 3/10 — not because it's bad, but because it doesn't match what this specific community upvotes in 2023–2024.

This bias is documented and discussed in the project, and is part of what makes it interesting: the model reveals the gap between traditional photographic principles and real online community preferences.

---

## Limitations

- Dataset is limited to ~1,400 photos — a larger dataset would improve accuracy
- Model accuracy is modest (R² ≈ 0.10 for regression, AUC ≈ 0.66 for classification) — predicting viral content is genuinely hard
- The face detector uses Haar cascades (OpenCV), which misses profile faces and non-frontal subjects
- Scores reflect a snapshot of community taste from a specific time period
