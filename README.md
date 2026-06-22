# Street Photo Scorer

> Upload a street photo — get an aesthetic score, a technical breakdown, and see where your photo fits among 22,000+ real community-rated shots.

---

## What is this?

**Street Photo Scorer** is an AI-powered web app that analyses street photography. You upload a photo and it tells you:

- **How it would likely perform** in the r/streetphotography community (scored 0–10)
- **What visual style it has** — black & white, low light, candid portrait, etc.
- **Where it fits** among 22,374 real street photos on an interactive 3D style map
- **Which photos look most similar** to yours in the training dataset
- **Technical quality details** — sharpness, exposure, noise, contrast, and rule of thirds

The score is trained on real Reddit upvotes, so it reflects genuine community taste rather than a subjective opinion.

---

## How it works

### 1. Data collection
22,374 photos were scraped from [r/streetphotography](https://reddit.com/r/streetphotography) using the [Arctic Shift](https://arctic-shift.photon-reddit.com) Reddit archive — no Reddit account needed. Upvote counts are used as a proxy for aesthetic quality.

### 2. AI visual fingerprinting (CLIP)
Each photo is passed through **CLIP ViT-B/32** (OpenAI's vision model) which converts it into a 512-number fingerprint capturing its visual style, lighting, mood, and composition. Photos that look similar end up with similar fingerprints.

### 3. Style clustering (UMAP + KMeans)
The 512-number fingerprints are compressed into 3D space using **UMAP**, then grouped into 8 visual style clusters using **KMeans**. This reveals natural groupings like "night scenes", "candid portraits", "high contrast B&W", etc.

### 4. Score prediction (machine learning)
Three models are trained to predict upvotes from the fingerprints:
- **Ridge Regression** — best overall performance (R² = 0.10)
- **Random Forest** — captures non-linear patterns
- **Gradient Boosting** — alternative ensemble approach

A separate **binary classifier** also predicts whether a photo is "high quality" or "low quality" (above/below median upvotes). The best model is selected automatically based on validation performance.

The raw output is **percentile-ranked** against all 22,374 training photos so a score of 5.0 literally means better than 50% of the dataset.

### 5. Technical analysis (scipy/numpy/PIL)
Completely separate from the AI score, classical image analysis measures:
- **Sharpness** — Laplacian variance detects blur
- **Exposure** — histogram analysis catches over/underexposure
- **Noise** — estimated from smooth image regions after Gaussian blur
- **Contrast** — RMS contrast across the image
- **Rule of thirds** — saliency-based subject position compared to grid intersections

### 6. Web app (Streamlit)
Everything is tied together in an interactive web app with four tabs:
| Tab | What it shows |
|---|---|
| 📊 Score & Style | Big score, AI quality verdict, visual style tags, style cluster |
| 🔍 Similar Photos | 6 most visually similar photos from the dataset |
| 🗺️ Visual Style Map | 3D interactive map of all training photos coloured by cluster |
| 🔬 Technical Analysis | Composition overlay, metric scores, colour palette |

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd street_photo_scorer
pip install -r requirements.txt

# 2. Run the data pipeline (first time only — takes a few hours with 22k photos)
python -m src.scraping.arctic_shift_scraper    # Download ~22k photos
python -m src.features.extract_embeddings      # Extract AI fingerprints
python -m src.models.cluster                   # Cluster into style groups
python -m src.models.scorer                    # Train scoring models

# 3. Launch the app
streamlit run src/app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

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
│   ├── analysis/        # Technical quality analysis (scipy/numpy/PIL)
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
| Technical analysis | scipy, numpy, PIL |
| Web app | Streamlit + Plotly |
| Language | Python 3.10+ |

---

## Why upvotes as a quality proxy?

Upvotes are imperfect but useful. The data shows:

- **Photos with people score higher** — the community strongly rewards candid human moments
- **Black & white is popular** — signals intentionality and style
- **Night and low-light scenes perform well** — dramatic and visually distinctive
- **Architecture without people scores lower** — even if technically excellent

This means the score measures **community taste**, not universal photographic quality. A technically perfect landscape with no people may score lower than a blurry candid portrait — because that's what this community upvotes.

---

## Limitations

- Model accuracy is modest (R² = 0.10 for regression, AUC = 0.50 for classification — essentially random) — predicting viral content from visual features alone is genuinely hard
- Scores reflect a snapshot of community taste from a specific time period (2021–2024)
- Leading lines and horizon detection are not available in the current build
- The 3D style map uses nearest-neighbour approximation to place uploaded photos
