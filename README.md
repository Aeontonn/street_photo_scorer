# Street Photo Scorer

> Individuellt stort projekt — Tillämpad AI

Ett end-to-end machine learning-system som bedömer estetisk kvalitet i gatufotografi. Systemet skrapar foton från r/streetphotography, använder CLIP för bildembeddings, klustar stilar med UMAP + KMeans, och tränar en regressionsmodell med upvotes som proxy för estetisk kvalitet.

---

## Pipeline-översikt

```
Reddit API ──► Bildnedladdning ──► CLIP Embeddings ──► UMAP + Klustring
                                                              │
                                                    Regressionsmodell
                                                    (score → 0–10)
                                                              │
                                                    Streamlit Demo App
```

### Steg 1 — Datascraping (`src/scraping/reddit_scraper.py`)
- Hämtar top-posts från r/streetphotography via Reddit API (PRAW)
- Sparar metadata: score, upvote_ratio, num_comments, bildURL
- Laddar ner bilder lokalt

### Steg 2 — Feature-extraktion (`src/features/extract_embeddings.py`)
- CLIP (ViT-B/32) extraherar 512-dimensionella embeddings per bild
- Embeddings representerar estetisk stil, komposition, ljussättning

### Steg 3 — Klustring (`src/models/cluster.py`)
- UMAP reducerar 512D → 2D/3D för visualisering
- KMeans (eller HDBSCAN) identifierar stilkluster
- Varje kluster representerar en fotografisk stil (ex: svartvit urban, golden hour, abstrakt)

### Steg 4 — Scoring-modell (`src/models/scorer.py`)
- **Target**: `log1p(score)` — log-normaliserade upvotes som estetisk proxy
- **Modeller**: Ridge Regression, Random Forest, Gradient Boosting
- **Utvärdering**: R², MAE, visualisering av prediktioner vs faktiska scores

### Steg 5 — Demo-app (`src/app/streamlit_app.py`)
- Ladda upp en valfri gatufoto och få ett estetiskt score (0–10)

---

## Snabbstart

```bash
# 1. Klona och installera
git clone <repo>
cd street_photo_scorer
pip install -r requirements.txt

# 2. Sätt upp Reddit API-nycklar
cp .env.example .env
# Fyll i REDDIT_CLIENT_ID och REDDIT_CLIENT_SECRET

# 3. Kör pipeline
python -m src.scraping.reddit_scraper          # ~2000 bilder
python -m src.features.extract_embeddings      # CLIP embeddings
python -m src.models.cluster                   # UMAP + klustring
python -m src.models.scorer                    # träna scorer

# 4. Starta demo-appen
streamlit run src/app/streamlit_app.py
```

---

## Projektstruktur

```
street_photo_scorer/
├── data/
│   ├── raw/            # Nedladdade bilder + posts.json (git-ignorad)
│   ├── processed/      # Embeddings, CSV med kluster (git-ignorad)
│   └── models/         # Tränade modeller (git-ignorad)
├── notebooks/          # EDA och experiment
├── src/
│   ├── scraping/       # Reddit-scraper
│   ├── features/       # CLIP embedding-extraktion
│   ├── models/         # Klustring + scoring-modell
│   ├── evaluation/     # Metrics och plots
│   └── app/            # Streamlit-demo
├── tests/
├── requirements.txt
└── .env.example
```

---

## Tekniker som används

| Område | Teknik |
|---|---|
| Scraping | PRAW (Reddit API) |
| Feature-extraktion | CLIP (OpenAI, via HuggingFace) |
| Dimensionsreduktion | UMAP |
| Unsupervised learning | KMeans, HDBSCAN |
| Supervised learning | Ridge, Random Forest, Gradient Boosting |
| Utvärdering | R², MAE, confusion matrix för kluster |
| Deployment | Streamlit |

---

## Diskussion — Subjektivitet och bias

Upvotes är en imperfekt proxy för estetisk kvalitet:
- Popularitetsbias: välkända fotografer får fler upvotes
- Trendkänslighet: stilar som är populära just nu premieras
- Gemenskapsbias: r/streetphotography har en specifik smakprofil

Modellen lär sig vad **denna community** gillar — inte universell estetik. Det är en feature, inte en bug: systemet kan visa hur community-smak skiljer sig från traditionella fotografiprinciper.

---

## Reddit API-setup

1. Gå till [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Skapa en ny app (typ: script)
3. Kopiera `client_id` och `client_secret` till `.env`
