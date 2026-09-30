# Plan: Public hosted Street Photo Scorer

Decisions (from SPEC.md): Hugging Face Spaces (Streamlit SDK), thumbnails for Similar Photos,
Compare page ships, FastAPI not deployed.

## Approach
1. **Thumbnails in one zip.** 22k loose files is painful in a git repo; one `thumbs.zip`
   (~200px JPEG, q70, ~150-250 MB) keyed by dataframe row index, read with stdlib `zipfile`.
   Only two call sites read images: `Score_Photos.py:395` and `:416`.
2. **`ARTIFACTS_DIR` env var** (default `data`) in `scorer.py` so the deployed bundle layout is
   the same as local: `models/`, `processed/`, `thumbs.zip`. No new config layer.
3. **Bundle** = `data/models` + `data/processed` (embeddings, umap_3d, clustered.csv) + `thumbs.zip`.
   Pushed to the Space via Git LFS/Xet (files >10 MB). `clustered.csv` keeps `local_path` (unused at runtime).
4. **`requirements-deploy.txt`**: runtime only, CPU torch. Space's `README.md` front matter points
   `app_file` at `src/app/Score_Photos.py`.
5. Smoke test, deploy, README link.

## Risks
- Cold start: CLIP download (~600 MB) on each Space wake. Mitigate: HF caches it in the Space; accept ~1 min.
- Repo/LFS size limit on the Space. Mitigate: measure `thumbs.zip` first; drop to 160px if >300 MB.
- The 30 GB raw images live only in `../photo-scorer-app/data/raw` (this repo's `data/` is empty).
  The build script must take the source path as an argument.
- Copyright: Reddit photos as thumbnails, accepted per user's "go with recommendations".

## Order / checkpoints
T1 -> T2 -> T3 (local app works from the bundle only) -> T4 -> T5 (live) -> T6.
