# Spec: Public hosted Street Photo Scorer

## Objective
Anyone with a plain URL can open the app and score a photo. No local hosting by the owner,
and no scraping, embedding or training run by anyone.

Success: a stranger on a phone opens the link, uploads a photo, and sees the score,
style, similar photos, style map and technical analysis within one cold-start wait.

## Findings that shape the design
- Runtime needs only small artifacts: `data/models/` (128 KB) and `data/processed/` (49 MB: `embeddings.npy`, `umap_3d.npy`, `clustered.csv`). They are git-ignored, so a deploy from git has none of them. This is why hosting fails today.
- "Similar Photos" and "cluster examples" read `row["local_path"]` from `data/raw/images/`, which is 30 GB / ~22k files. That cannot ship to a host. This is the hard part.
- CLIP ViT-B/32 (~600 MB) is downloaded from HuggingFace on first start. `torch` makes the image large (~2 GB CPU-only).
- `src/scoring/scorer.py` already exposes `load_resources()`, and the API preloads it at startup.
- The Procfile runs only Streamlit. The FastAPI service exists but is not deployed.

## Assumptions (correct me now or I proceed)
1. Public **Streamlit app** is the deliverable. The FastAPI service stays local/optional.
2. Free or near-free hosting is fine (Hugging Face Spaces or Streamlit Community Cloud). Cold starts of ~1 min after idle are acceptable.
3. CPU-only inference. A single photo scores in a few seconds.
4. No accounts, no database, no storage of uploaded photos. Photos are processed in memory only.
5. The 22k training photos are scraped Reddit content. I will not rehost the full-size originals publicly.
6. Model quality is unchanged. This is deployment work, not a retrain.

## Tech stack
Streamlit (existing), CPU `torch`, `transformers` CLIP, artifacts served from a host-side store (see Plan).
Preferred host: **Hugging Face Spaces (Streamlit SDK)** — 16 GB RAM free, large-file storage for artifacts, a stable `*.hf.space` link.
Fallback: Render or Railway with a Docker image.

## Commands
```
Local run:      streamlit run src/app/Score_Photos.py
Tests:          python -m pytest tests -q
Build artifacts (one-off, owner only): python -m tools.build_deploy_assets   # new, creates thumbnails + bundle
Deploy:         git push hf main    # Spaces builds from the repo
```

## Project structure (changes only)
```
deploy_assets/    → thumbnails (~256px JPEG) of the photos shown as "similar"; committed via Git LFS or hosted on the Hub
tools/build_deploy_assets.py → makes thumbnails and rewrites local_path → thumb path in clustered.csv
requirements-deploy.txt      → runtime-only deps (no jupyter, praw, seaborn, matplotlib, tqdm; CPU torch)
README.md                    → "Try it" link, plus how artifacts are produced
```

## Code style
Match existing modules: small functions, type hints, `Path` constants at the top of `scorer.py`.
Artifact location comes from one env var, `ARTIFACTS_DIR` (default `data`), not new config layers.

## Testing strategy
- Existing `tests/test_photo_analysis.py` keeps running.
- One smoke test: `load_resources()` then `score_image()` on a fixture image, asserting a 0–10 score and a non-empty similar list. It runs against the deploy bundle, not the raw images.
- Manual acceptance on the live URL, from a phone and from a machine that has never seen the repo.

## Boundaries
- Always: keep raw images and scraped data out of git. Test the deployed bundle from a clean checkout.
- Ask first: adding a hosting account or paid tier, Git LFS, changing the model or dependencies, showing training photos publicly.
- Never: commit `.env` or tokens, upload user photos anywhere, retrain as part of deploy.

## Success criteria
1. A public URL loads the app with no local setup.
2. A fresh clone plus the documented deploy command produces a working app with no scraping, embedding or training step.
3. All four tabs work on the live site, including Similar Photos with images (thumbnails).
4. Cold start ends in under 2 minutes. A warm score returns in under 10 seconds for a 12 MP photo.
5. The deploy bundle is under the host's limits (target: less than 500 MB excluding the CLIP download).
6. Concurrent users do not crash it. Uploads are capped at ~10 MB and resized as they already are.
7. The README links the live app and says how to redeploy.

## Open questions (need your decision)
1. **Similar-photo images.** Pick one:
   a. ~256px thumbnails of all 22k photos (~150–300 MB), hosted with the app. *(Recommended: it works offline and the links can't rot.)*
   b. Hotlink Reddit or Arctic Shift URLs. Zero storage, but images break as posts are deleted.
   c. Drop the images and show only title, score and cluster.
2. **Host:** Hugging Face Spaces (recommended), Streamlit Community Cloud, or a paid box?
3. **Copyright and privacy:** are you fine showing other people's Reddit photos as thumbnails on a public site? Option c avoids that.
4. Should `Compare Photos` (page 2) ship, or only the main page?
5. Do you want the FastAPI service deployed too? I'd skip it (YAGNI) unless you have a frontend that needs it.

## Next phase (after approval)
Plan → `tasks/plan.md`. Rough order: (1) thumbnails and bundle script, (2) `requirements-deploy.txt` and `ARTIFACTS_DIR`, (3) smoke test, (4) deploy to Spaces, (5) README.
