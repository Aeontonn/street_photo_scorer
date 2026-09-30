# Tasks

- [x] T1: `tools/build_deploy_assets.py SRC_DATA_DIR OUT_DIR` - copies models/ + processed/, writes `thumbs.zip` (idx -> 200px JPEG q70)
  - Acceptance: OUT_DIR holds models/, processed/, thumbs.zip; zip has one entry per csv row; total < 500 MB
  - Verify: run it against `../photo-scorer-app/data`; `unzip -l` count == csv rows; `du -sh OUT_DIR`
  - Files: tools/build_deploy_assets.py

- [x] T2: `ARTIFACTS_DIR` env var + thumbnail loader; swap the two `Image.open(local_path)` calls
  - Acceptance: with only the bundle present (no data/raw), Similar Photos and cluster examples show images
  - Verify: `ARTIFACTS_DIR=<OUT_DIR> streamlit run src/app/Score_Photos.py`, upload a photo, check both tabs
  - Files: src/scoring/scorer.py, src/app/Score_Photos.py

- [x] T3: smoke test `tests/test_scoring_smoke.py` (load_resources + score_image on a generated image)
  - Acceptance: score in 0-10, similar list non-empty, uses the bundle only; skipped if bundle missing
  - Verify: `ARTIFACTS_DIR=<OUT_DIR> python -m pytest tests -q`
  - Files: tests/test_scoring_smoke.py

- [x] T4: `requirements-deploy.txt` + Space README front matter (sdk: streamlit, app_file) + upload cap check
  - Acceptance: fresh venv `pip install -r requirements-deploy.txt` runs the app; no jupyter/praw/seaborn/matplotlib
  - Verify: clean venv install + T2 run
  - Files: requirements-deploy.txt, README.md (or Space-only README)

- [ ] T5: Create the Space, push code + bundle (LFS/Xet), first build  [ASK FIRST: needs your HF account/token in your own terminal]
  - Acceptance: public `*.hf.space` URL scores a photo end to end, all 4 tabs + Compare page, cold start < 2 min
  - Verify: open from phone and a clean browser

- [ ] T6: README "Try it" link + redeploy instructions
  - Acceptance: README links live app, explains how to rebuild the bundle and push
  - Files: README.md
