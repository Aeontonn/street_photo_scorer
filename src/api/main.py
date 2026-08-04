"""FastAPI service exposing the Street Photo Scorer pipeline over HTTP.

All scoring logic lives in src.scoring.scorer — this module is just the
HTTP layer (upload handling, ranking, error responses) on top of it.

Run locally with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.scoring.scorer import score_image

app = FastAPI(
    title="Street Photo Scorer API",
    description="Score street photos for aesthetic quality using a CLIP + regression pipeline.",
    version="0.1.0",
)


def _load_image(data: bytes, filename: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"Could not read '{filename}' as an image")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    """Score a single uploaded image."""
    data = await file.read()
    image = _load_image(data, file.filename)
    result = score_image(image)
    return {"filename": file.filename, **result}


@app.post("/score-batch")
async def score_batch(files: list[UploadFile] = File(...)):
    """Score multiple uploaded images and return them ranked best-first."""
    scored, failed = [], []

    for f in files:
        data = await f.read()
        try:
            image = _load_image(data, f.filename)
            result = score_image(image)
            scored.append({"filename": f.filename, **result})
        except HTTPException as exc:
            failed.append({"filename": f.filename, "error": exc.detail})
        except Exception as exc:
            failed.append({"filename": f.filename, "error": f"Scoring failed: {exc}"})

    scored.sort(key=lambda r: r["aesthetic_score"], reverse=True)
    for rank, r in enumerate(scored, start=1):
        r["rank"] = rank

    return {"results": scored, "failed": failed}
