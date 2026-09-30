"""FastAPI service exposing the Street Photo Scorer pipeline over HTTP.

All scoring logic lives in src.scoring.scorer — this module is just the
HTTP layer (upload handling, ranking, error responses) on top of it.

Run locally with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import random
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.concurrency import run_in_threadpool

load_dotenv()

log = logging.getLogger("uvicorn.error")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# DEMO_MODE=1 skips the real model (no torch, no artifacts needed) so the frontend can be developed.
# If the real model fails to load, the API falls back to demo mode on its own.
demo_mode = os.environ.get("DEMO_MODE") == "1"

# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS=https://scorer.example.com,https://www.scorer.example.com
# Falls back to the React dev server's default ports (Vite/CRA) if unset.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8080").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load CLIP + the trained models once at startup instead of on the first
    # request, so the first real user isn't the one paying the load cost.
    global demo_mode
    if not demo_mode:
        try:
            from src.scoring.scorer import load_resources
            await run_in_threadpool(load_resources)
        except Exception as exc:
            log.warning("Could not load the real model, serving DEMO scores: %s", exc)
            demo_mode = True
    yield


app = FastAPI(
    title="Street Photo Scorer API",
    description="Score street photos for aesthetic quality using a CLIP + regression pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_image(data: bytes, filename: str) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That image is over 10 MB. Please upload a smaller one.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise ValueError(image.format)
        image.load()  # decode now so a corrupt file fails here, not inside the model
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG or WebP image.")


def _demo_result(data: bytes) -> dict:
    """Fake but deterministic result (same image gives the same score). Clearly labelled via demo=True."""
    rng = random.Random(hashlib.sha256(data).hexdigest())
    percentile = round(rng.uniform(5, 95), 1)
    label = ("Low match", "Below average", "Above average", "Strong", "Exceptional")[
        sum(percentile >= t for t in (25, 50, 75, 90))
    ]
    return {
        "demo": True,
        "aesthetic_score": round(percentile / 10, 1),
        "percentile": percentile,
        "score_label": label,
        "quality_verdict": "high" if percentile >= 50 else "low",
        "confidence": round(rng.uniform(0.5, 0.9), 3),
        "cluster_id": 3,
        "cluster_name": "B&W Classics",
        "attributes": [
            {"label": "Black & white", "similarity": 0.284},
            {"label": "People present", "similarity": 0.251},
            {"label": "High contrast", "similarity": 0.233},
        ],
        "genre_scores": [
            {"label": "Candid / decisive moment", "score": 0.41},
            {"label": "Fine art B&W", "score": 0.27},
            {"label": "Documentary", "score": 0.14},
        ],
    }


async def _score_one(data: bytes, filename: str) -> dict:
    image = _load_image(data, filename)
    if demo_mode:
        return {"filename": filename, **_demo_result(data)}
    from src.scoring.scorer import score_image
    result = await run_in_threadpool(score_image, image)
    return {"filename": filename, "demo": False, **result}


@app.get("/health")
def health():
    return {"status": "ok", "model": "demo" if demo_mode else "real"}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    """Score a single uploaded image (JPG, PNG or WebP, max 10 MB)."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return await _score_one(data, file.filename)
    except HTTPException:
        raise
    except Exception:
        log.exception("Scoring failed")
        raise HTTPException(status_code=500, detail="Something went wrong while scoring. Please try another photo.")


@app.post("/score-batch")
async def score_batch(files: list[UploadFile] = File(...)):
    """Score multiple uploaded images and return them ranked best-first."""
    scored, failed = [], []

    for f in files:
        data = await f.read(MAX_UPLOAD_BYTES + 1)
        try:
            scored.append(await _score_one(data, f.filename))
        except HTTPException as exc:
            failed.append({"filename": f.filename, "error": exc.detail})
        except Exception as exc:
            failed.append({"filename": f.filename, "error": f"Scoring failed: {exc}"})

    scored.sort(key=lambda r: r["aesthetic_score"], reverse=True)
    for rank, r in enumerate(scored, start=1):
        r["rank"] = rank

    return {"results": scored, "failed": failed}
