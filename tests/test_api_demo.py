"""API checks that need no model: DEMO_MODE=1 python -m pytest tests/test_api_demo.py"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api import main

main.demo_mode = True  # force demo so the test never loads CLIP
client = TestClient(main.app)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "gray").save(buf, "PNG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok", "model": "demo"}


def test_score_demo_is_labelled_and_deterministic():
    post = lambda: client.post("/score", files={"file": ("a.png", _png(), "image/png")}).json()
    first = post()
    assert first["demo"] is True and 0 <= first["aesthetic_score"] <= 10
    assert post()["aesthetic_score"] == first["aesthetic_score"]


def test_rejects_non_image_and_oversize():
    r = client.post("/score", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/score", files={"file": ("a.png", b"0" * (main.MAX_UPLOAD_BYTES + 1), "image/png")})
    assert r.status_code == 413
