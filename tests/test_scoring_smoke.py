"""End-to-end smoke test on the deploy bundle: ARTIFACTS_DIR=deploy_bundle pytest tests/test_scoring_smoke.py"""

import numpy as np
import pytest
from PIL import Image

from src.scoring import scorer

pytestmark = pytest.mark.skipif(
    not (scorer.MODELS_DIR / "scorer.pkl").exists(), reason="artifacts not found (set ARTIFACTS_DIR)"
)


def test_score_and_thumbnails():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    assert 0 <= scorer.score_image(img)["aesthetic_score"] <= 10

    r = scorer.run_pipeline(img, scorer.load_resources())
    df = scorer.load_resources().df
    assert len(r["similar_indices"]) > 0
    assert scorer.load_thumb(df.iloc[r["similar_indices"][0]]).size[0] > 0
