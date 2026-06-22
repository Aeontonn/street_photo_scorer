import numpy as np
import pytest
from PIL import Image

from src.analysis.photo_analysis import (
    analyze,
    analyze_sharpness,
    analyze_contrast,
    analyze_exposure,
    analyze_noise,
    analyze_composition,
    safe_analyze,
)


def _make_image(width=200, height=150, mode="RGB", fill=128):
    return Image.new(mode, (width, height), fill)


def _gray_array(width=200, height=150, value=128):
    return np.full((height, width), value, dtype=np.float32)


# ── Sharpness ─────────────────────────────────────────────────────────────────

def test_sharpness_returns_valid_score():
    result = analyze_sharpness(_gray_array())
    assert 0 <= result.score <= 10
    assert result.label
    assert isinstance(result.good, bool)


def test_sharpness_noisy_image_scores_higher():
    flat = _gray_array()
    noisy = np.random.default_rng(0).integers(0, 255, flat.shape).astype(np.float32)
    flat_score = analyze_sharpness(flat).score
    noisy_score = analyze_sharpness(noisy).score
    assert noisy_score > flat_score


# ── Exposure ──────────────────────────────────────────────────────────────────

def test_exposure_well_lit_image():
    gray = _gray_array(value=128)
    result = analyze_exposure(gray)
    assert result.good is True
    assert result.score >= 7.0


def test_exposure_black_image_is_underexposed():
    gray = _gray_array(value=5)
    result = analyze_exposure(gray)
    assert result.good is False
    assert result.score < 6.0


def test_exposure_white_image_is_overexposed():
    gray = _gray_array(value=250)
    result = analyze_exposure(gray)
    assert result.good is False
    assert result.score < 6.0


# ── Noise ─────────────────────────────────────────────────────────────────────

def test_noise_flat_image_is_clean():
    result = analyze_noise(_gray_array())
    assert result.score >= 8.0


def test_noise_random_image_scores_lower():
    flat_score = analyze_noise(_gray_array()).score
    noisy = np.random.default_rng(1).integers(0, 255, (150, 200)).astype(np.float32)
    noisy_score = analyze_noise(noisy).score
    assert noisy_score < flat_score


# ── Contrast ──────────────────────────────────────────────────────────────────

def test_contrast_flat_image_is_low():
    result = analyze_contrast(_gray_array())
    assert result.score <= 4.0


def test_contrast_high_contrast_image():
    checker = np.tile([[0, 255], [255, 0]], (75, 100)).astype(np.float32)
    result = analyze_contrast(checker)
    assert result.score >= 8.0


# ── Composition ───────────────────────────────────────────────────────────────

def test_composition_returns_valid_fields():
    result = analyze_composition(_gray_array())
    assert 0 <= result.rule_of_thirds_score <= 10
    assert result.subject_x is not None
    assert result.subject_y is not None
    assert result.subject_found is False
    assert result.leading_lines_count == 0


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_analyze_returns_all_fields():
    img = _make_image()
    result = analyze(img)
    assert 0 <= result.sharpness.score <= 10
    assert 0 <= result.exposure.score <= 10
    assert 0 <= result.noise.score <= 10
    assert 0 <= result.contrast.score <= 10
    assert 0 <= result.overall_technical <= 10
    assert result.annotated_image is not None
    assert result.annotated_image.shape[2] == 3  # RGB


def test_analyze_resizes_large_image():
    big = _make_image(width=2000, height=1500)
    result = analyze(big, max_dim=800)
    assert result.annotated_image.shape[1] <= 800
    assert result.annotated_image.shape[0] <= 800


def test_safe_analyze_returns_none_on_bad_input():
    result = safe_analyze(None)
    assert result is None


def test_safe_analyze_works_on_valid_image():
    result = safe_analyze(_make_image())
    assert result is not None
    assert 0 <= result.overall_technical <= 10
