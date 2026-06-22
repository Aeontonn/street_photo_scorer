"""
Technical photo quality analysis using scipy/numpy/PIL.
No OpenCV dependency — all operations are native Python/numpy and cannot segfault.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import laplace, gaussian_filter


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    score: float          # 0–10
    label: str
    detail: str
    good: bool


@dataclass
class CompositionResult:
    rule_of_thirds_score: float
    rule_of_thirds_label: str
    subject_x: Optional[float]
    subject_y: Optional[float]
    subject_found: bool

    leading_lines_count: int
    leading_lines_score: float
    lines: list

    horizon_angle: Optional[float]
    horizon_score: float
    horizon_line: Optional[tuple]


@dataclass
class PhotoAnalysis:
    sharpness: MetricResult
    exposure: MetricResult
    noise: MetricResult
    contrast: MetricResult
    composition: CompositionResult
    overall_technical: float
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo=0.0, hi=10.0) -> float:
    return max(lo, min(hi, v))


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32)


def _thirds_score(nx: float, ny: float) -> float:
    thirds_x = [1/3, 2/3]
    thirds_y = [1/3, 2/3]
    min_dist = min(
        math.hypot(nx - tx, ny - ty)
        for tx in thirds_x for ty in thirds_y
    )
    return _clamp(10 - min_dist * 30)


# ── Individual metrics ────────────────────────────────────────────────────────

def analyze_sharpness(gray: np.ndarray) -> MetricResult:
    lap_var = float(np.var(laplace(gray)))

    if lap_var >= 600:
        score, label = 9.5, "Very sharp"
    elif lap_var >= 300:
        score, label = 8.0, "Sharp"
    elif lap_var >= 150:
        score, label = 6.5, "Acceptable sharpness"
    elif lap_var >= 60:
        score, label = 4.0, "Slightly blurry"
    elif lap_var >= 20:
        score, label = 2.0, "Blurry"
    else:
        score, label = 0.5, "Very blurry"

    good = score >= 6.0
    detail = (
        f"Laplacian variance: {lap_var:.0f}. "
        + ("Motion blur or out-of-focus areas detected." if not good else "Image appears in focus.")
    )
    return MetricResult(score=score, label=label, detail=detail, good=good)


def analyze_exposure(gray: np.ndarray) -> MetricResult:
    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    total = gray.size
    mean_brightness = float(gray.mean())

    dark_frac   = float(hist[:30].sum()  / total)
    bright_frac = float(hist[230:].sum() / total)

    if dark_frac > 0.6:
        score, label = 2.5, "Severely underexposed"
        detail = f"{dark_frac*100:.0f}% of pixels are very dark. The photo may lack detail in shadows."
        good = False
    elif dark_frac > 0.35:
        score, label = 5.0, "Underexposed"
        detail = f"Heavy shadow coverage ({dark_frac*100:.0f}% dark pixels). May be intentional low-key."
        good = False
    elif bright_frac > 0.5:
        score, label = 2.5, "Severely overexposed"
        detail = f"{bright_frac*100:.0f}% of pixels are blown out. Highlight detail is lost."
        good = False
    elif bright_frac > 0.25:
        score, label = 5.0, "Overexposed"
        detail = f"Significant highlight clipping ({bright_frac*100:.0f}% bright pixels)."
        good = False
    elif 80 <= mean_brightness <= 180:
        score, label = 9.0, "Well exposed"
        detail = f"Mean brightness {mean_brightness:.0f}/255 — good tonal range."
        good = True
    else:
        score, label = 7.0, "Acceptable exposure"
        detail = f"Mean brightness {mean_brightness:.0f}/255."
        good = True

    return MetricResult(score=score, label=label, detail=detail, good=good)


def analyze_noise(gray: np.ndarray) -> MetricResult:
    blurred = gaussian_filter(gray, sigma=2.0)
    residual = np.abs(gray - blurred)

    # Edge mask via gradient magnitude (avoids sampling noisy areas near edges)
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    edge_mag = np.hypot(gx, gy)
    edge_mask = edge_mag > (edge_mag.mean() + edge_mag.std())
    non_edge = residual[~edge_mask]
    noise_std = float(non_edge.std()) if len(non_edge) > 100 else float(residual.std())

    if noise_std < 2.0:
        score, label = 9.5, "Very clean"
    elif noise_std < 4.0:
        score, label = 8.0, "Clean"
    elif noise_std < 7.0:
        score, label = 6.5, "Low noise"
    elif noise_std < 12.0:
        score, label = 4.5, "Moderate noise"
    elif noise_std < 20.0:
        score, label = 2.5, "Noisy"
    else:
        score, label = 1.0, "Very noisy"

    good = score >= 6.0
    detail = f"Noise estimate: {noise_std:.1f}. {'High ISO or low-light shot.' if not good else 'Image is clean.'}"
    return MetricResult(score=score, label=label, detail=detail, good=good)


def analyze_contrast(gray: np.ndarray) -> MetricResult:
    rms = float(gray.std())

    if rms >= 70:
        score, label = 9.5, "High contrast"
    elif rms >= 50:
        score, label = 8.0, "Good contrast"
    elif rms >= 35:
        score, label = 6.0, "Moderate contrast"
    elif rms >= 20:
        score, label = 4.0, "Low contrast"
    else:
        score, label = 2.0, "Very flat"

    good = score >= 6.0
    detail = f"RMS contrast: {rms:.1f}. {'Photo lacks tonal variation.' if not good else 'Good tonal range.'}"
    return MetricResult(score=score, label=label, detail=detail, good=good)


# ── Composition ───────────────────────────────────────────────────────────────

def analyze_composition(gray: np.ndarray) -> CompositionResult:
    h, w = gray.shape

    # Saliency proxy: find the brightest region after strong blur
    blurred = gaussian_filter(gray, sigma=max(h, w) / 20)
    max_pos  = np.unravel_index(blurred.argmax(), blurred.shape)
    subject_y = max_pos[0] / h
    subject_x = max_pos[1] / w

    rot_score = _thirds_score(subject_x, subject_y)
    if rot_score >= 8:
        rot_label = "On a thirds intersection"
    elif rot_score >= 6:
        rot_label = "Near a thirds line"
    elif rot_score >= 4:
        rot_label = "Slightly off-thirds"
    else:
        rot_label = "Centered or at edge"

    return CompositionResult(
        rule_of_thirds_score=rot_score,
        rule_of_thirds_label=rot_label,
        subject_x=subject_x,
        subject_y=subject_y,
        subject_found=False,
        # Leading lines and horizon not computed (no Hough transform without OpenCV)
        leading_lines_count=0,
        leading_lines_score=5.0,
        lines=[],
        horizon_angle=None,
        horizon_score=7.0,
        horizon_line=None,
    )


# ── Annotated overlay using PIL ImageDraw ─────────────────────────────────────

def _draw_overlay(img_pil: Image.Image, comp: CompositionResult) -> np.ndarray:
    overlay = img_pil.convert("RGBA")
    draw    = ImageDraw.Draw(overlay)
    w, h    = overlay.size

    # Rule-of-thirds grid
    for f in [1/3, 2/3]:
        draw.line([(int(w * f), 0), (int(w * f), h)], fill=(200, 200, 200, 180), width=1)
        draw.line([(0, int(h * f)), (w, int(h * f))], fill=(200, 200, 200, 180), width=1)

    # Intersection dots
    for tx in [1/3, 2/3]:
        for ty in [1/3, 2/3]:
            cx, cy = int(w * tx), int(h * ty)
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6],
                         outline=(255, 255, 255, 220), width=2)

    # Subject saliency dot
    if comp.subject_x is not None:
        sx = int(comp.subject_x * w)
        sy = int(comp.subject_y * h)
        draw.ellipse([sx - 12, sy - 12, sx + 12, sy + 12],
                     outline=(0, 200, 255, 220), width=3)

    return np.array(overlay.convert("RGB"))


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze(img_pil: Image.Image, max_dim: int = 800) -> PhotoAnalysis:
    """Run all technical analyses on a PIL image. Returns a PhotoAnalysis."""
    img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = _to_gray(img_pil)

    sharpness   = analyze_sharpness(gray)
    exposure    = analyze_exposure(gray)
    noise       = analyze_noise(gray)
    contrast    = analyze_contrast(gray)
    composition = analyze_composition(gray)

    overall = (
        sharpness.score * 0.30 +
        exposure.score  * 0.20 +
        noise.score     * 0.10 +
        contrast.score  * 0.15 +
        composition.rule_of_thirds_score * 0.25
    )

    annotated_rgb = _draw_overlay(img_pil, composition)

    return PhotoAnalysis(
        sharpness=sharpness,
        exposure=exposure,
        noise=noise,
        contrast=contrast,
        composition=composition,
        overall_technical=round(_clamp(overall), 1),
        annotated_image=annotated_rgb,
    )


# Keep safe_analyze for backwards compatibility (now just calls analyze directly
# since there's no native code to crash)
def safe_analyze(img_pil: Image.Image, timeout: int = 30):
    try:
        return analyze(img_pil)
    except Exception:
        return None
