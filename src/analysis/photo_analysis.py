"""
Technical photo quality analysis using classical computer vision (OpenCV).

Measures sharpness, exposure, noise, contrast, rule of thirds, leading lines,
and horizon levelness. Returns structured results ready for display.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import cv2
import numpy as np
from PIL import Image


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    score: float          # 0–10
    label: str            # e.g. "Sharp", "Slightly blurry"
    detail: str           # one-sentence explanation
    good: bool            # True = green, False = amber/red


@dataclass
class CompositionResult:
    rule_of_thirds_score: float       # 0–10
    rule_of_thirds_label: str
    subject_x: Optional[float]        # 0–1 normalised position of main subject
    subject_y: Optional[float]
    subject_found: bool

    leading_lines_count: int
    leading_lines_score: float        # 0–10
    lines: list                       # list of (x1,y1,x2,y2) for overlay

    horizon_angle: Optional[float]    # degrees off level (None if not found)
    horizon_score: float              # 0–10
    horizon_line: Optional[tuple]     # (x1,y1,x2,y2) for overlay


@dataclass
class PhotoAnalysis:
    sharpness: MetricResult
    exposure: MetricResult
    noise: MetricResult
    contrast: MetricResult
    composition: CompositionResult
    overall_technical: float          # 0–10 weighted average
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _clamp(v: float, lo=0.0, hi=10.0) -> float:
    return max(lo, min(hi, v))


# ── Individual metrics ────────────────────────────────────────────────────────

def analyze_sharpness(gray: np.ndarray) -> MetricResult:
    """Laplacian variance — higher = sharper."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Empirically calibrated thresholds for typical photos at ~1000px wide
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
    detail = f"Laplacian variance: {lap_var:.0f}. {'Motion blur or out-of-focus areas detected.' if not good else 'Image appears in focus.'}"
    return MetricResult(score=score, label=label, detail=detail, good=good)


def analyze_exposure(gray: np.ndarray) -> MetricResult:
    """Histogram-based exposure analysis."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size
    mean_brightness = float(gray.mean())

    dark_frac = float(hist[:30].sum() / total)
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
    """Estimate noise level from high-frequency residual after Gaussian blur."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = np.abs(gray.astype(np.float32) - blurred)

    # Sample from mid-tone regions (avoid edges/detail which look like noise)
    edge_mask = cv2.Canny(gray, 50, 150)
    non_edge = residual[edge_mask == 0]
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
    """RMS contrast — standard deviation of pixel values."""
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

_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE


def _thirds_score(nx: float, ny: float) -> float:
    """Score 0–10 based on how close (nx, ny) is to a rule-of-thirds intersection."""
    thirds_x = [1/3, 2/3]
    thirds_y = [1/3, 2/3]
    min_dist = min(
        math.hypot(nx - tx, ny - ty)
        for tx in thirds_x for ty in thirds_y
    )
    # Distance of 0 = perfect thirds, distance of 0.25+ = center or edge
    return _clamp(10 - min_dist * 30)


def analyze_composition(bgr: np.ndarray, gray: np.ndarray) -> CompositionResult:
    h, w = gray.shape

    # ── Rule of thirds: try face detection first, fall back to saliency proxy ──
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    subject_x = subject_y = None
    subject_found = False

    if len(faces) > 0:
        # Use largest face as main subject
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        subject_x = (fx + fw / 2) / w
        subject_y = (fy + fh / 2) / h
        subject_found = True
    else:
        # Saliency proxy: use the brightest region in the image
        # Works surprisingly well for high-contrast street shots
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        subject_x = max_loc[0] / w
        subject_y = max_loc[1] / h

    rot_score = _thirds_score(subject_x, subject_y)
    if rot_score >= 8:
        rot_label = "On a thirds intersection"
    elif rot_score >= 6:
        rot_label = "Near a thirds line"
    elif rot_score >= 4:
        rot_label = "Slightly off-thirds"
    else:
        rot_label = "Centered or at edge"

    # ── Leading lines (Hough line transform) ──────────────────────────────────
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=200, minLineLength=int(min(h, w) * 0.28),
        maxLineGap=10
    )

    lines = []
    if raw_lines is not None:
        # Keep diagonal lines only (not purely horizontal/vertical)
        candidates = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            length = math.hypot(x2 - x1, y2 - y1)
            if 12 < angle < 78 or 102 < angle < 168:
                candidates.append((length, angle, int(x1), int(y1), int(x2), int(y2)))

        # Deduplicate: keep at most one representative per 15-degree angle bucket
        candidates.sort(key=lambda c: c[0], reverse=True)
        seen_buckets: set[int] = set()
        for length, angle, x1, y1, x2, y2 in candidates:
            bucket = int(angle // 15)
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                lines.append((x1, y1, x2, y2))

    n_lines = len(lines)
    if n_lines >= 5:
        ll_score, ll_label = 9.0, f"{n_lines} strong leading lines"
    elif n_lines >= 3:
        ll_score, ll_label = 7.0, f"{n_lines} leading lines"
    elif n_lines >= 1:
        ll_score, ll_label = 5.0, f"{n_lines} leading line"
    else:
        ll_score, ll_label = 3.0, "No clear leading lines"

    # ── Horizon levelness ─────────────────────────────────────────────────────
    h_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=60, minLineLength=int(w * 0.25),
        maxLineGap=30
    )

    horizon_angle = None
    horizon_line_coords = None

    if h_lines is not None:
        candidates = []
        for line in h_lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            if abs(angle) < 15:  # near-horizontal
                candidates.append((abs(angle), length, (x1, y1, x2, y2)))
        if candidates:
            # Longest near-horizontal line is most likely the horizon
            candidates.sort(key=lambda x: x[1], reverse=True)
            horizon_angle = candidates[0][0]
            horizon_line_coords = candidates[0][2]

    if horizon_angle is None:
        h_score = 7.0  # no strong horizontal — not penalised
    elif horizon_angle < 1.0:
        h_score = 10.0
    elif horizon_angle < 2.5:
        h_score = 8.0
    elif horizon_angle < 5.0:
        h_score = 5.0
    else:
        h_score = 2.0

    return CompositionResult(
        rule_of_thirds_score=rot_score,
        rule_of_thirds_label=rot_label,
        subject_x=subject_x,
        subject_y=subject_y,
        subject_found=subject_found,
        leading_lines_count=n_lines,
        leading_lines_score=ll_score,
        lines=lines[:8],  # cap at 8 for display
        horizon_angle=horizon_angle,
        horizon_score=h_score,
        horizon_line=horizon_line_coords,
    )


# ── Annotated overlay image ───────────────────────────────────────────────────

def _draw_overlay(bgr: np.ndarray, comp: CompositionResult) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    # Rule-of-thirds grid (thin white lines)
    grid_color = (200, 200, 200)
    for f in [1/3, 2/3]:
        cv2.line(out, (int(w * f), 0), (int(w * f), h), grid_color, 1)
        cv2.line(out, (0, int(h * f)), (w, int(h * f)), grid_color, 1)

    # Thirds intersection dots
    for tx in [1/3, 2/3]:
        for ty in [1/3, 2/3]:
            cv2.circle(out, (int(w * tx), int(h * ty)), 6, (255, 255, 255), 2)

    # Subject position
    if comp.subject_x is not None:
        sx, sy = int(comp.subject_x * w), int(comp.subject_y * h)
        color = (0, 255, 0) if comp.subject_found else (0, 200, 255)
        cv2.circle(out, (sx, sy), 12, color, 3)
        cv2.putText(out, "subject" if comp.subject_found else "saliency",
                    (sx + 14, sy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Leading lines
    for x1, y1, x2, y2 in comp.lines:
        cv2.line(out, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # Horizon line
    if comp.horizon_line:
        x1, y1, x2, y2 = comp.horizon_line
        color = (0, 255, 255) if (comp.horizon_angle or 0) < 2.5 else (0, 80, 255)
        cv2.line(out, (x1, y1), (x2, y2), color, 2)

    return out


# ── CLIP storytelling prompts ─────────────────────────────────────────────────

STORYTELLING_PROMPTS = [
    ("a photo that tells a compelling story",           "Storytelling"),
    ("a photo with strong emotional impact",            "Emotional impact"),
    ("a decisive moment caught in street photography",  "Decisive moment"),
    ("a cinematic, film-like photograph",               "Cinematic quality"),
    ("an original and unique photograph",               "Originality"),
    ("a technically perfect photograph",                "Technical quality"),
    ("a snapshot with no clear subject",                "Snapshot (negative)"),
    ("a clichéd or generic photograph",                 "Generic (negative)"),
]


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze(img_pil: Image.Image, max_dim: int = 1200) -> PhotoAnalysis:
    """Run all technical analyses on a PIL image. Returns a PhotoAnalysis."""
    # Resize for speed without losing detail
    img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    bgr = _to_bgr(img_pil)
    gray = _to_gray(bgr)

    sharpness = analyze_sharpness(gray)
    exposure = analyze_exposure(gray)
    noise = analyze_noise(gray)
    contrast = analyze_contrast(gray)
    composition = analyze_composition(bgr, gray)

    # Weighted overall technical score
    overall = (
        sharpness.score * 0.30 +
        exposure.score  * 0.20 +
        noise.score     * 0.10 +
        contrast.score  * 0.15 +
        composition.rule_of_thirds_score * 0.15 +
        composition.leading_lines_score  * 0.10
    )

    annotated = _draw_overlay(bgr, composition)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return PhotoAnalysis(
        sharpness=sharpness,
        exposure=exposure,
        noise=noise,
        contrast=contrast,
        composition=composition,
        overall_technical=round(_clamp(overall), 1),
        annotated_image=annotated_rgb,
    )
