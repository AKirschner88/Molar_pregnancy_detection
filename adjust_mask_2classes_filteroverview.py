#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone FILTER/THRESHOLD OVERVIEW for villus core selection.

What it does:
- Opens a whole-slide image (OpenSlide)
- Loads GeoJSON annotations (FeatureCollection or list)
- Picks N random geometries
- For each: crops ROI, rasterizes villus mask, and shows an overview grid of:
  RGB, H/E (HED), S=H+E, ratio R, LAB L*, plus Otsu + quantile masks,
  plus "SOLID" versions (largest CC + fill) so you can judge core selection.

Deps:
pip install openslide-python shapely scikit-image scipy matplotlib
"""

import json
import gzip
import math
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import affinity

from skimage.draw import polygon as rrcc_polygon
from skimage.color import rgb2hed, rgb2lab
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes


# =========================
# EDIT THESE
# =========================
SLIDE_TIFF   = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002001.tif"
GEOJSON_PATH = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002001_adjusted.geojson"

READ_LEVEL = 3
N_RANDOM = 5
RANDOM_SEED = None  # None => different each run; set int for reproducible
PAD_PX_L0 = 96

# Overview settings
SIGMA = 2.0                 # smoothing for maps (in READ_LEVEL pixels)
Q = 0.30                    # quantile for "low" masks
SOLID_CLOSE_RADIUS = 5      # disk radius for solidification
MIN_CANDIDATE_PIX = 200     # fallback if mask becomes too small
# =========================


def load_geojson_features(path: str):
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        return gj["features"]
    if isinstance(gj, list):
        return gj
    raise ValueError("Unsupported GeoJSON format (expected FeatureCollection or list).")


def rasterize_geom_to_mask(geom, x0, y0, w, h):
    """Rasterize shapely Polygon/MultiPolygon into boolean mask (h,w)."""
    mask = np.zeros((h, w), dtype=bool)

    def burn_poly(poly: Polygon):
        if poly.is_empty:
            return
        ext = np.asarray(poly.exterior.coords, dtype=np.float32)
        rr, cc = rrcc_polygon(ext[:, 1] - y0, ext[:, 0] - x0, shape=mask.shape)
        mask[rr, cc] = True
        for hole in poly.interiors:
            hxy = np.asarray(hole.coords, dtype=np.float32)
            rrh, cch = rrcc_polygon(hxy[:, 1] - y0, hxy[:, 0] - x0, shape=mask.shape)
            mask[rrh, cch] = False

    if isinstance(geom, Polygon):
        burn_poly(geom)
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            burn_poly(p)
    else:
        g2 = geom.buffer(0)
        if isinstance(g2, Polygon):
            burn_poly(g2)
        elif isinstance(g2, MultiPolygon):
            for p in g2.geoms:
                burn_poly(p)

    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask)
    if lab.max() == 0:
        return mask
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    return lab == areas.argmax()


def solidify(mask: np.ndarray, close_radius: int = 5) -> np.ndarray:
    """Largest CC + closing + fill holes."""
    m = keep_largest_component(mask)
    if close_radius > 0:
        m = binary_closing(m, footprint=disk(close_radius))
    m = binary_fill_holes(m)
    return m


def safe_otsu(vals: np.ndarray):
    if vals.size < 2:
        return float(vals.mean()) if vals.size else 0.0
    # Otsu can fail if constant; guard it
    vmin, vmax = float(vals.min()), float(vals.max())
    if abs(vmax - vmin) < 1e-12:
        return vmin
    return float(threshold_otsu(vals))


def show_filter_overview(rgb: np.ndarray, ann_mask: np.ndarray, sigma: float, q: float,
                         solid_close_radius: int, min_candidate_pix: int):
    """
    Displays overview grid for one ROI.
    """
    rgb_f = rgb.astype(np.float32) / 255.0

    hed = rgb2hed(rgb_f)
    H = gaussian(hed[..., 0], sigma=sigma, preserve_range=True)
    E = gaussian(hed[..., 1], sigma=sigma, preserve_range=True)

    S = H + E
    R = H / (H + E + 1e-6)

    lab = rgb2lab(rgb_f)
    L = gaussian(lab[..., 0], sigma=sigma, preserve_range=True)  # brightness

    # inside-only values
    H_in = H[ann_mask]; E_in = E[ann_mask]; S_in = S[ann_mask]; R_in = R[ann_mask]; L_in = L[ann_mask]
    if ann_mask.sum() == 0:
        return

    # Otsu thresholds inside mask
    tH = safe_otsu(H_in)
    tE = safe_otsu(E_in)
    tS = safe_otsu(S_in)
    tR = safe_otsu(R_in)
    tL = safe_otsu(L_in)

    # Otsu masks
    m_H_high = ann_mask & (H >= tH)
    m_E_high = ann_mask & (E >= tE)
    m_S_low  = ann_mask & (S <= tS)
    m_R_high = ann_mask & (R >= tR)
    m_L_high = ann_mask & (L >= tL)

    # Quantile masks (inside-only)
    qH = float(np.quantile(H_in, q))
    qE = float(np.quantile(E_in, q))
    qS = float(np.quantile(S_in, q))
    qR = float(np.quantile(R_in, 1.0 - q))  # "high" ratio often interesting
    qL = float(np.quantile(L_in, 1.0 - q))  # "high" brightness

    m_S_q_low = ann_mask & (S <= qS)
    m_H_q_low = ann_mask & (H <= qH)
    m_E_q_low = ann_mask & (E <= qE)
    m_R_q_high = ann_mask & (R >= qR)
    m_L_q_high = ann_mask & (L >= qL)

    # Low-H AND Low-E (pale candidate)
    m_lowHE = ann_mask & (H <= qH) & (E <= qE)

    # Fallback if candidate too small (for display, keep it visible)
    def ensure(m):
        return ann_mask if m.sum() < min_candidate_pix else m

    m_S_low = ensure(m_S_low)
    m_lowHE = ensure(m_lowHE)
    m_L_high = ensure(m_L_high)

    # SOLID versions (core candidates)
    solid_S_low  = solidify(m_S_low,  close_radius=solid_close_radius)
    solid_lowHE  = solidify(m_lowHE,  close_radius=solid_close_radius)
    solid_L_high = solidify(m_L_high, close_radius=solid_close_radius)

    # Plot grid
    items = [
        ("RGB", rgb),
        ("H (smoothed)", H),
        ("E (smoothed)", E),
        ("S=H+E (low=pale)", S),
        ("R=H/(H+E)", R),
        ("L* (bright)", L),

        (f"Otsu: H high (t={tH:.3g})", m_H_high),
        (f"Otsu: E high (t={tE:.3g})", m_E_high),
        (f"Otsu: S low  (t={tS:.3g})", m_S_low),
        (f"Otsu: R high (t={tR:.3g})", m_R_high),
        (f"Otsu: L high (t={tL:.3g})", m_L_high),

        (f"Q{int(q*100)}: S low", m_S_q_low),
        (f"Q{int(q*100)}: H low", m_H_q_low),
        (f"Q{int(q*100)}: E low", m_E_q_low),
        (f"Q{int((1-q)*100)}: R high", m_R_q_high),
        (f"Q{int((1-q)*100)}: L high", m_L_q_high),

        ("Low H & Low E (pale)", m_lowHE),
        ("SOLID from S low", solid_S_low),
        ("SOLID from lowH&lowE", solid_lowHE),
        ("SOLID from L high", solid_L_high),
    ]

    cols = 4
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (title, img) in zip(axes, items):
        if img.ndim == 2 and img.dtype != bool:
            ax.imshow(img)
        elif img.ndim == 2 and img.dtype == bool:
            ax.imshow(rgb)
            ax.imshow(img, alpha=0.35)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)
    ds = float(slide.level_downsamples[READ_LEVEL])
    Wl, Hl = slide.level_dimensions[READ_LEVEL]

    feats = load_geojson_features(GEOJSON_PATH)
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(feats)
    idxs = rng.choice(n, size=min(N_RANDOM, n), replace=False)

    pad = max(1, int(round(PAD_PX_L0 / ds)))

    for feature_index in idxs:
        feat = feats[feature_index]
        geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
        geom0 = shape(geom_dict)

        # scale geometry to READ_LEVEL coords
        geomL = affinity.scale(geom0, xfact=1.0 / ds, yfact=1.0 / ds, origin=(0, 0))

        # ROI in READ_LEVEL coords
        minx, miny, maxx, maxy = geomL.bounds
        x0L = int(math.floor(minx)) - pad
        y0L = int(math.floor(miny)) - pad
        wL = int(math.ceil(maxx - minx)) + 2 * pad
        hL = int(math.ceil(maxy - miny)) + 2 * pad

        # clamp
        x0L = max(0, x0L)
        y0L = max(0, y0L)
        wL = min(wL, Wl - x0L)
        hL = min(hL, Hl - y0L)
        if wL <= 0 or hL <= 0:
            continue

        # read_region expects level-0 location
        x0_0 = int(round(x0L * ds))
        y0_0 = int(round(y0L * ds))

        rgb = np.array(
            slide.read_region((x0_0, y0_0), READ_LEVEL, (wL, hL)).convert("RGB"),
            dtype=np.uint8
        )
        ann_mask = rasterize_geom_to_mask(geomL, x0L, y0L, wL, hL)

        if ann_mask.sum() == 0:
            continue

        print(f"IDX={feature_index}  ann_mask_px={ann_mask.sum()}  level={READ_LEVEL}  ds={ds:g}")
        show_filter_overview(
            rgb=rgb,
            ann_mask=ann_mask,
            sigma=SIGMA,
            q=Q,
            solid_close_radius=SOLID_CLOSE_RADIUS,
            min_candidate_pix=MIN_CANDIDATE_PIX,
        )


if __name__ == "__main__":
    main()
