#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone test script:
- For N random villus geometries (GeoJSON), crop ROI from WSI (OpenSlide)
- Rasterize the annotation mask (ann_mask)
- Compute HED, then S = H + E (smoothed)
- Build class2 candidates using an S "band-pass" inside annotation:
      keep pixels with t_low < S <= t_high
  where:
      t_low  = quantile(S_in, Q_LOW_EXCLUDE)   (exclude extreme low stain, often empty/white)
      t_high = quantile(S_in, q_high)          (test q_high values in TEST_Q_HIGH_LIST)
- Optional brightness veto (toggle True/False):
      remove brightest pixels (top BRIGHT_VETO_Q of L* inside ann)
- For each q_high:
      class2 = largest connected component of candidate, closing + fill holes
- Plot per q_high so you can pick the best setting.

Install deps:
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
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes


# =========================
# EDIT THESE
# =========================
SLIDE_TIFF   = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002010.tif"
GEOJSON_PATH = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP00002010.geojson.gz"

READ_LEVEL = 3
N_RANDOM = 5
RANDOM_SEED = None

PAD_PX_L0 = 96          # ROI padding in level-0 px (scaled to READ_LEVEL)
SIGMA_S_L0 = 6          # smoothing sigma for H/E/S in level-0 px (scaled)

# ---- S band-pass settings (inside ann_mask) ----
Q_LOW_EXCLUDE = 0.03
TEST_Q_HIGH_LIST = [0.10, 0.15, 0.20, 0.30, 0.40]

# ---- Optional brightness veto (inside ann_mask) ----
USE_BRIGHT_VETO = False
BRIGHT_VETO_Q = 0.98     # veto top 2% brightest inside annotation (L* >= quantile)
SIGMA_L_L0 = 6           # smoothing for L* (scaled like SIGMA_S_L0)

# ---- Postprocessing ----
CLOSE_RADIUS_L0 = 30      # optional closing on class2 (level-0 px; scaled)
MIN_C2_PIX = 200          # skip candidate if too small

SHOW_CONTOUR_ANN = True
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


def scale_px_int(px_l0: int, ds: float, min_val: int = 0) -> int:
    return max(min_val, int(round(px_l0 / ds)))


def scale_px_float(px_l0: float, ds: float, min_val: float = 0.5) -> float:
    return max(min_val, float(px_l0) / ds)


def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)

    ds = float(slide.level_downsamples[READ_LEVEL])  # level0 -> levelL factor
    Wl, Hl = slide.level_dimensions[READ_LEVEL]

    PAD_PX = scale_px_int(PAD_PX_L0, ds, min_val=1)
    SIGMA_S = scale_px_float(SIGMA_S_L0, ds, min_val=0.5)
    SIGMA_L = scale_px_float(SIGMA_L_L0, ds, min_val=0.5)

    CLOSE_RADIUS = scale_px_int(CLOSE_RADIUS_L0, ds, min_val=0)

    feats = load_geojson_features(GEOJSON_PATH)
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(feats)
    idxs = rng.choice(n, size=min(N_RANDOM, n), replace=False)

    for idx in idxs:
        feat = feats[idx]
        geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
        geom0 = shape(geom_dict)

        # Convert level-0 coords -> READ_LEVEL coords
        geomL = affinity.scale(geom0, xfact=1.0 / ds, yfact=1.0 / ds, origin=(0, 0))

        # ROI in READ_LEVEL coords
        minx, miny, maxx, maxy = geomL.bounds
        x0L = int(math.floor(minx)) - PAD_PX
        y0L = int(math.floor(miny)) - PAD_PX
        wL  = int(math.ceil(maxx - minx)) + 2 * PAD_PX
        hL  = int(math.ceil(maxy - miny)) + 2 * PAD_PX

        # Clamp
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

        # --- HED -> S = H + E ---
        rgb_f = rgb.astype(np.float32) / 255.0
        hed = rgb2hed(rgb_f)
        H = gaussian(hed[..., 0], sigma=SIGMA_S, preserve_range=True)
        E = gaussian(hed[..., 1], sigma=SIGMA_S, preserve_range=True)
        S = H + E

        S_in = S[ann_mask]
        if S_in.size < 10:
            continue

        # Band lower cutoff (exclude extreme low stain)
        t_low = float(np.quantile(S_in, Q_LOW_EXCLUDE))

        # Optional brightness veto mask
        if USE_BRIGHT_VETO:
            lab = rgb2lab(rgb_f)
            L = gaussian(lab[..., 0], sigma=SIGMA_L, preserve_range=True)  # brightness
            L_in = L[ann_mask]
            tL = float(np.quantile(L_in, BRIGHT_VETO_Q))
            bright_veto = (L >= tL)
        else:
            L = None
            tL = None
            bright_veto = None

        print(f"IDX={idx} ann_px={ann_mask.sum()} ds={ds:g} level={READ_LEVEL}  t_low={t_low:.3g}")

        for q_high in TEST_Q_HIGH_LIST:
            t_high = float(np.quantile(S_in, q_high))

            # Candidate: moderately low stain (exclude very lowest)
            c2_cand = ann_mask & (S > t_low) & (S <= t_high)

            # Optional: remove very bright pixels
            if USE_BRIGHT_VETO:
                c2_cand = c2_cand & (~bright_veto)

            if c2_cand.sum() < MIN_C2_PIX:
                print(f"  q_high={q_high:.2f}: cand too small ({c2_cand.sum()} px), skipping")
                continue

            # Largest component + close + fill holes -> solid class2
            class2 = keep_largest_component(c2_cand)
            if CLOSE_RADIUS > 0:
                class2 = binary_closing(class2, footprint=disk(CLOSE_RADIUS))
            class2 = binary_fill_holes(class2)

            class1 = ann_mask & (~class2)

            # ---- plot ----
            fig, ax = plt.subplots(1, 5, figsize=(26, 6))

            ax[0].imshow(rgb)
            if SHOW_CONTOUR_ANN:
                ax[0].imshow(ann_mask, alpha=0.15)
            ax[0].set_title(f"RGB + ann (IDX={idx})")
            ax[0].axis("off")

            ax[1].imshow(S, cmap="gray")
            ax[1].set_title(f"S=H+E (sigma={SIGMA_S:.2f})")
            ax[1].axis("off")

            ax[2].imshow(rgb)
            ax[2].imshow((ann_mask & (S <= t_low)), alpha=0.35)
            ax[2].set_title(f"Excluded low-S: S <= Q{int(Q_LOW_EXCLUDE*100)} (t_low={t_low:.3g})")
            ax[2].axis("off")

            if USE_BRIGHT_VETO:
                ax[3].imshow(rgb)
                ax[3].imshow((ann_mask & bright_veto), alpha=0.35)
                ax[3].set_title(f"Bright veto: L* >= Q{int(BRIGHT_VETO_Q*100)} (tL={tL:.3g})")
                ax[3].axis("off")
            else:
                ax[3].imshow(rgb)
                ax[3].set_title("Bright veto: OFF")
                ax[3].axis("off")

            ax[4].imshow(rgb)
            ax[4].imshow(class2, alpha=0.35)
            ax[4].set_title(
                f"class2: band S in ({t_low:.3g}, Q{int(q_high*100)}={t_high:.3g}]"
                f"\n(c2={class2.sum()} px, c1={class1.sum()} px)"
            )
            ax[4].axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
