#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test: "background" (low stain) detection *WITHIN* annotation mask only.

- S = H + E (HED), low S = pale/unstained
- Threshold is computed only from pixels inside ann
- Output overlay shows bg_in_ann = ann & (S <= threshold)
"""

import json, gzip, math
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import affinity

from skimage.draw import polygon as rrcc_polygon
from skimage.color import rgb2hed
from skimage.filters import gaussian, threshold_otsu


# =========================
# EDIT THESE
# =========================
SLIDE_TIFF   = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002002.tif"
GEOJSON_PATH = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP00002002.geojson.gz"

READ_LEVEL = 2
N_RANDOM = 10
RANDOM_SEED = None
PAD_PX_L0 = 96

SIGMA = 2.0  # smoothing in READ_LEVEL pixels

# Show both methods:
USE_OTSU = True
Q_LIST = [0.05, 0.10, 0.15]  # low tail of S within ann
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


def safe_otsu(vals: np.ndarray) -> float:
    if vals.size < 2:
        return float(vals.mean()) if vals.size else 0.0
    vmin, vmax = float(vals.min()), float(vals.max())
    if abs(vmax - vmin) < 1e-12:
        return vmin
    return float(threshold_otsu(vals))


def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)
    ds = float(slide.level_downsamples[READ_LEVEL])
    Wl, Hl = slide.level_dimensions[READ_LEVEL]

    feats = load_geojson_features(GEOJSON_PATH)
    rng = np.random.default_rng(RANDOM_SEED)
    idxs = rng.choice(len(feats), size=min(N_RANDOM, len(feats)), replace=False)

    pad = max(1, int(round(PAD_PX_L0 / ds)))

    for idx in idxs:
        feat = feats[idx]
        geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
        geom0 = shape(geom_dict)
        geomL = affinity.scale(geom0, xfact=1.0 / ds, yfact=1.0 / ds, origin=(0, 0))

        # ROI in READ_LEVEL coords
        minx, miny, maxx, maxy = geomL.bounds
        x0L = int(math.floor(minx)) - pad
        y0L = int(math.floor(miny)) - pad
        wL  = int(math.ceil(maxx - minx)) + 2 * pad
        hL  = int(math.ceil(maxy - miny)) + 2 * pad

        # clamp
        x0L = max(0, x0L); y0L = max(0, y0L)
        wL = min(wL, Wl - x0L); hL = min(hL, Hl - y0L)
        if wL <= 0 or hL <= 0:
            continue

        # read_region location in level-0 coords
        x0_0 = int(round(x0L * ds))
        y0_0 = int(round(y0L * ds))

        rgb = np.array(
            slide.read_region((x0_0, y0_0), READ_LEVEL, (wL, hL)).convert("RGB"),
            dtype=np.uint8
        )

        ann = rasterize_geom_to_mask(geomL, x0L, y0L, wL, hL)
        if ann.sum() == 0:
            continue

        # --- HED -> S = H + E ---
        rgb_f = rgb.astype(np.float32) / 255.0
        hed = rgb2hed(rgb_f)
        H = gaussian(hed[..., 0], sigma=SIGMA, preserve_range=True)
        E = gaussian(hed[..., 1], sigma=SIGMA, preserve_range=True)
        S = H + E  # low = pale/unstained

        S_in = S[ann]

        # Thresholds computed ONLY inside ann
        panels = []

        if USE_OTSU:
            t = safe_otsu(S_in)
            bg_in_ann = ann & (S <= t)
            panels.append((f"Otsu on ann: remove S <= {t:.3g}", bg_in_ann))

        for q in Q_LIST:
            tq = float(np.quantile(S_in, q))
            bg_in_ann_q = ann & (S <= tq)
            panels.append((f"Q{int(q*100)} on ann: remove S <= {tq:.3g}", bg_in_ann_q))

        # --- Plot ---
        cols = 2
        rows = 1 + int(math.ceil(len(panels) / cols))
        fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        ax = np.array(ax).reshape(-1)

        ax[0].imshow(rgb)
        ax[0].imshow(ann, alpha=0.20)
        ax[0].set_title(f"RGB + ann overlay (IDX={idx})")
        ax[0].axis("off")

        ax[1].imshow(S, cmap="gray")
        ax[1].set_title("S = H+E (gray, low=pale)")
        ax[1].axis("off")

        for i, (title, m) in enumerate(panels):
            a = ax[2 + i]
            a.imshow(rgb)
            a.imshow(m, alpha=0.35)
            a.set_title(title)
            a.axis("off")

        for j in range(2 + len(panels), len(ax)):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
