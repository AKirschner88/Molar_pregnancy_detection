#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, gzip, math
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import affinity

from skimage.draw import polygon as rrcc_polygon
from skimage.color import rgb2hed
from skimage.filters import gaussian
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from skimage.measure import find_contours, label as cc_label
from scipy.ndimage import binary_fill_holes


# ============================================================
# EDIT THESE
# ============================================================
SLIDE_TIFF   = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002010.tif"
GEOJSON_PATH = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002010_adjusted.geojson"

# Show ~20 random geometries each run
N_RANDOM = 20
RANDOM_SEED = None  # None => different each run; set int for reproducible

READ_LEVEL = 2

# Parameters (assumed tuned at level-0 pixels; will be scaled to READ_LEVEL)
PAD_PX_L0 = 64
SIGMA_H_L0 = 6

# Core (class2) selection by "low stain": low H AND low E inside villus
QH_LOW = 0.8
QE_LOW = 0.8

MIN_SEED_AREA_L0 = 500  # will be scaled to level

# Morphology (tuned at level-0; will be scaled)
OPEN_RADIUS_L0 = 10
CLOSE_RADIUS_L0 = 50

SMOOTH_WIN_L0 = 0  # contour smoothing window (will be scaled & forced odd)
GEOM_LW = 8
# ============================================================


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
    raise ValueError("Unsupported GeoJSON format.")


def rasterize_geom_to_mask(geom, x0, y0, w, h):
    """
    Rasterize geometry into a boolean mask with shape (h, w).
    Important: geom coordinates and (x0,y0,w,h) must be in the SAME coordinate system.
    """
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
    lab = cc_label(mask)
    if lab.max() == 0:
        return mask
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    return lab == areas.argmax()


def smooth_closed_contour(c_rc: np.ndarray, win: int = 21) -> np.ndarray:
    """Smooth a closed contour (row,col) using a circular moving average."""
    if win < 3:
        return c_rc
    if win % 2 == 0:
        win += 1

    n = c_rc.shape[0]
    if n < win:
        return c_rc

    pad = win // 2
    cpad = np.vstack([c_rc[-pad:], c_rc, c_rc[:pad]])

    kernel = np.ones(win, dtype=np.float64) / win
    r = np.convolve(cpad[:, 0], kernel, mode="valid")
    c = np.convolve(cpad[:, 1], kernel, mode="valid")

    return np.stack([r, c], axis=1)


def filled_from_outer_contour(seed_mask: np.ndarray, shape_hw, smooth_win: int = 21) -> np.ndarray:
    """Take the longest contour of seed_mask, smooth it, and fill it."""
    cs = find_contours(seed_mask.astype(np.uint8), level=0.5)
    if len(cs) == 0:
        return np.zeros(shape_hw, dtype=bool)

    c = max(cs, key=lambda x: x.shape[0])  # (row, col)
    c_sm = smooth_closed_contour(c, win=smooth_win)

    rr, cc = rrcc_polygon(c_sm[:, 0], c_sm[:, 1], shape=shape_hw)
    out = np.zeros(shape_hw, dtype=bool)
    out[rr, cc] = True
    out = binary_fill_holes(out)
    return out


def scale_px_int(px_l0: int, ds: float, min_val: int = 1) -> int:
    """Scale a level-0 pixel count to READ_LEVEL pixels."""
    return max(min_val, int(round(px_l0 / ds)))


def scale_px_float(px_l0: float, ds: float, min_val: float = 0.5) -> float:
    """Scale a level-0 pixel float (e.g. sigma) to READ_LEVEL pixels."""
    return max(min_val, float(px_l0) / ds)


def force_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)

    ds = float(slide.level_downsamples[READ_LEVEL])  # level0 -> levelL factor
    Wl, Hl = slide.level_dimensions[READ_LEVEL]

    # Scale parameters to READ_LEVEL
    PAD_PX = scale_px_int(PAD_PX_L0, ds, min_val=1)
    SIGMA_H = scale_px_float(SIGMA_H_L0, ds, min_val=0.5)

    OPEN_RADIUS = scale_px_int(OPEN_RADIUS_L0, ds, min_val=0)
    CLOSE_RADIUS = scale_px_int(CLOSE_RADIUS_L0, ds, min_val=0)

    MIN_SEED_AREA = max(10, int(round(MIN_SEED_AREA_L0 / (ds * ds))))
    SMOOTH_WIN = force_odd(max(5, int(round(SMOOTH_WIN_L0 / ds))))

    feats = load_geojson_features(GEOJSON_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    n_feats = len(feats)
    k = min(N_RANDOM, n_feats)
    rand_indices = rng.choice(n_feats, size=k, replace=False)

    for feature_index in rand_indices:
        feat = feats[feature_index]
        geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
        geom0 = shape(geom_dict)

        # Convert geometry from level-0 coords to READ_LEVEL coords
        geomL = affinity.scale(geom0, xfact=1.0 / ds, yfact=1.0 / ds, origin=(0, 0))
        print("FEATURE_INDEX:", feature_index, "is_valid:", geom0.is_valid)

        # ROI in READ_LEVEL coordinates
        minx, miny, maxx, maxy = geomL.bounds
        x0L = int(math.floor(minx)) - PAD_PX
        y0L = int(math.floor(miny)) - PAD_PX
        wL = int(math.ceil(maxx - minx)) + 2 * PAD_PX
        hL = int(math.ceil(maxy - miny)) + 2 * PAD_PX

        # Clamp in READ_LEVEL coordinates
        x0L = max(0, x0L)
        y0L = max(0, y0L)
        wL = min(wL, Wl - x0L)
        hL = min(hL, Hl - y0L)
        if wL <= 0 or hL <= 0:
            print("  -> skipped (empty ROI after clamping)")
            continue

        # OpenSlide read_region location is in level-0 coordinates, size is in level coordinates
        x0_0 = int(round(x0L * ds))
        y0_0 = int(round(y0L * ds))

        rgb = np.array(
            slide.read_region((x0_0, y0_0), READ_LEVEL, (wL, hL)).convert("RGB"),
            dtype=np.uint8,
        )

        # Rasterize annotation in READ_LEVEL coords
        ann_mask = rasterize_geom_to_mask(geomL, x0L, y0L, wL, hL)
        if ann_mask.sum() == 0:
            print("  -> skipped (empty ann_mask)")
            continue

        # ----------------------------
        # CLASS2 (core) via low-H / low-E (pale)
        # ----------------------------
        rgb_f = rgb.astype(np.float32) / 255.0
        hed = rgb2hed(rgb_f)
        H = hed[..., 0]
        E = hed[..., 1]

        H_smooth = gaussian(H, sigma=SIGMA_H, preserve_range=True)
        E_smooth = gaussian(E, sigma=SIGMA_H, preserve_range=True)

        Hv = H_smooth[ann_mask]
        Ev = E_smooth[ann_mask]

        # Low-stain thresholds inside the villus
        tH = np.quantile(Hv, QH_LOW)
        tE = np.quantile(Ev, QE_LOW)

        pale = ann_mask & (H_smooth <= tH) & (E_smooth <= tE)

        # Seed cleanup -> one core blob
        seed2_raw = pale
        seed2 = keep_largest_component(seed2_raw)
        seed2 = remove_small_objects(seed2, MIN_SEED_AREA)

        if OPEN_RADIUS > 0:
            seed2 = binary_opening(seed2, footprint=disk(OPEN_RADIUS))
        if CLOSE_RADIUS > 0:
            seed2 = binary_closing(seed2, footprint=disk(CLOSE_RADIUS))

        seed2 = binary_fill_holes(seed2)

        # Solid filled interior region
        class2 = filled_from_outer_contour(seed2, ann_mask.shape, smooth_win=SMOOTH_WIN) & ann_mask

        # Class1 is remainder (may be empty)
        class1 = ann_mask & (~class2)

        # If class1 is tiny, zero it out (optional)
        MIN_CLASS1_FRAC = 0.01
        if class1.sum() / max(1, ann_mask.sum()) < MIN_CLASS1_FRAC:
            class1[:] = False

        # Plot
        fig, ax = plt.subplots(1, 4, figsize=(24, 6))

        ax[0].imshow(rgb)
        ax[0].set_title(f"RGB (ROI) - level {READ_LEVEL}")
        ax[0].axis("off")

        ax[1].imshow(rgb)
        ax[1].imshow(seed2_raw, alpha=0.35)
        ax[1].set_title(f"Seed2 raw = pale (QH={QH_LOW}, QE={QE_LOW})")
        ax[1].axis("off")

        ax[2].imshow(rgb)
        ax[2].contour(ann_mask.astype(np.uint8), levels=[0.5], linewidths=GEOM_LW)
        ax[2].imshow(class2, alpha=0.35)
        ax[2].contour(class2.astype(np.uint8), levels=[0.5], linewidths=2)
        ax[2].set_title("Class2 = solid filled core")
        ax[2].axis("off")

        ax[3].imshow(rgb)
        ax[3].contour(ann_mask.astype(np.uint8), levels=[0.5], linewidths=GEOM_LW)
        ax[3].imshow(class1, alpha=0.35)
        ax[3].set_title("Class1 = ann_mask \\ class2 (may be empty)")
        ax[3].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
