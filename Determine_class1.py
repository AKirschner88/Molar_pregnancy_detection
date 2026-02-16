#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OD-based inward rim thickness estimation for ONE villus annotation.

What it does:
- Loads TIFF (OpenSlide) + GeoJSON(.gz)
- Picks one annotation (IDX)
- Crops ROI around it
- Builds villus mask
- Converts ROI RGB -> OD sum image
- Estimates inward normals along the boundary (via distance transform gradient)
- Samples OD along inward normal lines
- Detects thickness as the first strong OD-gradient peak within a range
- Reports thickness distribution + coverage
- Saves/plots debug figures

Deps:
pip install openslide-python shapely scikit-image scipy matplotlib tqdm
"""

import json, gzip, math
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union

from skimage.draw import polygon as rrcc_polygon
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, sobel
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation



# -----------------------------
# USER SETTINGS
# -----------------------------
SLIDE_TIFF = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002001.tif"
IN_GEOJSON = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP00002001_bgclampedV8.geojson"

IDX = 0          # which annotation to test
PAD_PX = 128     # ROI padding around annotation bbox

# Sampling / detection
R_IN = 150       # sample inward length (px)
STEP = 1         # sampling step (px)
SMOOTH_SIGMA = 2.0   # smooth 1D OD profile
GRAD_MIN = 0.015     # gradient peak threshold (tune)
MIN_T = 3            # min thickness px
MAX_T = 120          # max thickness px

# Boundary sampling density (use every Nth boundary pixel)
BOUNDARY_STRIDE = 5


# -----------------------------
# HELPERS
# -----------------------------
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


def rgb_to_od_sum(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    od = -np.log((rgb + 1.0) / 255.0)   # avoid log(0)
    return od.sum(axis=2).astype(np.float32)


def load_one_geom(path, idx):
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

    features = gj["features"] if isinstance(gj, dict) and gj.get("type") == "FeatureCollection" else gj
    feat = features[idx]
    geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
    geom = shape(geom_dict)

    if not geom.is_valid:
        geom = make_valid(geom)
        if not isinstance(geom, Polygon):
            geom = unary_union(geom)

    return geom


def estimate_normals_from_mask(mask: np.ndarray):
    """
    Compute inward unit normals using gradient of distance transform.
    For pixels inside mask, DT increases inward.
    Normal approx: n = grad(DT) normalized.
    """
    dt = distance_transform_edt(mask).astype(np.float32)
    gx = sobel(dt, axis=1)  # x-gradient (cols)
    gy = sobel(dt, axis=0)  # y-gradient (rows)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    nx = gx / mag
    ny = gy / mag
    return nx, ny, dt


def sample_profile(img: np.ndarray, r0: int, c0: int, nx: float, ny: float,
                   r_in=150, step=1):
    """
    Sample img along inward normal starting at boundary point (r0,c0).
    img indexed as [row, col]. nx,ny in col/row coords.
    """
    rs = r0 + np.arange(0, r_in+1, step) * ny
    cs = c0 + np.arange(0, r_in+1, step) * nx

    # bilinear sampling
    H, W = img.shape
    rs = np.clip(rs, 0, H-1)
    cs = np.clip(cs, 0, W-1)

    r0f = np.floor(rs).astype(int)
    c0f = np.floor(cs).astype(int)
    r1f = np.minimum(r0f + 1, H-1)
    c1f = np.minimum(c0f + 1, W-1)

    dr = rs - r0f
    dc = cs - c0f

    v00 = img[r0f, c0f]
    v01 = img[r0f, c1f]
    v10 = img[r1f, c0f]
    v11 = img[r1f, c1f]

    vals = (1-dr)*((1-dc)*v00 + dc*v01) + dr*((1-dc)*v10 + dc*v11)
    return vals.astype(np.float32)


def detect_thickness_from_profile(p: np.ndarray):
    ps = gaussian_filter1d(p, SMOOTH_SIGMA)
    g = np.abs(np.gradient(ps))

    lo = int(MIN_T)
    hi = int(min(MAX_T, len(g) - 1))
    if hi <= lo + 2:
        return None, 0.0

    # pick the strongest gradient peak in [lo, hi)
    i = lo + int(np.argmax(g[lo:hi]))
    pk = float(g[i])

    if pk < GRAD_MIN:
        return None, pk

    return i, pk


# -----------------------------
# MAIN
# -----------------------------
def main():
    geom = load_one_geom(IN_GEOJSON, IDX)

    slide = openslide.OpenSlide(SLIDE_TIFF)
    W, H = slide.dimensions

    minx, miny, maxx, maxy = geom.bounds
    x0 = int(math.floor(minx)) - PAD_PX
    y0 = int(math.floor(miny)) - PAD_PX
    w  = int(math.ceil(maxx - minx)) + 2 * PAD_PX
    h  = int(math.ceil(maxy - miny)) + 2 * PAD_PX

    x0 = max(0, x0); y0 = max(0, y0)
    w = min(w, W - x0); h = min(h, H - y0)

    rgb = np.array(slide.read_region((x0, y0), 0, (w, h)).convert("RGB"), dtype=np.uint8)
    od = rgb_to_od_sum(rgb)

    mask = rasterize_geom_to_mask(geom, x0, y0, w, h)
    if mask.sum() == 0:
        raise RuntimeError("Mask empty after rasterization.")

    # boundary pixels
    boundary = mask & (~binary_erosion(mask))

    # normals (inward)
    nx, ny, dt = estimate_normals_from_mask(mask)

    # sample boundary points
    br, bc = np.where(boundary)
    if len(br) == 0:
        raise RuntimeError("No boundary pixels found.")

    # subsample for speed
    sel = np.arange(0, len(br), BOUNDARY_STRIDE)
    br = br[sel]; bc = bc[sel]

    thickness = []
    peaks = []
    detected = np.zeros_like(mask, dtype=bool)

    for r, c in zip(br, bc):
        # use inward normal at this boundary pixel
        nxc = float(nx[r, c])
        nyc = float(ny[r, c])

        # if normal is degenerate, skip
        if not np.isfinite(nxc) or not np.isfinite(nyc):
            continue
        if abs(nxc) + abs(nyc) < 1e-3:
            continue

        prof = sample_profile(od, r, c, nxc, nyc, r_in=R_IN, step=STEP)
        t, pk = detect_thickness_from_profile(prof)
        if t is None:
            continue

        thickness.append(t)
        peaks.append(pk)

        # mark detected location (for overlay)
        rr = int(round(r + t * nyc))
        cc = int(round(c + t * nxc))
        if 0 <= rr < detected.shape[0] and 0 <= cc < detected.shape[1]:
            detected[rr, cc] = True

    thickness = np.array(thickness, dtype=np.float32)

    coverage = (len(thickness) / max(1, len(br))) * 100.0
    print(f"Boundary points tested: {len(br)}")
    print(f"Detected thickness points: {len(thickness)}")
    print(f"Coverage: {coverage:.1f}%")

    if len(thickness) > 0:
        print(f"Thickness px: median={np.median(thickness):.1f}, "
              f"p25={np.percentile(thickness,25):.1f}, p75={np.percentile(thickness,75):.1f}")
    else:
        print("No thickness detected with current thresholds.")

    # --- plots ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    ax[0].imshow(rgb)
    ax[0].set_title("ROI RGB")
    ax[0].axis("off")

    ax[1].imshow(od, cmap="gray")
    ax[1].set_title("OD sum")
    ax[1].axis("off")

    LINE_W = 10  # pixels (increase to 5+ if needed)
    
    boundary_thick = binary_dilation(boundary, iterations=LINE_W)
    detected_thick = binary_dilation(detected, iterations=LINE_W)
    
    overlay = rgb.copy()
    overlay[boundary_thick] = [255, 0, 0]
    overlay[detected_thick] = [0, 255, 0]

    ax[2].imshow(overlay)
    ax[2].set_title("Red=boundary, Green=detected inner edge")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    if len(thickness) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(thickness, bins=30)
        plt.title("Thickness distribution (px)")
        plt.xlabel("Thickness (px)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()