#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip, json, math
import numpy as np
import openslide
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from skimage.draw import polygon as rrcc_polygon
from skimage.exposure import rescale_intensity
from skimage.filters import sobel, gaussian
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.measure import label as sk_label


# ============================
# USER SETTINGS
# ============================
SLIDE_TIFF = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002009.tif"
IN_GEOJSON = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP00002009.geojson.gz"

IDX = 66
PAD_PX = 64

# MGAC parameters (start here)
ITERATIONS = 120          # 80-200
SMOOTHING  = 2            # 1-3 (higher = smoother contour)
BALLOON    = -1.0         # negative shrinks, positive expands. Try -0.5, -1, -2
THRESH     = 0.25         # stopping threshold. Try 0.15-0.35

# Edge map construction
SIGMA = 1.0               # gaussian blur before edge map (0.5-2.0)

# Safeguard
MIN_KEEP_RATIO = 0.70

# Overlay
LINE_W = 6


# ============================
# HELPERS
# ============================
def load_feature_geometry(geojson_path, idx):
    if str(geojson_path).lower().endswith(".gz"):
        with gzip.open(geojson_path, "rt", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

    features = gj["features"] if isinstance(gj, dict) and gj.get("type") == "FeatureCollection" else gj
    feat = features[idx]
    geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
    geom = shape(geom_dict)

    if not geom.is_valid:
        geom = make_valid(geom)
        if not isinstance(geom, (Polygon, MultiPolygon)):
            geom = unary_union(geom)

    return geom


def rasterize_geom_to_mask(geom, x0, y0, w, h):
    mask = np.zeros((h, w), dtype=bool)

    def burn_poly(poly: Polygon):
        if poly.is_empty:
            return
        ext = np.asarray(poly.exterior.coords, dtype=np.float32)
        xs = ext[:, 0] - x0
        ys = ext[:, 1] - y0
        rr, cc = rrcc_polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = True

        for hole in poly.interiors:
            hxy = np.asarray(hole.coords, dtype=np.float32)
            xh = hxy[:, 0] - x0
            yh = hxy[:, 1] - y0
            rrh, cch = rrcc_polygon(yh, xh, shape=mask.shape)
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


def compute_od_sum(rgb_u8):
    I = (rgb_u8.astype(np.float32) + 1.0) / 255.0
    od = -np.log(I)
    return od[..., 0] + od[..., 1] + od[..., 2]


def mask_boundary(mask, width=1):
    er = ndi.binary_erosion(mask, structure=np.ones((3, 3), np.uint8))
    b = mask & (~er)
    if width > 1:
        b = ndi.binary_dilation(b, structure=np.ones((3, 3), np.uint8), iterations=int(width))
    return b


def keep_largest_component(mask):
    lab = sk_label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best = int(counts.argmax())
    return lab == best


# ============================
# RUN
# ============================
geom = load_feature_geometry(IN_GEOJSON, IDX)

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
ann_mask = rasterize_geom_to_mask(geom, x0, y0, w, h)

if not ann_mask.any():
    raise ValueError("Annotation mask is empty for this ROI.")

# --- Build an edge-stopping map from OD ---
od_sum = compute_od_sum(rgb)
od_rs = rescale_intensity(od_sum, in_range="image", out_range=(0, 1)).astype(np.float32)

# smooth OD to reduce noise
od_sm = gaussian(od_rs, sigma=SIGMA, preserve_range=True)

# gimage: small near edges, larger in flat regions (MGAC uses it to stop at edges)
gimage = inverse_gaussian_gradient(od_sm, alpha=100.0, sigma=1.0)

# --- Initial level set: your annotation ---
# MGAC expects 0/1 array
init_ls = ann_mask.astype(np.uint8)

# --- Run MGAC ---
ls = morphological_geodesic_active_contour(
    gimage,
    num_iter=ITERATIONS,
    init_level_set=init_ls,
    smoothing=SMOOTHING,
    threshold=THRESH,
    balloon=BALLOON
).astype(bool)

# --- Constrain: never expand outside original annotation ---
ls = ls & ann_mask

# --- Clean: keep largest component + fill holes (optional but usually helps)
ls = keep_largest_component(ls)
ls = ndi.binary_fill_holes(ls)

# --- Safeguard: avoid catastrophic shrink ---
orig_n = int(ann_mask.sum())
new_n  = int(ls.sum())
if new_n < MIN_KEEP_RATIO * orig_n:
    print("Safeguard triggered: too much shrink -> fallback to original")
    ls = ann_mask.copy()
    new_n = int(ls.sum())

print(f"IDX={IDX}")
print(f"orig pixels={orig_n}")
print(f"new pixels ={new_n}")
print(f"keep ratio ={new_n / max(1, orig_n):.3f}")
print(f"MGAC: iters={ITERATIONS}, smoothing={SMOOTHING}, balloon={BALLOON}, threshold={THRESH}, SIGMA={SIGMA}")

# --- Visualization ---
b_old = mask_boundary(ann_mask, width=LINE_W)
b_new = mask_boundary(ls, width=LINE_W)

overlay = rgb.copy()
overlay[b_old] = [255, 0, 0]   # red original
overlay[b_new] = [0, 255, 0]   # green new

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
ax = ax.ravel()

ax[0].imshow(rgb); ax[0].set_title("RGB"); ax[0].axis("off")
ax[1].imshow(od_rs, cmap="gray"); ax[1].set_title("OD sum (0..1)"); ax[1].axis("off")
ax[2].imshow(gimage, cmap="gray"); ax[2].set_title("gimage (edge-stopping)"); ax[2].axis("off")
ax[3].imshow(sobel(od_sm), cmap="gray"); ax[3].set_title("Sobel(OD smoothed)"); ax[3].axis("off")

ax[4].imshow(ann_mask, cmap="gray"); ax[4].set_title("Original mask"); ax[4].axis("off")
ax[5].imshow(ls, cmap="gray"); ax[5].set_title("MGAC result (constrained)"); ax[5].axis("off")
ax[6].imshow(overlay); ax[6].set_title("Overlay: red=old, green=new"); ax[6].axis("off")
ax[7].axis("off")

plt.tight_layout()
plt.show()
