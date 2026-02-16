#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import numpy as np
import gzip
import openslide
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
import scipy.ndimage as ndi

from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from skimage.color import rgb2lab
from skimage.draw import polygon as rrcc_polygon
from skimage.measure import find_contours, label as sk_label
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, reconstruction, disk
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_propagation, distance_transform_edt, binary_dilation


# -----------------------------
# USER SETTINGS
# -----------------------------
SLIDE_TIFF = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002009.tif"
IN_GEOJSON = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP00002009.geojson.gz"

# Output split:
OUT_GEOJSON_ADJUSTED = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002009_adjusted.geojson"
OUT_GEOJSON_FAILED   = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002009_failed.geojson"

# ROI / geometry handling
PAD_PX = 64
MIN_OBJ_AREA = 0
MIN_HOLE_AREA = 2000
SMOOTH_RADIUS = 6
SMOOTH_BUF = 2
SIMPLIFY_TOL = 1.5

# --- Background detection in LAB ---
BG_L_MIN = 99.0
BG_CHROMA_MAX = 4.0

# --- OD barrier settings (optional but recommended) ---
USE_OD_BARRIER = True
OD_Q_EDGE = 0.50
OD_USE_BOUNDARY_BAND = True
OD_BAND_PX = 25
OD_SMOOTH_BARRIER_R = 15
OD_SMOOTH_BG_EDGE_R = 10

# --- Bridge/thickness filters (optional) ---
BRIDGE_R = 10     # erosion+reconstruction on bg_edge (removes thin corridors)
THICK_MIN = 0     # thickness-core filter on bg_edge (0 disables)

# --- Repair (after background removal) ---
R_CLOSE = 1       # small closing on selected component (0 disables)

# --- PASS/FAIL safeguards (this decides adjusted vs failed) ---
MIN_KEEP_RATIO = 0.5

SMALL_ORIG_PX = 100_000
MAX_REMOVE_SMALL = 20_000


# -----------------------------
# HELPERS
# -----------------------------
def geom_to_xy_bounds_px(geom):
    minx, miny, maxx, maxy = geom.bounds
    return int(math.floor(minx)), int(math.floor(miny)), int(math.ceil(maxx)), int(math.ceil(maxy))


def clamp_roi(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    if w <= 2 or h <= 2:
        return None
    return x, y, w, h


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


def background_mask_white(rgb_u8, BG_L_MIN, BG_CHROMA_MAX):
    rgb = rgb_u8.astype(np.float32) / 255.0
    lab = rgb2lab(rgb)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a*a + b*b)
    return (L >= BG_L_MIN) & (chroma <= BG_CHROMA_MAX)


def bg_touching_annotation_boundary(bg, ann_mask):
    ann_er = binary_erosion(ann_mask, structure=np.ones((3, 3), np.uint8))
    ann_boundary = ann_mask & (~ann_er)

    allowed = bg & ann_mask
    seeds = ann_boundary & allowed
    return binary_propagation(seeds, mask=allowed)


def bg_touching_annotation_boundary_odbarrier(
    rgb_u8: np.ndarray,
    bg: np.ndarray,
    ann_mask: np.ndarray,
    q_edge: float = 0.90,
    use_boundary_band: bool = True,
    band_px: int = 25,
    smooth_barrier_r: int = 0,
    smooth_bg_edge_r: int = 0,
    close_barrier: bool = True,
):
    # annotation boundary seeds
    ann_er = ndi.binary_erosion(ann_mask, structure=np.ones((3, 3), np.uint8))
    ann_boundary = ann_mask & (~ann_er)

    # OD sum
    I = (rgb_u8.astype(np.float32) + 1.0) / 255.0
    od = -np.log(I)
    od_sum = od[..., 0] + od[..., 1] + od[..., 2]

    # grad magnitude on OD sum
    gx = ndi.sobel(od_sum, axis=1)
    gy = ndi.sobel(od_sum, axis=0)
    grad = np.hypot(gx, gy)

    # threshold estimate region
    if use_boundary_band:
        inner = ndi.binary_erosion(
            ann_mask, structure=np.ones((3, 3), np.uint8),
            iterations=max(1, int(band_px))
        )
        band = ann_mask & (~inner)
        vals = grad[band]
        if vals.size < 1000:
            vals = grad[ann_mask]
    else:
        vals = grad[ann_mask]

    if vals.size == 0:
        edge_barrier = np.zeros_like(ann_mask, dtype=bool)
    else:
        T = float(np.quantile(vals, q_edge))
        edge_barrier = (grad >= T) & ann_mask

    # smooth barrier
    if smooth_barrier_r and smooth_barrier_r > 0:
        se = disk(int(smooth_barrier_r))
        if close_barrier:
            edge_barrier = ndi.binary_dilation(edge_barrier, structure=se)
            edge_barrier = ndi.binary_erosion(edge_barrier, structure=se)
        edge_barrier = ndi.binary_erosion(edge_barrier, structure=se)
        edge_barrier = ndi.binary_dilation(edge_barrier, structure=se)
        edge_barrier &= ann_mask

    # flood fill of bg connected to boundary, blocked by barrier
    allowed = bg & ann_mask & (~edge_barrier)
    seeds = ann_boundary & allowed
    bg_edge = ndi.binary_propagation(seeds, mask=allowed)

    # smooth bg_edge (optional)
    if smooth_bg_edge_r and smooth_bg_edge_r > 0:
        se = disk(int(smooth_bg_edge_r))
        bg_edge = ndi.binary_dilation(bg_edge, structure=se)
        bg_edge = ndi.binary_erosion(bg_edge, structure=se)
        bg_edge &= (bg & ann_mask)

    return bg_edge, edge_barrier, grad


def mask_to_best_polygon(mask, x0, y0):
    if mask.sum() < MIN_OBJ_AREA:
        return None

    contours = find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return None

    best_poly = None
    best_area = -1.0
    for c in contours:
        xy = np.column_stack([c[:, 1] + x0, c[:, 0] + y0])
        if len(xy) < 3:
            continue
        p = Polygon(xy)
        if not p.is_valid:
            p = make_valid(p)
        if p.is_empty:
            continue
        if not isinstance(p, Polygon):
            try:
                p = unary_union(p)
            except Exception:
                continue
        if p.is_empty:
            continue
        area = p.area
        if area > best_area:
            best_area = area
            best_poly = p

    if best_poly is None or best_poly.is_empty:
        return None

    if SIMPLIFY_TOL and SIMPLIFY_TOL > 0:
        best_poly = best_poly.simplify(SIMPLIFY_TOL, preserve_topology=True)

    if not best_poly.is_valid:
        best_poly = make_valid(best_poly)
        if not isinstance(best_poly, Polygon):
            best_poly = unary_union(best_poly)

    if best_poly.is_empty:
        return None

    if SMOOTH_BUF > 0:
        best_poly = best_poly.buffer(SMOOTH_BUF).buffer(-SMOOTH_BUF)

    return best_poly


def repair_mask_per_component(new_mask, ann_mask, r_close=0):
    if not new_mask.any():
        return new_mask

    lab = sk_label(new_mask, connectivity=2)
    if lab.max() == 0:
        return new_mask

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best_id = int(counts.argmax())
    comp = (lab == best_id)

    if r_close and r_close > 0:
        comp = binary_dilation(comp, structure=disk(int(r_close)))
        comp = binary_erosion(comp, structure=disk(int(r_close)))

    comp = binary_fill_holes(comp)
    return comp & ann_mask


# -----------------------------
# CORE ADJUSTMENT (returns new_geom, passed_bool)
# -----------------------------
def adjust_annotation_geom(slide, geom):
    W, H = slide.dimensions

    minx, miny, maxx, maxy = geom_to_xy_bounds_px(geom)
    x0 = minx - PAD_PX
    y0 = miny - PAD_PX
    w = (maxx - minx) + 2 * PAD_PX
    h = (maxy - miny) + 2 * PAD_PX

    roi = clamp_roi(x0, y0, w, h, W, H)
    if roi is None:
        return geom, False
    x0, y0, w, h = roi

    rgb = np.array(slide.read_region((x0, y0), 0, (w, h)).convert("RGB"), dtype=np.uint8)

    ann_mask = rasterize_geom_to_mask(geom, x0, y0, w, h)
    if ann_mask.sum() < MIN_OBJ_AREA:
        return geom, False

    bg = background_mask_white(rgb, BG_L_MIN, BG_CHROMA_MAX)

    # --- bg_edge ---
    if USE_OD_BARRIER:
        bg_edge = bg_touching_annotation_boundary_odbarrier(
            rgb, bg, ann_mask,
            q_edge=OD_Q_EDGE,
            use_boundary_band=OD_USE_BOUNDARY_BAND,
            band_px=OD_BAND_PX,
            smooth_barrier_r=OD_SMOOTH_BARRIER_R,
            smooth_bg_edge_r=OD_SMOOTH_BG_EDGE_R
        )
    else:
        bg_edge = bg_touching_annotation_boundary(bg, ann_mask)

    # --- bridge prune ---
    if BRIDGE_R and BRIDGE_R > 0:
        se = disk(int(BRIDGE_R)).astype(bool)
        seed = ndi.binary_erosion(bg_edge, structure=se)
        bg_edge = reconstruction(
            seed.astype(np.uint8),
            bg_edge.astype(np.uint8),
            method="dilation"
        ).astype(bool)

    # --- thickness-core filter ---
    if THICK_MIN and THICK_MIN > 0:
        dt = distance_transform_edt(bg_edge)
        core = dt >= int(THICK_MIN)
        se2 = disk(int(THICK_MIN)).astype(bool)
        bg_edge = ndi.binary_dilation(core, structure=se2) & bg_edge
    # remove bg_edge from ann
    new_mask = ann_mask & (~bg_edge)

    # repair (largest component)
    new_mask = repair_mask_per_component(new_mask, ann_mask, r_close=int(R_CLOSE) if R_CLOSE else 0)

    # light cleanup
    if SMOOTH_RADIUS and SMOOTH_RADIUS > 0:
        new_mask = binary_closing(new_mask, footprint=disk(SMOOTH_RADIUS))
        new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
        new_mask = binary_fill_holes(new_mask)

    new_mask = remove_small_objects(new_mask, MIN_OBJ_AREA)
    new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
    new_mask = binary_fill_holes(new_mask)

    # --- PASS/FAIL safeguards ---
    orig = int(ann_mask.sum())
    new  = int(new_mask.sum())
    removed = orig - new

    if new < MIN_KEEP_RATIO * orig:
        return geom, False

    if orig < SMALL_ORIG_PX and removed > MAX_REMOVE_SMALL:
        return geom, False

    poly = mask_to_best_polygon(new_mask, x0, y0)
    if poly is None:
        return geom, False

    return poly, True


# -----------------------------
# OPTIONAL DEBUG ONE FEATURE
# -----------------------------
def debug_one_feature(idx):
    slide = openslide.OpenSlide(SLIDE_TIFF)

    with gzip.open(IN_GEOJSON, "rt", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj["features"] if isinstance(gj, dict) and gj.get("type") == "FeatureCollection" else gj
    feat = features[idx]
    geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
    geom = shape(geom_dict)

    # --- ROI (same as adjust function) ---
    minx, miny, maxx, maxy = geom.bounds
    x0 = int(math.floor(minx)) - PAD_PX
    y0 = int(math.floor(miny)) - PAD_PX
    w  = int(math.ceil(maxx - minx)) + 2 * PAD_PX
    h  = int(math.ceil(maxy - miny)) + 2 * PAD_PX

    W, H = slide.dimensions
    x0 = max(0, x0); y0 = max(0, y0)
    w = min(w, W - x0); h = min(h, H - y0)

    rgb = np.array(slide.read_region((x0, y0), 0, (w, h)).convert("RGB"), dtype=np.uint8)
    ann_mask = rasterize_geom_to_mask(geom, x0, y0, w, h)

    if not ann_mask.any():
        print("EMPTY ann_mask")
        return

    # --- LAB background ---
    bg = background_mask_white(rgb, BG_L_MIN, BG_CHROMA_MAX)

    # --- bg_edge + barrier (depending on your setting) ---
    edge_barrier = np.zeros_like(ann_mask, dtype=bool)

    if USE_OD_BARRIER:
        # reproduce exactly what adjust_annotation_geom() uses
        bg_edge = bg_touching_annotation_boundary_odbarrier(
            rgb, bg, ann_mask,
            q_edge=OD_Q_EDGE,
            use_boundary_band=OD_USE_BOUNDARY_BAND,
            band_px=OD_BAND_PX,
            smooth_barrier_r=OD_SMOOTH_BARRIER_R,
            smooth_bg_edge_r=OD_SMOOTH_BG_EDGE_R
        )

        # If you want the barrier plotted, call a version that returns it.
        # Minimal change: compute it again using your earlier function that returns barrier+grad
        bg_edge2, edge_barrier, grad = bg_touching_annotation_boundary_odbarrier(
            rgb, bg, ann_mask,
            q_edge=OD_Q_EDGE,
            use_boundary_band=OD_USE_BOUNDARY_BAND,
            band_px=OD_BAND_PX,
            smooth_barrier_r=OD_SMOOTH_BARRIER_R,
            smooth_bg_edge_r=OD_SMOOTH_BG_EDGE_R,
            close_barrier=True
        )
        # bg_edge and bg_edge2 should be identical; keep bg_edge2 for plotting
        bg_edge = bg_edge2

    else:
        bg_edge = bg_touching_annotation_boundary(bg, ann_mask)
        grad = None

    bg_edge_before_filters = bg_edge.copy()

    # --- bridge prune ---
    if BRIDGE_R and BRIDGE_R > 0:
        seed = binary_erosion(bg_edge, structure=disk(int(BRIDGE_R)))
        bg_edge = reconstruction(
            seed.astype(np.uint8),
            bg_edge.astype(np.uint8),
            method="dilation"
        ).astype(bool)

    # --- thickness-core filter ---
    if THICK_MIN and THICK_MIN > 0:
        dt = distance_transform_edt(bg_edge)
        core = dt >= int(THICK_MIN)
        bg_edge = binary_dilation(core, structure=disk(int(THICK_MIN))) & bg_edge

    bg_edge_after_filters = bg_edge.copy()

    # --- apply removal ---
    new_mask_raw = ann_mask & (~bg_edge)

    # --- repair + cleanup (same as adjust) ---
    new_mask = repair_mask_per_component(new_mask_raw, ann_mask, r_close=int(R_CLOSE) if R_CLOSE else 0)

    if SMOOTH_RADIUS and SMOOTH_RADIUS > 0:
        new_mask = binary_closing(new_mask, footprint=disk(SMOOTH_RADIUS))
        new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
        new_mask = binary_fill_holes(new_mask)

    new_mask = remove_small_objects(new_mask, MIN_OBJ_AREA)
    new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
    new_mask = binary_fill_holes(new_mask)

    # --- decide PASS/FAIL (same rules) ---
    orig = int(ann_mask.sum())
    new  = int(new_mask.sum())
    removed = orig - new

    fail_reasons = []
    if new < MIN_KEEP_RATIO * orig:
        fail_reasons.append(f"MIN_KEEP_RATIO triggered: new={new} < {MIN_KEEP_RATIO:.2f}*orig={orig}")
    if orig < SMALL_ORIG_PX and removed > MAX_REMOVE_SMALL:
        fail_reasons.append(f"MAX_REMOVE_SMALL triggered: removed={removed} > {MAX_REMOVE_SMALL} (orig={orig})")

    poly = mask_to_best_polygon(new_mask, x0, y0)
    if poly is None:
        fail_reasons.append("mask_to_best_polygon failed (no polygon)")

    passed = (len(fail_reasons) == 0)

    print("\n==============================")
    print("idx:", idx)
    print("orig px:", orig, "new px:", new, "removed:", removed, "keep ratio:", round(new / max(1, orig), 4))
    print("passed:", passed)
    if not passed:
        print("FAIL REASONS:")
        for r in fail_reasons:
            print(" -", r)

    # --- overlay boundaries (old vs new) ---
    overlay = rgb.copy()

    b_old = ann_mask & (~ndi.binary_erosion(ann_mask, structure=np.ones((3,3), np.uint8)))
    b_new = new_mask & (~ndi.binary_erosion(new_mask, structure=np.ones((3,3), np.uint8)))

    b_old = ndi.binary_dilation(b_old, structure=np.ones((3,3), np.uint8), iterations=8)
    b_new = ndi.binary_dilation(b_new, structure=np.ones((3,3), np.uint8), iterations=8)

    overlay[b_old] = [255, 0, 0]
    overlay[b_new] = [0, 255, 0]

    # --- PLOTS ---
    # 3x4 grid = 12 panels
    fig, ax = plt.subplots(3, 4, figsize=(18, 12))
    ax = ax.ravel()

    ax[0].imshow(rgb); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(ann_mask, cmap="gray"); ax[1].set_title("ann_mask"); ax[1].axis("off")
    ax[2].imshow(bg, cmap="gray"); ax[2].set_title(f"bg (LAB) L>={BG_L_MIN} chroma<={BG_CHROMA_MAX}"); ax[2].axis("off")

    ax[3].imshow(edge_barrier, cmap="gray"); ax[3].set_title("edge_barrier (OD)"); ax[3].axis("off")

    ax[4].imshow(bg_edge_before_filters, cmap="gray"); ax[4].set_title("bg_edge BEFORE bridge/thick"); ax[4].axis("off")
    ax[5].imshow(bg_edge_after_filters, cmap="gray"); ax[5].set_title("bg_edge AFTER bridge/thick"); ax[5].axis("off")

    ax[6].imshow(new_mask_raw, cmap="gray"); ax[6].set_title("new_mask_raw = ann & ~bg_edge"); ax[6].axis("off")
    ax[7].imshow(new_mask, cmap="gray"); ax[7].set_title("new_mask FINAL (repair+cleanup)"); ax[7].axis("off")

    # Optional: show grad if available
    if grad is not None:
        ax[8].imshow(grad, cmap="gray"); ax[8].set_title("OD grad magnitude"); ax[8].axis("off")
    else:
        ax[8].axis("off")

    ax[9].imshow(overlay); ax[9].set_title(f"Overlay red=old green=new (passed={passed})"); ax[9].axis("off")

    # histogram inside ann for bg mask sanity
    lab = rgb2lab(rgb.astype(np.float32)/255.0)
    L = lab[..., 0]
    chroma = np.sqrt(lab[...,1]**2 + lab[...,2]**2)
    ax[10].hist(L[ann_mask].ravel(), bins=50); ax[10].set_title("L inside ann")
    ax[11].hist(chroma[ann_mask].ravel(), bins=50); ax[11].set_title("Chroma inside ann")

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN: split into adjusted vs failed
# -----------------------------
def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)

    with gzip.open(IN_GEOJSON, "rt", encoding="utf-8") as f:
        gj = json.load(f)

    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        features = gj.get("features", [])
        out_template = gj
        out_mode = "featurecollection"
    elif isinstance(gj, list):
        features = gj
        out_template = None
        out_mode = "list"
    else:
        raise ValueError(f"Unsupported JSON root type: {type(gj)}")

    out_adj = []
    out_fail = []

    n_total = 0
    n_geom = 0
    n_pass = 0
    n_fail = 0

    for feat in tqdm(features, total=len(features), desc="Adjusting annotations"):
        n_total += 1

        # normalize feat -> Feature with geometry
        if isinstance(feat, dict) and "geometry" in feat:
            geom_dict = feat["geometry"]
            feat_base = dict(feat)
        elif isinstance(feat, dict) and ("type" in feat) and ("coordinates" in feat):
            geom_dict = feat
            feat_base = {"type": "Feature", "properties": {}, "geometry": geom_dict}
        else:
            # passthrough unknown items to BOTH (keeps structure)
            out_adj.append(feat)
            out_fail.append(feat)
            continue

        geom = shape(geom_dict)
        if geom.is_empty:
            out_adj.append(feat_base)
            out_fail.append(feat_base)
            continue

        n_geom += 1

        new_geom, passed = adjust_annotation_geom(slide, geom)

        if passed:
            n_pass += 1
            feat_ok = dict(feat_base)
            feat_ok["geometry"] = mapping(new_geom)
            out_adj.append(feat_ok)
        else:
            n_fail += 1
            feat_bad = dict(feat_base)
            feat_bad["geometry"] = mapping(geom)  # keep ORIGINAL in failed file
            out_fail.append(feat_bad)

    if out_mode == "featurecollection":
        outA = dict(out_template); outA["features"] = out_adj
        outF = dict(out_template); outF["features"] = out_fail
    else:
        outA = out_adj
        outF = out_fail

    with open(OUT_GEOJSON_ADJUSTED, "w", encoding="utf-8") as f:
        json.dump(outA, f)

    with open(OUT_GEOJSON_FAILED, "w", encoding="utf-8") as f:
        json.dump(outF, f)

    print("\nDone.")
    print(f"Total items: {n_total}")
    print(f"Geometry items processed: {n_geom}")
    print(f"Passed (adjusted): {n_pass}")
    print(f"Failed (kept original): {n_fail}")
    print(f"ADJUSTED output: {OUT_GEOJSON_ADJUSTED}")
    print(f"FAILED output:   {OUT_GEOJSON_FAILED}")


if __name__ == "__main__":
    main()

# # Example debug call in Spyder (optional):
# # ---- DEBUG: run on selected features ----
# test_idxs = [11,22,33,44,55,66]

# for i in test_idxs:
#     print("\n==============================")
#     print("Testing idx:", i)

#     # temporarily override globals (edit these numbers)
#     BG_L_MIN = 99.0
#     BG_CHROMA_MAX = 4.0

#     USE_OD_BARRIER = True
#     OD_Q_EDGE = 0.5
#     OD_USE_BOUNDARY_BAND = True
#     OD_BAND_PX = 25
#     OD_SMOOTH_BARRIER_R = 15
#     OD_SMOOTH_BG_EDGE_R = 10

#     BRIDGE_R = 20
#     THICK_MIN = 0
#     R_CLOSE = 0

#     debug_one_feature(i)
