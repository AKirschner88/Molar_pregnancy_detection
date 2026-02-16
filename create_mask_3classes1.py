#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json, gzip, math
import numpy as np
import openslide

from tqdm import tqdm

from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely import affinity
from shapely.ops import unary_union
from shapely.validation import make_valid

from skimage.draw import polygon as rrcc_polygon
from skimage.color import rgb2hed
from skimage.filters import gaussian
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.measure import find_contours, label as cc_label
from scipy.ndimage import distance_transform_edt, binary_fill_holes


# ============================================================
# EDIT THESE
# ============================================================
SLIDE_TIFF   = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002001.tif"
IN_GEOJSON   = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002001_adjusted.geojson"

OUT_GEOJSON_2CLASS = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002001_2class_7.geojson"
OUT_GZ = False  # True -> write .geojson.gz

READ_LEVEL = 3  # compute at this level

# Parameters tuned at level-0; we scale them to READ_LEVEL
PAD_PX_L0 = 64
SIGMA_H_L0 = 6

OUTER_BAND_PX_L0 = 10
INNER_MIN_DIST_PX_L0 = 75
INNER_QUANTILE = 0.9

MIN_SEED_AREA_L0 = 500

OPEN_RADIUS_L0 = 10
CLOSE_RADIUS_L0 = 50
SMOOTH_WIN_L0 = 0

SIMPLIFY_TOL_L0 = 0.0  # optional simplify in level-0 px, 0 disables

# If class2 fails, keep only class1 = original (set True/False)
KEEP_CLASS1_IF_CLASS2_FAIL = True
# ============================================================


def load_json_any(path: str):
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_geojson(path: str, fc: dict, gzip_out: bool):
    if gzip_out or path.lower().endswith(".gz"):
        if not path.lower().endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(fc, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f)


def scale_px_int(px_l0: int, ds: float, min_val: int = 1) -> int:
    return max(min_val, int(round(px_l0 / ds)))


def scale_px_float(px_l0: float, ds: float, min_val: float = 0.5) -> float:
    return max(min_val, float(px_l0) / ds)


def force_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

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


def rasterize_geom_to_mask_level(geom0, x0_0, y0_0, wL, hL, ds):
    """
    Rasterize a LEVEL-0 geometry onto a READ_LEVEL ROI grid.

    - geom0: shapely in level-0 coords
    - x0_0, y0_0: ROI origin in level-0 coords (used for read_region)
    - wL, hL: ROI size in level pixels
    - ds: level_downsample for READ_LEVEL
    """
    mask = np.zeros((int(hL), int(wL)), dtype=bool)
    ds = float(ds)

    def to_roi_level_xy(coords_xy):
        arr = np.asarray(coords_xy, dtype=np.float32)
        xs = (arr[:, 0] - x0_0) / ds
        ys = (arr[:, 1] - y0_0) / ds
        return xs, ys

    def burn_poly(poly: Polygon):
        if poly.is_empty:
            return
        xs, ys = to_roi_level_xy(poly.exterior.coords)
        rr, cc = rrcc_polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = True
        for hole in poly.interiors:
            xh, yh = to_roi_level_xy(hole.coords)
            rrh, cch = rrcc_polygon(yh, xh, shape=mask.shape)
            mask[rrh, cch] = False

    g = geom0
    if not g.is_valid:
        g = make_valid(g)
        if not isinstance(g, (Polygon, MultiPolygon)):
            g = unary_union(g)

    if isinstance(g, Polygon):
        burn_poly(g)
    elif isinstance(g, MultiPolygon):
        for p in g.geoms:
            burn_poly(p)
    else:
        g2 = g.buffer(0)
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


def smooth_closed_contour(c_rc: np.ndarray, win: int) -> np.ndarray:
    if win < 3:
        return c_rc
    win = force_odd(win)

    n = c_rc.shape[0]
    if n < win:
        return c_rc

    pad = win // 2
    cpad = np.vstack([c_rc[-pad:], c_rc, c_rc[:pad]])
    kernel = np.ones(win, dtype=np.float64) / win
    r = np.convolve(cpad[:, 0], kernel, mode="valid")
    c = np.convolve(cpad[:, 1], kernel, mode="valid")
    return np.stack([r, c], axis=1)


def filled_from_outer_contour(seed_mask: np.ndarray, smooth_win: int) -> np.ndarray:
    cs = find_contours(seed_mask.astype(np.uint8), level=0.5)
    if len(cs) == 0:
        return np.zeros(seed_mask.shape, dtype=bool)
    c = max(cs, key=lambda x: x.shape[0])  # (row, col)
    c_sm = smooth_closed_contour(c, win=smooth_win)
    rr, cc = rrcc_polygon(c_sm[:, 0], c_sm[:, 1], shape=seed_mask.shape)
    out = np.zeros(seed_mask.shape, dtype=bool)
    out[rr, cc] = True
    out = binary_fill_holes(out)
    return out


def mask_to_best_polygon_level0(maskL: np.ndarray, x0_0: int, y0_0: int, ds: float, simplify_tol_l0: float):
    """
    Convert a boolean mask on READ_LEVEL grid back to ONE best polygon in level-0 coords.
    """
    cs = find_contours(maskL.astype(np.uint8), 0.5)
    if not cs:
        return None

    best = None
    best_area = -1.0
    ds = float(ds)

    for c in cs:
        if c.shape[0] < 30:
            continue
        xs0 = x0_0 + c[:, 1] * ds
        ys0 = y0_0 + c[:, 0] * ds
        poly = Polygon(np.column_stack([xs0, ys0]))
        if not poly.is_valid:
            poly = make_valid(poly)
            if not isinstance(poly, Polygon):
                poly = unary_union(poly)
        if poly.is_empty:
            continue
        if simplify_tol_l0 and simplify_tol_l0 > 0:
            poly = poly.simplify(simplify_tol_l0, preserve_topology=True)
        if poly.is_empty:
            continue
        if poly.area > best_area:
            best_area = poly.area
            best = poly

    return best


def compute_class2_inner_polygon_level0(slide, geom0):
    """
    Compute class2 inner region for ONE geometry.
    Returns shapely geometry in level-0 coords, or None on failure.
    """
    LVL = READ_LEVEL
    ds = float(slide.level_downsamples[LVL])

    # scale params to level grid
    OUTER_BAND_PX = scale_px_int(OUTER_BAND_PX_L0, ds, 1)
    PAD_PX = scale_px_int(PAD_PX_L0, ds, 1)
    SIGMA_H = scale_px_float(SIGMA_H_L0, ds, 0.5)
    INNER_MIN_DIST = scale_px_int(INNER_MIN_DIST_PX_L0, ds, 1)

    OPEN_RADIUS = scale_px_int(OPEN_RADIUS_L0, ds, 0)
    CLOSE_RADIUS = scale_px_int(CLOSE_RADIUS_L0, ds, 0)

    MIN_SEED_AREA = max(10, int(round(MIN_SEED_AREA_L0 / (ds * ds))))
    SMOOTH_WIN = force_odd(max(5, int(round(SMOOTH_WIN_L0 / ds))))

   # Convert geometry from level-0 to READ_LEVEL coordinates
    geomL = affinity.scale(geom0, xfact=1.0 / ds, yfact=1.0 / ds, origin=(0, 0))
    
    # ROI bounds in READ_LEVEL coordinates
    minxL, minyL, maxxL, maxyL = geomL.bounds
    x0L = int(math.floor(minxL)) - PAD_PX
    y0L = int(math.floor(minyL)) - PAD_PX
    wL  = int(math.ceil(maxxL - minxL)) + 2 * PAD_PX
    hL  = int(math.ceil(maxyL - minyL)) + 2 * PAD_PX
    
    # Clamp ROI in READ_LEVEL coordinates
    Wl, Hl = slide.level_dimensions[LVL]
    x0L = max(0, x0L)
    y0L = max(0, y0L)
    wL = min(wL, Wl - x0L)
    hL = min(hL, Hl - y0L)
    if wL <= 2 or hL <= 2:
        return None
    
    # OpenSlide read_region uses level-0 origin, level-sized width/height
    x0_0 = int(round(x0L * ds))
    y0_0 = int(round(y0L * ds))
    
    rgb = np.array(
        slide.read_region((x0_0, y0_0), LVL, (wL, hL)).convert("RGB"),
        dtype=np.uint8
    )
    
    # Rasterize annotation directly in READ_LEVEL coords
    ann_mask = rasterize_geom_to_mask(geomL, x0L, y0L, wL, hL)
    if ann_mask.sum() < 50:
        return None

    rgb_f = rgb.astype(np.float32) / 255.0
    H = rgb2hed(rgb_f)[..., 0]
    Hs = gaussian(H, sigma=SIGMA_H, preserve_range=True)

    dist_in = distance_transform_edt(ann_mask)
    inner_region = ann_mask & (dist_in >= INNER_MIN_DIST)
    if inner_region.sum() < 200:
        fallback = max(OUTER_BAND_PX + 2, int(dist_in.max() * 0.3))
        inner_region = ann_mask & (dist_in >= fallback)

    vals = Hs[inner_region]
    if vals.size == 0:
        return None

    thr = float(np.quantile(vals, INNER_QUANTILE))
    seed_raw = inner_region & (Hs <= thr)
    
    seed = keep_largest_component(seed_raw)
    seed = remove_small_objects(seed, MIN_SEED_AREA)
    if OPEN_RADIUS > 0:
        seed = binary_opening(seed, footprint=disk(OPEN_RADIUS))
    if CLOSE_RADIUS > 0:
        seed = binary_closing(seed, footprint=disk(CLOSE_RADIUS))
    seed = binary_fill_holes(seed)

    if seed.sum() < 50:
        return None

    class2_mask = filled_from_outer_contour(seed, smooth_win=SMOOTH_WIN) & ann_mask
    if class2_mask.sum() < 50:
        return None

    poly2 = mask_to_best_polygon_level0(class2_mask, x0_0, y0_0, ds, SIMPLIFY_TOL_L0)
    if poly2 is None or poly2.is_empty:
        return None

    if not poly2.is_valid:
        poly2 = make_valid(poly2)
        if not isinstance(poly2, (Polygon, MultiPolygon)):
            poly2 = unary_union(poly2)

    return poly2


def as_feature(geom, props: dict, fid):
    return {"type": "Feature", "id": fid, "geometry": mapping(geom), "properties": dict(props)}


def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)
    gj = load_json_any(IN_GEOJSON)

    # Your input is a list of Features (as you printed).
    if isinstance(gj, list):
        features = gj
    elif isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        features = gj.get("features", [])
    else:
        raise ValueError(f"Unsupported input root type: {type(gj)}")

    out_features = []
    n_total = 0
    n_geom = 0
    n_ok = 0
    n_fail = 0

    for idx, feat in enumerate(tqdm(features, total=len(features), desc="Creating 2-class GeoJSON")):
        n_total += 1

        if not (isinstance(feat, dict) and feat.get("type") == "Feature" and feat.get("geometry") is not None):
            n_fail += 1
            continue

        geom0 = shape(feat["geometry"])
        if geom0.is_empty:
            n_fail += 1
            continue

        n_geom += 1

        # class2 for this geometry
        poly2 = compute_class2_inner_polygon_level0(slide, geom0)

        base_props = feat.get("properties", {}) if isinstance(feat.get("properties", None), dict) else {}

        if poly2 is None or poly2.is_empty:
            n_fail += 1
            if KEEP_CLASS1_IF_CLASS2_FAIL:
                props1 = dict(base_props)
                props1.update({"class": 1, "name": "class1_ring", "source_index": idx, "class2_failed": True})
                out_features.append(as_feature(geom0, props1, fid=f"{idx}_c1"))
            continue

        # class1 ring = outer - inner
        poly1 = geom0.difference(poly2)
        if poly1.is_empty:
            n_fail += 1
            continue

        if SIMPLIFY_TOL_L0 and SIMPLIFY_TOL_L0 > 0:
            poly1 = poly1.simplify(SIMPLIFY_TOL_L0, preserve_topology=True)
            poly2 = poly2.simplify(SIMPLIFY_TOL_L0, preserve_topology=True)

        # write two features for this input
        props1 = dict(base_props)
        props1.update({"class": 1, "name": "class1_ring", "source_index": idx})

        props2 = dict(base_props)
        props2.update({"class": 2, "name": "class2_inner", "source_index": idx})

        out_features.append(as_feature(poly1, props1, fid=f"{idx}_c1"))
        out_features.append(as_feature(poly2, props2, fid=f"{idx}_c2"))
        n_ok += 1

    out_fc = {"type": "FeatureCollection", "features": out_features}

    out_path = OUT_GEOJSON_2CLASS
    if OUT_GZ and not out_path.lower().endswith(".gz"):
        out_path += ".gz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_geojson(out_path, out_fc, gzip_out=OUT_GZ)

    print("\nDone.")
    print(f"Total items: {n_total}")
    print(f"Geometry items processed: {n_geom}")
    print(f"Class2 succeeded: {n_ok}")
    print(f"Class2 failed: {n_fail}")
    print(f"Output features written: {len(out_features)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
