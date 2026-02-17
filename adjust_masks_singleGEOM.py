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
from scipy.ndimage import label
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from skimage.color import rgb2lab
from skimage.draw import polygon as rrcc_polygon
from skimage.measure import find_contours
from skimage.morphology import (
    remove_small_objects, remove_small_holes, binary_closing, reconstruction, disk
)
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion, binary_propagation, distance_transform_edt, binary_dilation, sobel



# -----------------------------
# USER SETTINGS
# -----------------------------
SLIDE_TIFF = r"E:\Digital_Pathology\Project\molar_data\tif_slides\DP00002009.tif"
IN_GEOJSON = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\DP000020019.geojson.gz"
OUT_GEOJSON = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002001_adjusted.geojson"
OUT_GEOJSON_RAW = r"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP00002001_unadjusted.geojson"


PAD_PX = 64                 # bbox padding (pixels)
MIN_OBJ_AREA = 0         # remove tiny bits after clamp (px^2)
MIN_HOLE_AREA = 2000        # fill small holes (px^2)
SMOOTH_RADIUS = 6           # closing radius (px); keep small
SMOOTH_BUF = 2              # pixels           
SIMPLIFY_TOL = 1.5          # polygon simplify tolerance (px); 0 to disable

# Background (white) detection in LAB:
# L in [0..100], a/b roughly [-128..127]
BG_L_MIN = 90.0             # "bright"
BG_CHROMA_MAX = 10.0        # "low chroma" => near white/gray


# -----------------------------
# HELPERS
# -----------------------------
def geom_to_xy_bounds_px(geom):
    """Return integer pixel bounds (minx, miny, maxx, maxy) from shapely geometry."""
    minx, miny, maxx, maxy = geom.bounds
    return int(math.floor(minx)), int(math.floor(miny)), int(math.ceil(maxx)), int(math.ceil(maxy))


def clamp_roi(x, y, w, h, W, H):
    """Clamp ROI to image bounds."""
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    if w <= 2 or h <= 2:
        return None
    return x, y, w, h


def rasterize_geom_to_mask(geom, x0, y0, w, h):
    """Rasterize level-0 shapely Polygon/MultiPolygon into a bool mask for ROI (x0,y0,w,h)."""
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

def bg_touching_annotation_boundary(bg, ann_mask):
    """
    Background pixels inside the annotation that are connected to the annotation boundary.
    Uses binary propagation constrained to bg & ann_mask.
    """
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
    # --- smoothing options ---
    smooth_barrier_r: int = 0,      # 0 disables. Try 1-4.
    smooth_bg_edge_r: int = 0,      # 0 disables. Try 1-3.
    close_barrier: bool = True,     # close barrier (fills small gaps)
):
    """
    Returns:
      bg_edge (bool): boundary-connected bg inside ann_mask, blocked by edge barrier
      edge_barrier (bool): barrier pixels
      grad (float): OD-sum gradient magnitude (for debugging/plotting)
    """

    # --- annotation boundary seeds ---
    ann_er = ndi.binary_erosion(ann_mask, structure=np.ones((3, 3), np.uint8))
    ann_boundary = ann_mask & (~ann_er)

    # --- OD sum ---
    I = (rgb_u8.astype(np.float32) + 1.0) / 255.0
    od = -np.log(I)
    od_sum = od[..., 0] + od[..., 1] + od[..., 2]

    # --- gradient magnitude on OD sum ---
    gx = ndi.sobel(od_sum, axis=1)
    gy = ndi.sobel(od_sum, axis=0)
    grad = np.hypot(gx, gy)

    # --- choose where to estimate the percentile threshold ---
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

    # --- smoothing the barrier (optional) ---
    if smooth_barrier_r and smooth_barrier_r > 0:
        se = disk(int(smooth_barrier_r))

        # close = dilation then erosion (fills small gaps)
        if close_barrier:
            edge_barrier = ndi.binary_dilation(edge_barrier, structure=se)
            edge_barrier = ndi.binary_erosion(edge_barrier, structure=se)

        # remove isolated barrier specks (open-ish)
        edge_barrier = ndi.binary_erosion(edge_barrier, structure=se)
        edge_barrier = ndi.binary_dilation(edge_barrier, structure=se)

        edge_barrier &= ann_mask

    # --- flood fill of background connected to boundary, blocked by edges ---
    allowed = bg & ann_mask & (~edge_barrier)
    seeds = ann_boundary & allowed
    bg_edge = ndi.binary_propagation(seeds, mask=allowed)

    # --- smoothing bg_edge itself (optional) ---
    if smooth_bg_edge_r and smooth_bg_edge_r > 0:
        se = disk(int(smooth_bg_edge_r))
        bg_edge = ndi.binary_dilation(bg_edge, structure=se)
        bg_edge = ndi.binary_erosion(bg_edge, structure=se)
        bg_edge &= (bg & ann_mask)  # stay valid

    return bg_edge, edge_barrier, grad



def background_mask_white(rgb_u8):
    """
    Detect white-ish background using LAB thresholds:
    background if (L > BG_L_MIN) and (sqrt(a^2+b^2) < BG_CHROMA_MAX)
    """
    rgb = rgb_u8.astype(np.float32) / 255.0
    lab = rgb2lab(rgb)  # L:0..100
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a*a + b*b)
    bg = (L >= BG_L_MIN) & (chroma <= BG_CHROMA_MAX)
    return bg


def mask_to_best_polygon(mask, x0, y0):
    """
    Convert boolean mask to a polygon in global coords.
    Takes largest connected contour.
    """
    if mask.sum() < MIN_OBJ_AREA:
        return None

    # contours expects float image; 0/1 works
    contours = find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return None

    # pick contour with max area (via shapely)
    best_poly = None
    best_area = -1.0
    for c in contours:
        # c is (row, col); convert to (x, y)
        xy = np.column_stack([c[:, 1] + x0, c[:, 0] + y0])
        if len(xy) < 3:
            continue
        p = Polygon(xy)
        if not p.is_valid:
            p = make_valid(p)
        if p.is_empty:
            continue
        # if make_valid returns multipolygon/geomcollection, union it
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

    # simplify a bit to reduce point count (optional)
    if SIMPLIFY_TOL and SIMPLIFY_TOL > 0:
        best_poly = best_poly.simplify(SIMPLIFY_TOL, preserve_topology=True)

    # final validity
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

    # largest component by pixel count
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best_id = int(counts.argmax())

    comp = (lab == best_id)

    if r_close and r_close > 0:
        comp = binary_dilation(comp, structure=disk(int(r_close)))
        comp = binary_erosion(comp, structure=disk(int(r_close)))

    comp = binary_fill_holes(comp)
    return comp & ann_mask




def debug_one_feature(slide_path, geojson_gz_path, idx=0, pad_px=64,
                      BG_L_MIN=50.0, BG_CHROMA_MAX=10.0,
                      R_CLOSE=10, THICK_MIN=15, BRIDGE_R=15):

    # --- load feature ---
    with gzip.open(geojson_gz_path, "rt", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj["features"] if isinstance(gj, dict) and gj.get("type") == "FeatureCollection" else gj
    feat = features[idx]
    geom_dict = feat["geometry"] if isinstance(feat, dict) and "geometry" in feat else feat
    geom = shape(geom_dict)

    # --- open slide ---
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions

    # --- bbox + pad ---
    minx, miny, maxx, maxy = geom.bounds
    x0 = int(math.floor(minx)) - pad_px
    y0 = int(math.floor(miny)) - pad_px
    w  = int(math.ceil(maxx - minx)) + 2 * pad_px
    h  = int(math.ceil(maxy - miny)) + 2 * pad_px

    # clamp
    x0 = max(0, x0); y0 = max(0, y0)
    w = min(w, W - x0); h = min(h, H - y0)

    print(f"ROI: x0={x0}, y0={y0}, w={w}, h={h}, pixels={w*h:,}")

    # --- read ROI RGB ---
    t0 = time.time()
    rgb = np.array(slide.read_region((x0, y0), 0, (w, h)).convert("RGB"), dtype=np.uint8)
    print("read_region:", round(time.time() - t0, 3), "s")

    # --- ann mask ---
    t0 = time.time()
    ann_mask = rasterize_geom_to_mask(geom, x0, y0, w, h)
    print("rasterize:", round(time.time() - t0, 3), "s")

    # --- LAB bg (you asked to keep these params in the call) ---
    t0 = time.time()
    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    bg = (L >= BG_L_MIN) & (chroma <= BG_CHROMA_MAX)
    print("rgb2lab+bg:", round(time.time() - t0, 3), "s")

    # --- boundary-connected background inside annotation ---
    t0 = time.time()
    bg_edge, edge_barrier, grad = bg_touching_annotation_boundary_odbarrier(
        rgb, bg, ann_mask,
        q_edge=0.90,
        use_boundary_band=True,
        band_px=25,
        smooth_barrier_r=3,     # try 0,1,2,3
        smooth_bg_edge_r=0      # try 0 or 1
    )

    # --- prune thin-bridge leaks (erosion + reconstruction) ---
    # NOTE: uses SciPy binary_erosion with structure= (works with your environment)
    if BRIDGE_R and BRIDGE_R > 0:
        seed = binary_erosion(bg_edge, structure=disk(int(BRIDGE_R)))
        bg_edge = reconstruction(
            seed.astype(np.uint8),
            bg_edge.astype(np.uint8),
            method="dilation"
        ).astype(bool)

    # --- optional thickness-core filter (use ONLY if you want; set THICK_MIN=0 to disable) ---
    if THICK_MIN and THICK_MIN > 0:
        dt = distance_transform_edt(bg_edge)
        core = dt >= int(THICK_MIN)
        bg_edge = binary_dilation(core, structure=disk(int(THICK_MIN))) & bg_edge

    print("bg_edge+filters:", round(time.time() - t0, 3), "s")

    # --- apply removal ---
    new_mask_raw = ann_mask & (~bg_edge)

    # --- optional light repair (keep it small) ---
    t0 = time.time()
    new_mask = repair_mask_per_component(new_mask_raw, ann_mask, r_close=int(R_CLOSE) if R_CLOSE else 0)
    print("repair:", round(time.time() - t0, 3), "s")

    # --- stats ---
    print("ann pixels:", int(ann_mask.sum()))
    print("bg pixels:", int(bg.sum()))
    print("bg inside ann:", int((bg & ann_mask).sum()))
    print("bg_edge pixels:", int(bg_edge.sum()))
    print("new_raw pixels:", int(new_mask_raw.sum()))
    print("new pixels:", int(new_mask.sum()))

    # --- show masks ---
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax = ax.ravel()

    ax[0].imshow(rgb); ax[0].set_title("RGB ROI"); ax[0].axis("off")
    ax[1].imshow(ann_mask, cmap="gray"); ax[1].set_title("Annotation mask"); ax[1].axis("off")
    ax[2].imshow(bg, cmap="gray"); ax[2].set_title("BG mask"); ax[2].axis("off")
    ax[3].imshow(bg_edge, cmap="gray"); ax[3].set_title("BG to remove"); ax[3].axis("off")

    ax[4].imshow(new_mask_raw, cmap="gray"); ax[4].set_title("Kept (new_mask_raw)"); ax[4].axis("off")
    ax[5].imshow(new_mask, cmap="gray"); ax[5].set_title(f"Kept (repaired)"); ax[5].axis("off")
    ax[6].imshow(edge_barrier, cmap="gray"); ax[6].set_title("Edge barrier"); ax[6].axis("off")

    overlay = rgb.copy()
    LINE_W = 10

    ann_b = ann_mask & (~binary_erosion(ann_mask, structure=np.ones((3, 3), np.uint8)))
    ann_b = binary_dilation(ann_b, structure=np.ones((3, 3), np.uint8), iterations=LINE_W)
    overlay[ann_b] = [255, 0, 0]

    rep_b = new_mask & (~binary_erosion(new_mask, structure=np.ones((3, 3), np.uint8)))
    rep_b = binary_dilation(rep_b, structure=np.ones((3, 3), np.uint8), iterations=LINE_W)
    overlay[rep_b] = [0, 255, 0]

    ax[7].imshow(overlay); ax[7].set_title("Overlay: red=old, green=new"); ax[7].axis("off")

    plt.tight_layout()
    plt.show()




def clamp_annotation_geom(slide, geom):
    """Main clamp: remove only white background inside the annotation."""
    W, H = slide.dimensions

    minx, miny, maxx, maxy = geom_to_xy_bounds_px(geom)
    x0 = minx - PAD_PX
    y0 = miny - PAD_PX
    w = (maxx - minx) + 2 * PAD_PX
    h = (maxy - miny) + 2 * PAD_PX

    roi = clamp_roi(x0, y0, w, h, W, H)
    if roi is None:
        return geom
    x0, y0, w, h = roi

    # Read ROI RGB
    region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
    rgb = np.array(region, dtype=np.uint8)

    # Build annotation mask in ROI
    ann_mask = rasterize_geom_to_mask(geom, x0, y0, w, h)
    if ann_mask.sum() < MIN_OBJ_AREA:
        return geom   # keep original; never drop it
    
    bg = background_mask_white(rgb)
    bg_edge = bg_touching_annotation_boundary(bg, ann_mask)
    new_mask = ann_mask & (~bg_edge)    
    new_mask = repair_mask_per_component(new_mask, ann_mask, r_close=60)

    # light cleanup (keep minimal so you don't distort non-background edges)
    if SMOOTH_RADIUS and SMOOTH_RADIUS > 0:
        new_mask = binary_closing(new_mask, footprint=disk(SMOOTH_RADIUS))
        new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
        new_mask = binary_fill_holes(new_mask)
    new_mask = remove_small_objects(new_mask, MIN_OBJ_AREA)
    new_mask = remove_small_holes(new_mask, MIN_HOLE_AREA)
    new_mask = binary_fill_holes(new_mask)

    # If clamp wiped too much, fallback to original (do nothing)
    # ---- protection: don't change too much ----
    orig = int(ann_mask.sum())
    new  = int(new_mask.sum())
    removed = orig - new
    
    # ratio cap (e.g. keep at least 80% of original area)
    MIN_KEEP_RATIO = 0.70
    if new < MIN_KEEP_RATIO * orig:
        return geom
    
    SMALL_ORIG_PX = 100_000   # tune
    MAX_REMOVE_SMALL = 20_000
    
    if orig < SMALL_ORIG_PX and removed > MAX_REMOVE_SMALL:
        return geom

    # Convert to polygon
    poly = mask_to_best_polygon(new_mask, x0, y0)
    if poly is None:
        return geom  # do nothing if conversion fails

    return poly

def keep_border_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only mask components connected to the ROI border."""
    lab, n = label(mask)
    if n == 0:
        return mask

    border = np.unique(np.concatenate([
        lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]
    ]))
    border = border[border != 0]
    if border.size == 0:
        return np.zeros_like(mask, dtype=bool)

    return np.isin(lab, border)


# -----------------------------
# RUN
# -----------------------------



def main():
    slide = openslide.OpenSlide(SLIDE_TIFF)

    # ---- read json / json.gz ----
    if str(IN_GEOJSON).lower().endswith(".gz"):
        with gzip.open(IN_GEOJSON, "rt", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        with open(IN_GEOJSON, "r", encoding="utf-8") as f:
            gj = json.load(f)

    # ---- normalize input ----
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

    out_features_adj = []
    out_features_raw = []

    n_total_items = 0
    n_geom_items = 0           # items that contain a geometry we processed
    n_adjusted = 0             # geometry changed (per your area rule)
    n_not_adjusted = 0         # geometry unchanged (per your area rule)

    for feat in tqdm(features, total=len(features), desc="Clamping annotations"):
        n_total_items += 1

        # Accept Feature objects OR geometry-only dicts
        if isinstance(feat, dict) and "geometry" in feat:
            geom_dict = feat["geometry"]
            feat_base = dict(feat)  # keep properties etc.
        elif isinstance(feat, dict) and ("type" in feat) and ("coordinates" in feat):
            geom_dict = feat
            feat_base = {"type": "Feature", "properties": {}, "geometry": geom_dict}
        else:
            # passthrough unknown items to BOTH outputs
            out_features_raw.append(feat)
            out_features_adj.append(feat)
            continue

        geom = shape(geom_dict)
        if geom.is_empty:
            out_features_raw.append(feat_base)
            out_features_adj.append(feat_base)
            continue

        n_geom_items += 1

        # -------- RAW (unadjusted) --------
        feat_raw = dict(feat_base)
        feat_raw["geometry"] = mapping(geom)
        out_features_raw.append(feat_raw)

        # -------- ADJUSTED --------
        new_geom = clamp_annotation_geom(slide, geom)
        if new_geom is None:
            new_geom = geom

        # your “changed” definition
        changed = (new_geom.area < 0.995 * geom.area)
        if changed:
            n_adjusted += 1
        else:
            n_not_adjusted += 1

        feat_adj = dict(feat_base)
        feat_adj["geometry"] = mapping(new_geom)
        out_features_adj.append(feat_adj)

    # ---- output in same style as input ----
    if out_mode == "featurecollection":
        out_raw = dict(out_template); out_raw["features"] = out_features_raw
        out_adj = dict(out_template); out_adj["features"] = out_features_adj
    else:
        out_raw = out_features_raw
        out_adj = out_features_adj

    # ---- write RAW ----
    if str(OUT_GEOJSON_RAW).lower().endswith(".gz"):
        with gzip.open(OUT_GEOJSON_RAW, "wt", encoding="utf-8") as f:
            json.dump(out_raw, f)
    else:
        with open(OUT_GEOJSON_RAW, "w", encoding="utf-8") as f:
            json.dump(out_raw, f)

    # ---- write ADJUSTED ----
    if str(OUT_GEOJSON).lower().endswith(".gz"):
        with gzip.open(OUT_GEOJSON, "wt", encoding="utf-8") as f:
            json.dump(out_adj, f)
    else:
        with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
            json.dump(out_adj, f)

    print("Done.")
    print(f"Total items: {n_total_items}")
    print(f"Geometry items processed: {n_geom_items}")
    print(f"Adjusted: {n_adjusted}")
    print(f"Not adjusted: {n_not_adjusted}")
    print(f"RAW output: {OUT_GEOJSON_RAW}")
    print(f"ADJUSTED output: {OUT_GEOJSON}")




if __name__ == "__main__":
    main()


# # --- load first geometry ---
# with gzip.open(IN_GEOJSON, "rt", encoding="utf-8") as f:
#     gj = json.load(f)

# features = gj["features"] if isinstance(gj, dict) and gj.get("type") == "FeatureCollection" else gj
# feat0 = features[1]
# geom_dict = feat0["geometry"] if "geometry" in feat0 else feat0
# geom = shape(geom_dict)

# minx, miny, maxx, maxy = geom.bounds
# x0 = int(math.floor(minx)) - PAD_PX
# y0 = int(math.floor(miny)) - PAD_PX
# w  = int(math.ceil(maxx - minx)) + 2 * PAD_PX
# h  = int(math.ceil(maxy - miny)) + 2 * PAD_PX

# # --- clamp to slide ---
# slide = openslide.OpenSlide(SLIDE_TIFF)
# W, H = slide.dimensions
# x0 = max(0, x0); y0 = max(0, y0)
# w = min(w, W - x0); h = min(h, H - y0)

# # --- read ROI and convert to LAB ---
# rgb = np.array(slide.read_region((x0, y0), 0, (w, h)).convert("RGB"), dtype=np.uint8)
# lab = rgb2lab(rgb.astype(np.float32) / 255.0)

# L = lab[..., 0]                  # 0..100
# a = lab[..., 1]
# b = lab[..., 2]
# chroma = np.sqrt(a*a + b*b)

# bg = (L >= BG_L_MIN) & (chroma <= BG_CHROMA_MAX)

# print("bg %:", 100.0 * bg.mean(), "L range:", (L.min(), L.max()), "chroma range:", (chroma.min(), chroma.max()))

# # --- show ---
# plt.figure(); plt.imshow(rgb); plt.title("RGB"); plt.axis("off")
# plt.figure(); plt.imshow(L, cmap="gray"); plt.title("L (brightness)"); plt.axis("off")
# plt.figure(); plt.imshow(chroma, cmap="gray"); plt.title("Chroma = sqrt(a^2+b^2)"); plt.axis("off")
# plt.figure(); plt.imshow(bg, cmap="gray"); plt.title(f"BG mask: L>={BG_L_MIN} & chroma<={BG_CHROMA_MAX}"); plt.axis("off")
# plt.show()


# # ---- run on one feature ----
# img= [66]
# for i in img:
#     debug_one_feature(
#         SLIDE_TIFF,
#         IN_GEOJSON,
#         idx=i,
#         pad_px=PAD_PX,
#         BG_L_MIN=99,
#         BG_CHROMA_MAX=4,
#         R_CLOSE=1,
#         THICK_MIN=0,
#         BRIDGE_R=13
#     )