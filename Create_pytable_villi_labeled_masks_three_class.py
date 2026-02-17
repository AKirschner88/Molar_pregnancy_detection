#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gzip
import geojson
import shapely
import numpy as np
import cv2
import openslide
import tables
import sklearn.model_selection
from tqdm import tqdm

from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree


# =========================
# EDIT THESE
# =========================
tile_size = 1024
openslidelevel = 3  # must match the level used in your dataset creation

image_numbers = [9,10]

# IMPORTANT: point this to your NEW 2-class geojson per slide
# Example naming: DP00002009_2class.geojson or .geojson.gz
def geojson_2class_path(img_num: int) -> str:
    return fr"E:\Digital_Pathology\Project\Aron_to_share\Annotations\Adjusted_annotations\adjustedagain\DP000020{str(img_num).zfill(2)}_2class.geojson"

def wsi_path(img_num: int) -> str:
    return fr"E:\Digital_Pathology\Project\molar_data\tif_slides\DP000020{str(img_num).zfill(2)}.tif"

# output
version = "AKV22"
img_tag = "_".join(str(i) for i in image_numbers)
hdf5_out_dir = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets"

# Random subset (optional)
MAX_SAMPLES = 1000  # set None to keep all
RANDOM_SEED = 42

# =========================


def convert_coords_to_int(x):
    return np.array(x).round().astype(np.int32)


def load_features_any(path: str):
    """Returns a list of GeoJSON Features (dict). Supports .gz and root=list or FeatureCollection."""
    if path.endswith(".gz"):
        with gzip.GzipFile(path, "r") as f:
            obj = geojson.loads(f.read())
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = geojson.load(f)

    # obj can be list-of-features or FeatureCollection
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        return obj.get("features", [])
    # sometimes geojson lib returns FeatureCollection as geojson.FeatureCollection
    if hasattr(obj, "features"):
        return list(obj.features)

    raise ValueError(f"Unsupported geojson root: {type(obj)}")


def polygon_fill(multi_mask, poly, x0, y0, scalefactor, color):
    """Fill Polygon or MultiPolygon into mask."""
    if poly.is_empty:
        return

    if isinstance(poly, shapely.geometry.Polygon):
        coords = np.array(poly.exterior.coords.xy).T
        pts = convert_coords_to_int([(coords - np.array([x0, y0])) / scalefactor])
        cv2.fillPoly(multi_mask, pts=pts, color=color)

    elif isinstance(poly, shapely.geometry.MultiPolygon):
        for sub_poly in poly.geoms:
            polygon_fill(multi_mask, sub_poly, x0, y0, scalefactor, color)

    else:
        # try to fix invalid geometry
        poly2 = poly.buffer(0)
        if isinstance(poly2, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            polygon_fill(multi_mask, poly2, x0, y0, scalefactor, color)


def initialize_storage(hdf5_file):
    storage = {}
    block_shape = {
        "img": np.array((tile_size, tile_size, 3)),
        "mask": np.array((tile_size, tile_size)),
        "label": np.array((tile_size, tile_size)),
    }
    imgtypes = ["img", "mask", "label"]
    img_dtype = {
        "img": tables.UInt8Atom(),
        "mask": tables.UInt8Atom(),
        "label": tables.Int16Atom(),
    }

    for imgtype in imgtypes:
        storage[imgtype] = hdf5_file.create_earray(
            hdf5_file.root, imgtype, img_dtype[imgtype],
            shape=(0,) + tuple(block_shape[imgtype]),
            chunkshape=(1,) + tuple(block_shape[imgtype])
        )
    return storage


def main():
    all_tiles = []
    all_masks = []

    for img_num in image_numbers:
        json_fname = geojson_2class_path(img_num)
        wsi_fname = wsi_path(img_num)

        print(f"\nProcessing image: {img_num}")
        print(f"2-class geojson: {json_fname}")
        print(f"WSI: {wsi_fname}")

        feats = load_features_any(json_fname)
        print("done loading polygons")

        # Build (poly, cls) list
        poly_cls = []
        for ft in feats:
            if not isinstance(ft, dict) or ft.get("type") != "Feature":
                continue
            g = ft.get("geometry")
            if g is None:
                continue
            props = ft.get("properties", {}) if isinstance(ft.get("properties", None), dict) else {}
            cls = props.get("class", None)
            if cls not in (1, 2):
                continue

            poly = shape(g)

            # if multipolygon, keep largest subpoly (like your current script)
            if isinstance(poly, shapely.MultiPolygon):
                areas = [sub.area for sub in poly.geoms]
                poly = poly.geoms[int(np.argmax(areas))]

            if poly.is_empty:
                continue

            poly_cls.append((poly, int(cls)))

        allpolygons = [p for p, c in poly_cls]
        classes_for_poly = [c for p, c in poly_cls]

        # STRtree on geometries
        searchtree = STRtree(allpolygons)
        print("done creating tree")

        osh = openslide.OpenSlide(wsi_fname)
        scalefactor = int(osh.level_downsamples[openslidelevel])

        W0, H0 = osh.level_dimensions[0]

        step = round(tile_size * scalefactor)

        for y in tqdm(range(0, H0, step), desc=f"outer {img_num}", leave=False):
            for x in range(0, W0, step):

                tilepoly = Polygon([
                    [x, y], [x + tile_size * scalefactor, y],
                    [x + tile_size * scalefactor, y + tile_size * scalefactor],
                    [x, y + tile_size * scalefactor]
                ])

                hit_idxs = searchtree.query(tilepoly, predicate="intersects")

                tile = np.array(osh.read_region((x, y), openslidelevel, (tile_size, tile_size)))[:, :, :3]
                multi_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

                # padding like your script
                if x + tile_size * scalefactor >= W0:
                    valid_x = int((W0 - 1 - x) // scalefactor)
                    tile[:, valid_x:, :] = 255
                if y + tile_size * scalefactor >= H0:
                    valid_y = int((H0 - 1 - y) // scalefactor)
                    tile[valid_y:, :, :] = 255

                if np.mean(cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)) > 250:
                    continue

                if len(hit_idxs) == 0:
                    continue

                # Draw: class1 first, class2 after (so inner overwrites ring where they overlap)
                # Collect hits with their class
                hits_c1 = []
                hits_c2 = []
                for hi in hit_idxs:
                    poly = searchtree.geometries.take(hi)
                    cls = classes_for_poly[hi]
                    if cls == 1:
                        hits_c1.append(poly)
                    else:
                        hits_c2.append(poly)

                for poly in hits_c1:
                    polygon_fill(multi_mask, poly, x, y, scalefactor, color=1)
                for poly in hits_c2:
                    polygon_fill(multi_mask, poly, x, y, scalefactor, color=2)

                # keep only tiles that have signal
                if np.any(multi_mask > 0):
                    all_tiles.append(tile)
                    all_masks.append(multi_mask)

    print("✅ Done looping through all images and tiles.")
    print(f"Total tiles kept: {len(all_tiles)}")

    # Optional random subset
    if MAX_SAMPLES is not None and len(all_tiles) > MAX_SAMPLES:
        np.random.seed(RANDOM_SEED)
        idxs = np.random.choice(len(all_tiles), size=MAX_SAMPLES, replace=False)
        all_tiles = [all_tiles[i] for i in idxs]
        all_masks = [all_masks[i] for i in idxs]

    # train/test split
    tiles_train, tiles_test, masks_train, masks_test = sklearn.model_selection.train_test_split(
        all_tiles, all_masks, train_size=0.8, random_state=42
    )

    # Save to pytable (same structure as your script)
    filters = tables.Filters(complevel=6, complib="zlib")
    os.makedirs(hdf5_out_dir, exist_ok=True)

    hdf5_train_path = os.path.join(hdf5_out_dir, f"{version}_img{img_tag}_data_villi_multi_train.pytable")
    hdf5_test_path  = os.path.join(hdf5_out_dir, f"{version}_img{img_tag}_data_villi_multi_test.pytable")

    hdf5_train = tables.open_file(hdf5_train_path, mode="w", filters=filters)
    hdf5_test  = tables.open_file(hdf5_test_path, mode="w", filters=filters)

    storage_train = initialize_storage(hdf5_train)
    storage_test  = initialize_storage(hdf5_test)

    classes = [0, 1, 2]
    totals_train = np.zeros((2, len(classes)), dtype=np.int32)
    totals_test  = np.zeros((2, len(classes)), dtype=np.int32)

    for tile, mask in zip(tiles_train, masks_train):
        storage_train["img"].append(tile[np.newaxis])
        storage_train["mask"].append(mask[np.newaxis])
        for i, key in enumerate(classes):
            totals_train[1, i] += np.sum(mask == key)

    for tile, mask in zip(tiles_test, masks_test):
        storage_test["img"].append(tile[np.newaxis])
        storage_test["mask"].append(mask[np.newaxis])
        for i, key in enumerate(classes):
            totals_test[1, i] += np.sum(mask == key)

    hdf5_train.create_carray(hdf5_train.root, "numpixels", tables.Atom.from_dtype(totals_train.dtype), totals_train.shape)[:] = totals_train
    hdf5_test.create_carray(hdf5_test.root, "numpixels", tables.Atom.from_dtype(totals_test.dtype), totals_test.shape)[:] = totals_test

    hdf5_train.close()
    hdf5_test.close()

    print("✅ Done constructing datasets")
    print(f"Saved:\n  {hdf5_train_path}\n  {hdf5_test_path}")
    print(f"Train tiles: {len(tiles_train)} | Test tiles: {len(tiles_test)}")


if __name__ == "__main__":
    main()
