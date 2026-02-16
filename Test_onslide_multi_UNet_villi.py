#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import gzip
import numpy as np
import openslide
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import cv2
import shapely
import geojson
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff

from villi_augment import *
from UNet_villi import *

# Ensure this is set before torch is imported if it's relevant for your environment
# os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


# In[2]:


def qupath_color_to_rgb(n):
    if n < 0:
        n = (1 << 24) + n

    red = (n >> 16) & 0xFF
    green = (n >> 8) & 0xFF
    blue = n & 0xFF

    return red, green, blue


def rgb_to_qupath_color(red, green, blue, alpha=255):
    unsigned_int = (alpha << 24) | (red << 16) | (green << 8) | blue
    signed_int = unsigned_int if unsigned_int < 0x80000000 else unsigned_int - 0x100000000
    return signed_int


def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n,::]


def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
    while sibling_id != -1:
        contour = contours[sibling_id].squeeze(axis=1)
        if len(contour) >= 3:
            first_child_id = hierarchy[sibling_id][2]
            children = [] if is_outer else None
            _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

            if is_outer:
                polygon = Polygon(contour, holes=children)
                polygons.append(polygon)
            else:
                siblings.append(contour)
        sibling_id = hierarchy[sibling_id][0]


def generate_polygons(contours, hierarchy):
    hierarchy = hierarchy[0]
    polygons = []
    _DFS(polygons, contours, hierarchy, 0, True, [])
    return polygons


def generate_polygons_from_mask(mask, coords, scalefactor):
    x, y = coords[0], coords[1]
    # Ensure mask is 8-bit unsigned integer type for cv2.findContours
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    if hierarchy is not None:
        polygons = generate_polygons(contours, hierarchy)
        # Apply scaling and offset to polygon coordinates
        polygons = [shapely.Polygon(shell=poly.exterior.coords._coords * scalefactor + np.array([x, y]),
                                    holes=[interior.coords._coords * scalefactor + np.array([x, y]) for interior in poly.interiors])
                    for poly in polygons]

    # print(f'Number of polygons found: {len(polygons)}')
    return polygons


def predict_without_padding(tile, model, device):
    # prepare data appropriately
    img_gpu = torch.from_numpy(np.expand_dims(tile, axis=0)).to(device, memory_format=torch.channels_last)
    img_gpu = img_gpu.permute(0, 3, 1, 2) / 255
    img_gpu = img_gpu.type(torch.float32)

    # apply unet to current patch
    unet_out_ext = model(img_gpu)
    unet_out_ext = unet_out_ext.detach().cpu().numpy()
    unet_out = np.squeeze(unet_out_ext, axis=0)

    # get mask (unet returns the values before softmax)
    unet_out = np.argmax(unet_out, axis=0)

    return unet_out


def predict_with_padding(tile, model, device):
    stride = 256
    tile_shape = tile.shape

    # prepare data appropriately
    img_gpu = torch.from_numpy(np.expand_dims(tile, axis=0)).to(device, memory_format=torch.channels_last)
    img_gpu = torch.nn.functional.pad(img_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode='reflect')
    img_gpu = img_gpu.permute(0, 3, 1, 2) / 255
    img_gpu = img_gpu.type(torch.float32)

    # apply unet to current patch
    unet_out_ext = model(img_gpu)
    unet_out_ext = unet_out_ext[:, :, stride // 2:, stride // 2:][:, :, :tile_shape[0], :tile_shape[1]]
    unet_out_ext = unet_out_ext.detach().cpu().numpy()
    unet_out = np.squeeze(unet_out_ext, axis=0)

    # get mask (unet returns the values before softmax)
    unet_out = np.argmax(unet_out, axis=0)

    return unet_out


# In[3]:


# set level and tilesize

openslidelevel = 3
tilesize = 1024
print(f"Openslide level: {openslidelevel}, tile size: {tilesize}")

#data for molar pregnancy slides
data_folder = r"E:\Digital_Pathology\Project\molar_data\tif_slides"
molar_type = ['P', 'P', 'P', 'P', 'C', 'C', 'C', 'C', 'N', 'N', 'N']
staining_type = ['HE', 'HE', 'HE', 'P57', 'HE', 'HE', 'HE', 'P57', 'HE', 'HE', 'P57']

print(f"Exploring folder {data_folder}")

#select single image
single_img = 9 # <- change here!

imgs = [data_folder + '/DP000020'+str(single_img).zfill(2) + '.tif']

print(imgs)

#create timestamp
timestamp = datetime.now().strftime("%y%m%d_%H%M")

# Create directory for debug masks
debug_mask_dir = f"E:\\Digital_Pathology\\Project\\Aron_to_share\\output_masks\\{timestamp}_{single_img}"
os.makedirs(debug_mask_dir, exist_ok=True)
print(f"Saving debug masks to: {debug_mask_dir}")

# load model
model = "AKV25"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model_path = f'E:\Digital_Pathology\Project\\Aron_to_share\\Models\\{model}_data_villi_multi_best_model_multi_UNet.pth'

# Check if model path exists before loading
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path.")

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
         padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
         up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
model.load_state_dict(checkpoint["model_dict"])
model.eval()


#loop over selected images
for img in imgs:
    img_name = os.path.basename(img).split('.')[0]
    img_num = int(img_name[-2:])
    print(f'Opening image {img} which corresponds to image named {img_name}')

    print('Starting time...')
    start = time.time()

    osh = openslide.OpenSlide(img)

    n_row, n_col = osh.level_dimensions[0]

    scalefactor = osh.level_downsamples[openslidelevel]
    
    W3, H3 = osh.level_dimensions[openslidelevel]  # (width, height) at level 3
    full_pred_L3 = np.zeros((H3, W3), dtype=np.uint8)  # values: 0,1,2

    # global polygons
    allinternalpolygons_out = []
    allboundarypolygons_out = []

    # loop over tiles
    for y in tqdm(range(0, osh.level_dimensions[0][1], round(tilesize * scalefactor)), desc='y'):
        for x in range(0, osh.level_dimensions[0][0], round(tilesize * scalefactor)):

            tilepoly = Polygon([[x, y], [x + tilesize * scalefactor, y], [x + tilesize * scalefactor, y + tilesize * scalefactor], [x, y + tilesize * scalefactor]])

            tile = np.array(osh.read_region((x, y), openslidelevel, (tilesize, tilesize)))[:, :, 0:3]

            # boundary fix to set padding of 255 instead of 0
            if x + tilesize * scalefactor >= osh.level_dimensions[0][0]:
                valid_x = int((osh.level_dimensions[0][0] - 1 - x) // scalefactor)
                tile[:, valid_x:, :] = 255
            if y + tilesize * scalefactor >= osh.level_dimensions[0][1]:
                valid_y = int((osh.level_dimensions[0][1] - 1 - y) // scalefactor)
                tile[valid_y:, :, :] = 255

            # predict without or with padding (reduces boundary effects)
            #unet_out = predict_without_padding(tile, model, device)
            unet_out = predict_with_padding(tile, model, device)
            
            # Convert level-0 tile origin (x,y) -> level-3 pixel coords
            x3 = int(round(x / scalefactor))
            y3 = int(round(y / scalefactor))
            
            # Handle boundary tiles (you already computed valid_x/valid_y in level-3 pixels)
            vx = tilesize
            vy = tilesize
            if x + tilesize * scalefactor >= osh.level_dimensions[0][0]:
                vx = valid_x
            if y + tilesize * scalefactor >= osh.level_dimensions[0][1]:
                vy = valid_y
            
            # Paste (clip-safe)
            full_pred_L3[y3:y3+vy, x3:x3+vx] = unet_out[:vy, :vx].astype(np.uint8)
            
            
            # --- START: Added code for saving debug masks ---
            # Normalize unet_out to 0-255 for proper image saving, if it's not already in that range
            # Assuming unet_out contains class indices (0, 1, 2), scale them for visualization
            # You might want to map these to specific colors for better visual distinction
            # For example, if 0=background, 1=boundary, 2=internal
            display_mask = unet_out * 100 # Simple scaling to make different classes visible
            plt.figure(figsize=(6, 6)) # Create a new figure for each plot
            plt.imshow(display_mask, cmap='viridis') # Use a colormap to distinguish classes
            plt.title(f"Prediction Mask (x={x}, y={y})")
            plt.colorbar(label='Class Index (Scaled)')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_mask_dir, f"tile_mask_x{x}_y{y}.png"))
            plt.close() # Close the figure to free memory and prevent it from being displayed
            # --- END: Added code for saving debug masks ---

            # generate polygons from mask
            allinternalpolygons_out += generate_polygons_from_mask(mask=unet_out == 2, coords=[x, y], scalefactor=scalefactor)
            allboundarypolygons_out += generate_polygons_from_mask(mask=unet_out == 1, coords=[x, y], scalefactor=scalefactor)

    osh.close()

    # simple fix to avoid issues with splitting in tiles (should be properly fixed with overlapping tiles, and this is rather slow)
    # do the union of the polygon with a small positive buffer to make sure that they overlap
    def do_union_fix(poly_list):
        print('Doing union...')
        poly_list = [shapely.buffer(poly, distance=1*scalefactor) for poly in poly_list]
        global_poly = shapely.unary_union(poly_list)
        # Ensure global_poly is a list of polygons, not a MultiPolygon if it results from unary_union
        if global_poly.geom_type == 'MultiPolygon':
            poly_list = list(global_poly.geoms)
        else:
            poly_list = [global_poly] # If it's a single Polygon, put it in a list
        return poly_list


    allboundarypolygons_out = do_union_fix(allboundarypolygons_out)
    allinternalpolygons_out = do_union_fix(allinternalpolygons_out)
    
    
    allannotations_raw = []
    
    # simple fixed colors for clarity
    color_internal_raw = rgb_to_qupath_color(255, 0, 0)   # red
    color_boundary_raw = rgb_to_qupath_color(0, 0, 255)   # blue
    
    RAW_MIN_AREA = 1.0            # only remove collapsed polygons
    RAW_SIMPLIFY_TOL = 1.0        # 1 pixel at level 0 (optional if you want simplify)
    
    # IMPORTANT: use the correct variable name (see Step 2)
    # internal masks (class 2)
    for poly in allinternalpolygons_out:
        poly = poly.buffer(0)  # fix self-intersections
        if (not poly.is_valid) or (poly.area < RAW_MIN_AREA):
            continue
    
        coords = np.asarray(poly.exterior.coords, dtype=np.int64).tolist()
        holes = [np.asarray(r.coords, dtype=np.int64).tolist() for r in poly.interiors]
    
        ann = {
            "geometry": {"type": "Polygon", "coordinates": [coords] + holes},
            "properties": {
                "object_type": "annotation",
                "isLocked": False,
                "classification": {"name": "Internal_raw", "colorRGB": color_internal_raw},
            },
            "type": "Feature",
        }
        allannotations_raw.append(ann)
    
    # boundary masks (class 1)
    for poly in allboundarypolygons_out:
        poly = poly.buffer(0)
        if (not poly.is_valid) or (poly.area < RAW_MIN_AREA):
            continue
    
        coords = np.asarray(poly.exterior.coords, dtype=np.int64).tolist()
        holes = [np.asarray(r.coords, dtype=np.int64).tolist() for r in poly.interiors]
    
        ann = {
            "geometry": {"type": "Polygon", "coordinates": [coords] + holes},
            "properties": {
                "object_type": "annotation",
                "isLocked": False,
                "classification": {"name": "Boundary_raw", "colorRGB": color_boundary_raw},
            },
            "type": "Feature",
        }
        allannotations_raw.append(ann)

    # generate annotations from polygons with postprocessing
    # postprocessing: delete holes, buffer internal, and keep only if surrounded by boundary
    allannotations_out = []
    colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))[:, 0:3]
    color_annotations = [rgb_to_qupath_color(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in colors]
    searchtree_boundary = STRtree([s for s in allboundarypolygons_out])
    for ids_poly, poly in enumerate(tqdm(allinternalpolygons_out)):
        # Ensure poly is a valid Polygon before attempting to access exterior/interiors
        if not poly.is_valid:
            poly = poly.buffer(0) # Attempt to fix invalid polygons
            if not poly.is_valid:
                print(f"Skipping invalid polygon after buffer attempt: {poly}")
                continue

        poly = shapely.Polygon(shell=poly.exterior.coords._coords)
        poly_buf = shapely.buffer(poly, distance=50)
        buf = shapely.difference(poly_buf, poly)

        # Ensure buf is a valid geometry for query
        if not buf.is_valid:
            buf = buf.buffer(0)
            if not buf.is_valid:
                print(f"Skipping invalid buffer geometry: {buf}")
                continue

        hits = searchtree_boundary.query(buf, predicate='intersects')
        area_intersect = 0
        for hit_idx in hits: # Iterate over indices returned by query
            intersect_poly = searchtree_boundary.geometries.take(hit_idx)
            # Ensure intersect_poly is valid before intersection
            if intersect_poly.is_valid:
                intersection_result = shapely.intersection(buf, intersect_poly)
                if intersection_result.is_valid:
                    area_intersect += intersection_result.area
        area_buffer = buf.area
        area_poly = poly.area
        if area_poly < 100*100 or area_intersect/(area_buffer + 1e-8) < 0.8:
            continue
        coords = poly_buf.exterior.coords._coords.astype(np.int64).tolist()
        coords_int = [interior.coords._coords.astype(np.int64).tolist() for interior in poly_buf.interiors]
        ann = {"geometry": {"type": 'Polygon', "coordinates": [coords] + coords_int},
               "properties": {"object_type": "detection", "isLocked": False,
                              'classification': {"name": "Villi" + str(ids_poly % len(color_annotations)),
                                                 "colorRGB": color_annotations[ids_poly % len(color_annotations)]},
                              "measurements": [{"name": "Intersection_rate", "value": str(area_intersect/(area_buffer + 1e-8))}]
                              },
               "type": "Feature"
               }
        allannotations_out.append(ann)

    # compress and save the output
    
    json_fname_output = f'E:\Digital_Pathology\Project\\Aron_to_share\\Output_json\\{timestamp}_model\\DP000020' + str(img_num).zfill(2) + '.json.gz'
    os.makedirs(os.path.dirname(json_fname_output), exist_ok=True)
    if json_fname_output.endswith(".gz"):
        with gzip.open(json_fname_output, 'wt', encoding="utf-8") as zipfile:
            geojson.dump(allannotations_out, zipfile)
    else:
        with open(json_fname_output, 'w') as outfile:
            geojson.dump(allannotations_out, outfile)
            
    
    json_fname_output_raw = (
    f'E:\Digital_Pathology\Project\\Aron_to_share\\Output_json\\{timestamp}_model\\'
    f'DP000020{str(img_num).zfill(2)}_raw.json.gz'
    )
    
    os.makedirs(os.path.dirname(json_fname_output_raw), exist_ok=True)
    
    with gzip.open(json_fname_output_raw, 'wt', encoding="utf-8") as zipfile:
        geojson.dump(allannotations_raw, zipfile)
    print("done saving annotations")
    
    out_dir = r"E:\Digital_Pathology\Project\Aron_to_share\output_masks"
    os.makedirs(out_dir, exist_ok=True)
    
    out_base = os.path.join(out_dir, f"{timestamp}_{img_name}_L{openslidelevel}")
    
    tiff.imwrite(out_base + "_labels.tif", full_pred_L3, compression="deflate")
    tiff.imwrite(out_base + "_class1_boundary.tif", (full_pred_L3 == 1).astype(np.uint8) * 255, compression="deflate")
    tiff.imwrite(out_base + "_class2_internal.tif", (full_pred_L3 == 2).astype(np.uint8) * 255, compression="deflate")

    end = time.time()
    print(f'Elapsed time for image {img} : {end-start}')

print(f"Done exploring all files!")


# in_tif  = r"E:\Digital_Pathology\Project\Aron_to_share\output_masks\260130_1007_DP00002010_L3_labels.tif"
# out_png = r"E:\Digital_Pathology\Project\Aron_to_share\output_masks\260130_1007_DP00002010_L3_overlay.png"

# lbl = tiff.imread(in_tif).astype(np.uint8)  # 0/1/2

# rgba = np.zeros((lbl.shape[0], lbl.shape[1], 4), dtype=np.uint8)

# # class 1 (boundary) = blue
# rgba[lbl == 1] = [0, 0, 255, 140]   # last value = alpha (0..255)

# # class 2 (internal) = red
# rgba[lbl == 2] = [255, 0, 0, 140]

# Image.fromarray(rgba, mode="RGBA").save(out_png)
# print("Wrote:", out_png, "shape:", lbl.shape)



# In[ ]:




