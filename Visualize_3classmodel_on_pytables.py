#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import tables
from UNet_villi import UNet  # adjust if your UNet is elsewhere

# ---------------------------
# EDIT THESE
# ---------------------------
MODEL_DIR = r"E:\Digital_Pathology\Project\Aron_to_share\Models"
DATANAME  = r"AKV25_data_villi_multi"  # must match your saved naming
PT_PATH   = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV22_img9_data_villi_multi_test.pytable"  # <-- change

# If your filenames follow your pattern:
CONFIG_PATH = os.path.join(MODEL_DIR, f"{DATANAME}_best_model_multi_UNet_config.txt")
MODEL_PATH  = os.path.join(MODEL_DIR, f"{DATANAME}_best_model_multi_UNet.pth")

# ---------------------------
# CONFIG LOADER (robust)
# ---------------------------
def load_config(path: str) -> dict:
    """
    Supports config files saved as:
    - JSON
    - Python dict repr (str(config))
    - key: value lines (basic fallback)
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # 1) JSON
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 2) Python dict literal
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Fallback: parse "key : value" per line
    cfg = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip()
            try:
                cfg[k] = ast.literal_eval(v)
            except Exception:
                cfg[k] = v.strip('"').strip("'")
    return cfg


# ---------------------------
# HELPERS
# ---------------------------
def overlay_mask(ax, bin_mask: np.ndarray, color, alpha: float = 0.35) -> None:
    if bin_mask.dtype != np.bool_:
        bin_mask = bin_mask.astype(bool)

    rgba = np.zeros((bin_mask.shape[0], bin_mask.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = bin_mask.astype(np.float32) * alpha
    ax.imshow(rgba, interpolation="none")


@torch.no_grad()
def predict_labels(model: torch.nn.Module, img_hwcn: np.ndarray, device: torch.device):
    img = img_hwcn
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:  # likely 0..255
        img = img / 255.0

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    logits = model(x)[0]                 # (C,H,W)
    pred = torch.argmax(logits, dim=0)   # (H,W)
    return pred.detach().cpu().numpy(), img


def build_model_from_cfg(cfg: dict, device: torch.device) -> torch.nn.Module:
    # only pass args your UNet constructor actually uses
    model = UNet(
        n_classes=int(cfg["n_classes"]),
        in_channels=int(cfg["in_channels"]),
        padding=bool(cfg["padding"]),
        depth=int(cfg["depth"]),
        wf=int(cfg["wf"]),
        up_mode=cfg["up_mode"],
        batch_norm=bool(cfg["batch_norm"]),
    ).to(device)
    return model


def load_weights(model: torch.nn.Module, model_path: str, device: torch.device) -> None:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_dict" in state:
        model.load_state_dict(state["model_dict"])
    else:
        model.load_state_dict(state)


# ---------------------------
# MAIN
# ---------------------------
def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found:  {MODEL_PATH}")
    if not os.path.exists(PT_PATH):
        raise FileNotFoundError(f"H5 not found:     {PT_PATH}")

    cfg = load_config(CONFIG_PATH)

    # device: prefer config if valid, else auto
    dev_str = str(cfg.get("device", "")).lower()
    if "cuda" in dev_str and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model_from_cfg(cfg, device)
    load_weights(model, MODEL_PATH, device)
    model.eval()

    GT_COLOR   = (1.0, 0.0, 0.0)  # red
    PRED_COLOR = (0.0, 1.0, 1.0)  # cyan
    ALPHA_GT   = 0.35
    ALPHA_PR   = 0.35

    max_samples = int(cfg.get("batch_size", 5))
    max_samples = 5 if max_samples <= 0 else min(5, max_samples)

    with tables.open_file(PT_PATH, "r") as db:
        imgs = db.root.img
        masks = db.root.mask

        n_show = min(max_samples, imgs.shape[0])

        for i in range(n_show):
            img = imgs[i]
            gt  = masks[i]

            pred_lbl, img_norm = predict_labels(model, img, device)

            gt_c1, pr_c1 = (gt == 1), (pred_lbl == 1)
            gt_c2, pr_c2 = (gt == 2), (pred_lbl == 2)

            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f"{DATANAME} | sample {i}: overlays (GT vs Pred)", fontsize=12)

            ax = axes[0]
            ax.set_title("Class 1: GT (red) vs Pred (cyan)")
            ax.imshow(img_norm)
            overlay_mask(ax, gt_c1, GT_COLOR, alpha=ALPHA_GT)
            overlay_mask(ax, pr_c1, PRED_COLOR, alpha=ALPHA_PR)
            ax.axis("off")

            ax = axes[1]
            ax.set_title("Class 2: GT (red) vs Pred (cyan)")
            ax.imshow(img_norm)
            overlay_mask(ax, gt_c2, GT_COLOR, alpha=ALPHA_GT)
            overlay_mask(ax, pr_c2, PRED_COLOR, alpha=ALPHA_PR)
            ax.axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
