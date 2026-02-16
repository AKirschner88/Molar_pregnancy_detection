#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import tables

# ---------------------------
# EDIT THESE
# ---------------------------
MODEL_DIR = r"E:\Digital_Pathology\Project\Aron_to_share\Models"
DATANAME  = r"AKV17_data_villi_multi"
PT_PATH   = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV11_img10_-75_data_villi_multi_test.pytable"

CONFIG_PATH = os.path.join(MODEL_DIR, f"{DATANAME}_best_model_multi_UNet_config.txt")
MODEL_PATH  = os.path.join(MODEL_DIR, f"{DATANAME}_best_model_multi_UNet.pth")

# ---------------------------
# IMPORT YOUR UNet
# ---------------------------
from UNet_villi import UNet  # adjust if needed


# ---------------------------
# CONFIG LOADER
# ---------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # JSON
    try:
        return json.loads(txt)
    except Exception:
        pass

    # Python dict literal
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: key: value per line
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
# VIS HELPERS
# ---------------------------
def overlay_mask(ax, bin_mask: np.ndarray, color, alpha: float = 0.35) -> None:
    bin_mask = np.asarray(bin_mask).astype(bool)
    rgba = np.zeros((bin_mask.shape[0], bin_mask.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = bin_mask.astype(np.float32) * alpha
    ax.imshow(rgba, interpolation="none")


def gt_to_binary(gt: np.ndarray) -> np.ndarray:
    """
    If GT is multi-class (0/1/2): foreground == (gt==2)
    If GT is binary (0/1):       foreground == (gt==1)
    """
    gt = np.asarray(gt)
    if gt.max() > 1:
        return (gt == 1)
    return (gt == 1)


@torch.no_grad()
def predict_binary(model: torch.nn.Module, img_hwcn: np.ndarray, device: torch.device):
    """
    Returns:
      pred_lbl (H,W) in {0,1,...}
      img_norm (H,W,C) float32 in [0,1]
    """
    img = img_hwcn.astype(np.float32, copy=False)
    if img.max() > 1.5:
        img = img / 255.0

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1,C,H,W)

    out = model(x)
    logits = out[0] if isinstance(out, (list, tuple)) else out  # (1,C,H,W) or (C,H,W)
    if logits.ndim == 4:
        logits = logits[0]  # (C,H,W)

    pred = torch.argmax(logits, dim=0)  # (H,W)
    return pred.detach().cpu().numpy(), img


# ---------------------------
# MODEL BUILD/LOAD
# ---------------------------
def build_model_from_cfg(cfg: dict, device: torch.device) -> torch.nn.Module:
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
        raise FileNotFoundError(f"PyTable not found: {PT_PATH}")

    cfg = load_config(CONFIG_PATH)

    dev_str = str(cfg.get("device", "")).lower()
    if "cuda" in dev_str and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model_from_cfg(cfg, device)
    load_weights(model, MODEL_PATH, device)
    model.eval()

    # TP/FP/FN colors
    TP_COLOR = (0.0, 1.0, 0.0)  # green
    FP_COLOR = (0.0, 1.0, 1.0)  # cyan
    FN_COLOR = (1.0, 0.0, 0.0)  # red

    alpha = 0.35
    max_samples = 5

    with tables.open_file(PT_PATH, "r") as db:
        imgs = db.root.img
        masks = db.root.mask
        n_show = min(max_samples, imgs.shape[0])

        for i in range(n_show):
            img = imgs[i]
            gt  = masks[i]

            pred_lbl, img_norm = predict_binary(model, img, device)

            gt_fg = gt_to_binary(gt)
            pr_fg = (pred_lbl == 1)

            # TP/FP/FN
            tp = gt_fg & pr_fg
            fp = (~gt_fg) & pr_fg
            fn = gt_fg & (~pr_fg)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(f"{DATANAME} | Class 1, edgeweight:1.1 sample {i}: TP(green) FP(cyan) FN(red)")

            ax.imshow(img_norm)
            overlay_mask(ax, tp, TP_COLOR, alpha=alpha)
            overlay_mask(ax, fp, FP_COLOR, alpha=alpha)
            overlay_mask(ax, fn, FN_COLOR, alpha=alpha)

            ax.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
