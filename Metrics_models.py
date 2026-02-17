#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import json
import inspect
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tables

from UNet_villi import UNet


# ---------------------------
# EDIT THESE
# ---------------------------
MODEL_DIR = r"E:\Digital_Pathology\Project\Aron_to_share\Models"
PT_PATH   = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV11_img9_-75_data_villi_multi_test.pytable"

MODEL_TAGS = [
    r"AKV13_data_villi_multi",
]

# Optional: force ignore_index for ALL models (usually keep None)
FORCE_IGNORE_INDEX: Optional[int] = None

# Output
OUT_CSV = os.path.join(MODEL_DIR, "model_comparison_metrics_report_V22.csv")



# ---------------------------
# Utils: config parsing
# ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # JSON
    try:
        return json.loads(txt)
    except Exception:
        pass

    # Python dict literal
    try:
        cfg = ast.literal_eval(txt)
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    # key:value or key=value
    cfg: Dict[str, Any] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and "=" not in line:
            k, v = line.split(":", 1)
        elif "=" in line:
            k, v = line.split("=", 1)
        else:
            continue

        k = k.strip()
        v = v.strip()
        try:
            v_parsed = ast.literal_eval(v)
        except Exception:
            v_parsed = v
        cfg[k] = v_parsed

    return cfg


def paths_for_model(model_dir: str, model_tag: str) -> Tuple[str, str]:
    cfg_path = os.path.join(model_dir, f"{model_tag}_best_model_multi_UNet_config.txt")
    wts_path = os.path.join(model_dir, f"{model_tag}_best_model_multi_UNet.pth")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(wts_path):
        raise FileNotFoundError(f"Weights not found: {wts_path}")
    return cfg_path, wts_path


# ---------------------------
# Utils: checkpoint handling
# ---------------------------
def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # Your checkpoints typically store weights under "model_dict"
    if isinstance(ckpt, dict) and "model_dict" in ckpt:
        sd = ckpt["model_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError("Could not extract a state_dict (expected dict).")

    # Remove DataParallel prefix if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


def _get_int(d: Dict[str, Any], keys: List[str], default: int) -> int:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                pass
    return default


def _get_bool(d: Dict[str, Any], keys: List[str], default: bool) -> bool:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "y")
            try:
                return bool(int(v))
            except Exception:
                return bool(v)
    return default


def _get_str(d: Dict[str, Any], keys: List[str], default: str) -> str:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return default


def build_unet_from_cfg_and_ckpt(cfg: Dict[str, Any], ckpt: Any) -> Tuple[torch.nn.Module, int, int, Optional[int]]:
    """
    Build UNet with params from cfg, fallback to ckpt.
    Returns: (model, num_classes, in_channels, ignore_index)
    """
    src = dict(cfg)
    if isinstance(ckpt, dict):
        for k in ["depth", "wf", "up_mode", "batch_norm", "padding", "conv_block",
                  "in_channels", "n_channels", "n_classes", "num_classes", "ignore_index"]:
            if k in ckpt and k not in src:
                src[k] = ckpt[k]

    num_classes = _get_int(src, ["num_classes", "n_classes"], default=3)
    in_ch = _get_int(src, ["in_channels", "n_channels"], default=3)

    ignore_index = FORCE_IGNORE_INDEX
    if ignore_index is None:
        ii = src.get("ignore_index", None)
        if ii is not None:
            try:
                ignore_index = int(ii)
            except Exception:
                ignore_index = None

    # Architecture params (only pass what UNet accepts)
    used_params = {
        "in_channels": in_ch,
        "n_channels": in_ch,
        "n_classes": num_classes,
        "num_classes": num_classes,
        "depth": _get_int(src, ["depth"], default=5),
        "wf": _get_int(src, ["wf"], default=3),
        "up_mode": _get_str(src, ["up_mode"], default="upconv"),
        "batch_norm": _get_bool(src, ["batch_norm"], default=True),
        "padding": _get_bool(src, ["padding"], default=True),
        "conv_block": _get_str(src, ["conv_block"], default="unet"),
    }

    sig = inspect.signature(UNet.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")
    kw = {k: v for k, v in used_params.items() if k in accepted}

    # Fallback attempts if constructor is strict
    try:
        model = UNet(**kw)
    except TypeError:
        # minimal fallback (common signature)
        try:
            model = UNet(n_channels=in_ch, n_classes=num_classes)
        except TypeError:
            model = UNet(in_channels=in_ch, n_classes=num_classes)

    return model, num_classes, in_ch, ignore_index


# ---------------------------
# Metrics from confusion matrix
# ---------------------------
def confusion_matrix_from_flat(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    idx = num_classes * y_true.astype(np.int64) + y_pred.astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def metrics_from_cm(cm: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    """
    Returns per-class precision/recall/f1/support and macro_f1.
    """
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    macro_f1 = float(np.mean(f1))

    out: Dict[str, Any] = {"macro_f1": macro_f1}
    for c in range(num_classes):
        out[f"precision_c{c}"] = float(precision[c])
        out[f"recall_c{c}"] = float(recall[c])
        out[f"f1_c{c}"] = float(f1[c])
        out[f"support_c{c}"] = float(support[c])

    return out


# ---------------------------
# Preprocess + remap for 2-class models
# ---------------------------
def preprocess_img(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[..., None]
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


def remap_gt_for_2class(gt: np.ndarray) -> np.ndarray:
    # config says class 2 is merged into background
    gt2 = gt.copy()
    gt2[gt2 == 2] = 0
    return gt2


def sanitize_to_num_classes(arr: np.ndarray, num_classes: int) -> np.ndarray:
    return np.clip(arr.astype(np.int64), 0, num_classes - 1)


# ---------------------------
# Evaluate one model
# ---------------------------
def eval_one_model(model_tag: str, model_dir: str, pt_path: str) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path, wts_path = paths_for_model(model_dir, model_tag)
    cfg = load_config(cfg_path)

    ckpt = torch.load(wts_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    model, num_classes, in_ch, ignore_index = build_unet_from_cfg_and_ckpt(cfg, ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    with tables.open_file(pt_path, mode="r") as h5, torch.no_grad():
        imgs = h5.root.img
        masks = h5.root.mask

        for i in range(imgs.shape[0]):
            img = imgs[i]
            gt = masks[i].astype(np.int64)

            if num_classes == 2:
                gt = remap_gt_for_2class(gt)

            x = preprocess_img(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            if ignore_index is not None:
                valid = (gt != ignore_index)
                gt_flat = gt[valid].ravel()
                pred_flat = pred[valid].ravel()
            else:
                gt_flat = gt.ravel()
                pred_flat = pred.ravel()

            gt_flat = sanitize_to_num_classes(gt_flat, num_classes)
            pred_flat = sanitize_to_num_classes(pred_flat, num_classes)

            cm_total += confusion_matrix_from_flat(gt_flat, pred_flat, num_classes)

    m = metrics_from_cm(cm_total)

    # Build your requested report columns:
    # "accuracy class X" -> we define as TP/support (this equals recall by definition).
    # If a class doesn't exist (e.g. class2 in 2-class model), leave NaN.
    row: Dict[str, Any] = {
        "model_tag": model_tag,
        "num_classes": num_classes,
        "precision_1": m.get("precision_c1", np.nan),
        "recall_1": m.get("recall_c1", np.nan),
        "accuracy_1": m.get("recall_c1", np.nan),  # same as TP/support
        "precision_2": m.get("precision_c2", np.nan),
        "recall_2": m.get("recall_c2", np.nan),
        "accuracy_2": m.get("recall_c2", np.nan),  # same as TP/support
        "F1": m.get("macro_f1", np.nan),           # macro F1 across available classes
        "weights_path": wts_path,
        "config_path": cfg_path,
    }
    return row


# ---------------------------
# Main
# ---------------------------
def main() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tag in MODEL_TAGS:
        print(f"Evaluating: {tag}")
        rows.append(eval_one_model(tag, MODEL_DIR, PT_PATH))

    df_report = pd.DataFrame(rows)

    ordered = [
        "model_tag", "num_classes",
        "accuracy_1", "precision_1", "recall_1",
        "accuracy_2", "precision_2", "recall_2",
        "F1",
        "weights_path", "config_path",
    ]
    cols = [c for c in ordered if c in df_report.columns] + [c for c in df_report.columns if c not in ordered]
    df_report = df_report[cols]

    # Save (optional)
    df_report.to_csv(OUT_CSV, index=False, sep=";", decimal=",")

    # Show (optional)
    print("\n=== Quick view ===")
    show_cols = [c for c in ["model_tag", "num_classes", "accuracy_1", "precision_1", "recall_1",
                             "accuracy_2", "precision_2", "recall_2", "F1"] if c in df_report.columns]
    print(df_report[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}".replace(".", ",")))
    print(f"\nSaved CSV: {OUT_CSV}")

    return df_report


if __name__ == "__main__":
    df_report = main()
