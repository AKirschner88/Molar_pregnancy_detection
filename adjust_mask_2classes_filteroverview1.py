import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2hed, rgb2lab
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import binary_fill_holes

def show_filter_overview(rgb, ann_mask, sigma=2.0):
    """
    rgb: uint8 (H,W,3)
    ann_mask: bool (H,W) villus container mask
    """
    rgb_f = rgb.astype(np.float32) / 255.0

    # --- HED ---
    hed = rgb2hed(rgb_f)
    H = gaussian(hed[..., 0], sigma=sigma, preserve_range=True)
    E = gaussian(hed[..., 1], sigma=sigma, preserve_range=True)

    # --- Some useful derived maps ---
    # Low-stain proxy: smaller means "whiter/paler"
    S = H + E

    # Ratio: how hematoxylin-dominant is the pixel
    R = H / (H + E + 1e-6)

    # LAB: L is brightness (higher = brighter)
    lab = rgb2lab(rgb_f)
    L = gaussian(lab[..., 0], sigma=sigma, preserve_range=True)

    # --- Thresholds computed ONLY inside ann_mask ---
    H_in = H[ann_mask]; E_in = E[ann_mask]; S_in = S[ann_mask]; R_in = R[ann_mask]; L_in = L[ann_mask]

    tH = threshold_otsu(H_in) if H_in.size else 0
    tE = threshold_otsu(E_in) if E_in.size else 0
    tS = threshold_otsu(S_in) if S_in.size else 0
    tR = threshold_otsu(R_in) if R_in.size else 0
    tL = threshold_otsu(L_in) if L_in.size else 0

    # --- Candidate masks (all restricted to ann_mask) ---
    # “Purple nuclei-ish”
    m_H_high = ann_mask & (H >= tH)

    # “Pink/rim-ish”
    m_E_high = ann_mask & (E >= tE)

    # “Pale core-ish” using low total stain
    m_S_low = ann_mask & (S <= tS)

    # “H-dominant” (purple vs pink) using ratio
    m_R_high = ann_mask & (R >= tR)

    # “Bright” regions (often lumen/white stroma)
    m_L_high = ann_mask & (L >= tL)

    # Quantile variants (often easier to tune than Otsu)
    q = 0.30
    m_S_q_low = ann_mask & (S <= np.quantile(S_in, q))
    m_H_q_low = ann_mask & (H <= np.quantile(H_in, q))
    m_E_q_low = ann_mask & (E <= np.quantile(E_in, q))

    # Simple “low H AND low E” (your earlier goal, but shown explicitly)
    m_lowHE = ann_mask & (H <= np.quantile(H_in, 0.35)) & (E <= np.quantile(E_in, 0.35))

    # Optional: make a SOLID core from a candidate (largest + fill) to see behavior
    def solidify(m):
        # keep largest CC
        from skimage.measure import label
        labm = label(m)
        if labm.max() == 0:
            return m
        areas = np.bincount(labm.ravel()); areas[0] = 0
        m2 = (labm == areas.argmax())
        m2 = binary_closing(m2, disk(5))
        m2 = binary_fill_holes(m2)
        return m2

    solid_S_low = solidify(m_S_low)
    solid_lowHE = solidify(m_lowHE)
    solid_L_high = solidify(m_L_high)

    # --- Plot ---
    items = [
        ("RGB", rgb),
        ("H (smoothed)", H),
        ("E (smoothed)", E),
        ("S=H+E (low=pale)", S),
        ("R=H/(H+E)", R),
        ("L* (bright)", L),

        ("Otsu: H high", m_H_high),
        ("Otsu: E high", m_E_high),
        ("Otsu: S low", m_S_low),
        ("Otsu: R high", m_R_high),
        ("Otsu: L high", m_L_high),

        ("Q30: S low", m_S_q_low),
        ("Q30: H low", m_H_q_low),
        ("Q30: E low", m_E_q_low),
        ("Low H & Low E", m_lowHE),

        ("SOLID from S low", solid_S_low),
        ("SOLID from lowH&lowE", solid_lowHE),
        ("SOLID from L high", solid_L_high),
    ]

    n = len(items)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.8*cols, 4.2*rows))
    axes = np.array(axes).reshape(-1)

    for ax, (title, img) in zip(axes, items):
        if img.ndim == 2 and img.dtype != bool:
            ax.imshow(img)
        elif img.ndim == 2 and img.dtype == bool:
            ax.imshow(rgb)
            ax.imshow(img, alpha=0.35)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
