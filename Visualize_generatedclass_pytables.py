import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

train_path = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV22_img9_data_villi_multi_test.pytable"
# or:
# test_path  = r"E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV7_9_data_villi_multi_test.pytable"

idx = 4  # change to any valid index

with tables.open_file(train_path, mode="r") as h5:
    # show pytable contents
    print("Nodes in file:")
    for node in h5.root:
        shape = getattr(node, "shape", None)
        dtype = getattr(node, "dtype", None)
        print(f" - {node._v_name:10s} shape={shape} dtype={dtype}")

    tile = h5.root.img[idx]     # (1024, 1024, 3)
    mask = h5.root.mask[idx]    # (1024, 1024) values in {0,1,2}

mask = mask.astype(np.uint8)
print("Sample pixel counts:", {k: int(np.sum(mask == k)) for k in [0, 1, 2]})

# 0 = background (transparent)
# 1 = class1 (filled)
# 2 = class2 (filled)
cmap = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),   # 0: transparent
    (1.0, 0.0, 0.0, 0.35),  # 1: red, semi-transparent
    (0.0, 0.4, 1.0, 0.35),  # 2: blue, semi-transparent
])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(tile)
ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")

# simple legend
legend_handles = [
    mpatches.Patch(color=(1.0, 0.0, 0.0, 0.35), label="class 1"),
    mpatches.Patch(color=(0.0, 0.4, 1.0, 0.35), label="class 2"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=True)

ax.set_title(f"idx={idx} | filled overlay: class1 (red), class2 (blue)")
ax.axis("off")
plt.show()