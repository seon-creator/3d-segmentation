import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math


def visualize_labeled_slices_from_h5(
    h5_path,
    save_dir,
    modality="t1ce",
    view="axial",          # NEW
    max_cols=6,
    figsize_per_slice=2.5,
    dpi=300
):
    """
    Visualize only slices that contain foreground labels (>0)
    in axial / coronal / sagittal view and save as a single figure.
    """

    modality_to_channel = {
        "t1": 0,
        "t1ce": 1,
        "t2": 2,
        "flair": 3
    }
    assert modality in modality_to_channel, "Invalid modality name"
    assert view in ["axial", "coronal", "sagittal"], "Invalid view"

    os.makedirs(save_dir, exist_ok=True)

    case_name = os.path.splitext(os.path.basename(h5_path))[0]
    save_path = os.path.join(
        save_dir, f"{case_name}_{modality}_{view}_labeled_slices.png"
    )

    # -------------------------
    # Load H5
    # -------------------------
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]  # (C, H, W, D)
        mask = f["mask"][:]    # (H, W, D)

    vol = image[modality_to_channel[modality]]  # (H, W, D)

    # -------------------------
    # Slice extraction helpers
    # -------------------------
    def get_slice(v, m, idx):
        if view == "axial":      # z
            return v[:, :, idx], m[:, :, idx]
        elif view == "coronal":  # y
            return v[:, idx, :], m[:, idx, :]
        elif view == "sagittal": # x
            return v[idx, :, :], m[idx, :, :]

    # -------------------------
    # Find slices with labels
    # -------------------------
    if view == "axial":
        total_slices = mask.shape[2]
    elif view == "coronal":
        total_slices = mask.shape[1]
    else:  # sagittal
        total_slices = mask.shape[0]

    labeled_slices = [
        i for i in range(total_slices)
        if np.any(get_slice(vol, mask, i)[1] > 0)
    ]

    if len(labeled_slices) == 0:
        print(f"[INFO] No labeled slices found: {case_name}")
        return

    # -------------------------
    # Figure layout
    # -------------------------
    n = len(labeled_slices)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_slice, rows * figsize_per_slice)
    )

    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    # -------------------------
    # Plot
    # -------------------------
    for i, idx in enumerate(labeled_slices):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        slice_img, slice_mask = get_slice(vol, mask, idx)

        ax.imshow(slice_img, cmap="gray")
        ax.imshow(
            slice_mask,
            cmap="jet",
            alpha=0.4,
            vmin=0,
            vmax=mask.max()
        )

        ax.set_title(f"{view.capitalize()} {idx}", fontsize=8)
        ax.axis("off")

    # Hide empty axes
    for i in range(len(labeled_slices), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")

h5_path = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED/BraTS-PED-00001-000.h5"
save_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/visualize/img"

visualize_labeled_slices_from_h5(
    h5_path=h5_path,
    save_dir=save_dir,
    modality="t1ce",
    view="sagittal", # axial, coronal, sagittal
    max_cols=8
)