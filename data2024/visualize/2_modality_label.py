import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def visualize_all_modalities_per_label(
    h5_path,
    slice_idx,
    save_dir,
    view="axial",   # NEW
    figsize=(16, 8),
    dpi=300
):
    """
    Visualize a given slice (axial / coronal / sagittal).
    For each label (1–4), generate a 2x4 figure:
      - Row 1: raw modalities
      - Row 2: same modalities with label overlay (green)
    """

    assert view in ["axial", "coronal", "sagittal"], "Invalid view type"

    os.makedirs(save_dir, exist_ok=True)

    modality_order = [
        ("t1", 0),
        ("t1ce", 1),
        ("flair", 3),
        ("t2", 2),
    ]

    # -------------------------
    # Load H5
    # -------------------------
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]  # (C, H, W, D)
        mask = f["mask"][:]    # (H, W, D)

    case_name = os.path.splitext(os.path.basename(h5_path))[0]

    # -------------------------
    # Slice extraction helpers
    # -------------------------
    def extract_slice(vol, msk):
        if view == "axial":
            return vol[:, :, slice_idx], msk[:, :, slice_idx]
        elif view == "coronal":
            return vol[:, slice_idx, :], msk[:, slice_idx, :]
        elif view == "sagittal":
            return vol[slice_idx, :, :], msk[slice_idx, :, :]

    # sanity check
    _, m = extract_slice(image[0], mask)
    present_labels = set(np.unique(m).astype(int))
    required_labels = {1, 2, 3, 4}
    if not required_labels.issubset(present_labels):
        raise ValueError(
            f"{view} slice {slice_idx} does not contain all labels {required_labels}. "
            f"Found: {present_labels}"
        )

    # -------------------------
    # Loop over labels
    # -------------------------
    for label in [1, 2, 3, 4]:
        fig, axes = plt.subplots(2, 4, figsize=figsize)

        for col, (mod_name, ch) in enumerate(modality_order):
            vol_slice, mask_slice = extract_slice(image[ch], mask)
            label_mask = (mask_slice == label)

            # -------- Row 1: raw --------
            axes[0, col].imshow(vol_slice, cmap="gray")
            axes[0, col].set_title(f"{mod_name.upper()} (Raw)", fontsize=10)
            axes[0, col].axis("off")

            # -------- Row 2: overlay --------
            vol_norm = vol_slice.astype(np.float32)
            vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.ptp() + 1e-8)
            rgb = np.stack([vol_norm] * 3, axis=-1)

            rgb[label_mask] = np.array([0.0, 1.0, 0.0])  # green label

            axes[1, col].imshow(rgb)
            axes[1, col].set_title(
                f"{mod_name.upper()} + Label {label}", fontsize=10
            )
            axes[1, col].axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"{case_name}_{view}_slice{slice_idx}_label{label}.png"
        )
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] {save_path}")

h5_path = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED/BraTS-PED-00001-000.h5"
save_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/visualize/img"
slice_idx = 88  # 라벨 1,2,3,4 모두 존재하는 슬라이스

visualize_all_modalities_per_label(
    h5_path=h5_path,
    slice_idx=slice_idx,
    save_dir=save_dir,
    view="coronal"    # axial, coronal, sagittal
)