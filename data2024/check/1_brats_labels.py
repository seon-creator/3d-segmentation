import os
import h5py
import numpy as np
import csv
from tqdm import tqdm


def scan_h5_labels_to_csv(h5_dir, output_csv_path):
    """
    Scan all .h5 files and record existing label types per file.
    Stop scanning a file early if all labels {0,1,2,3,4} are found.
    """

    TARGET_LABELS = {0, 1, 2, 3, 4}

    h5_files = sorted([
        f for f in os.listdir(h5_dir)
        if f.endswith(".h5")
    ])

    rows = []

    for h5_file in tqdm(h5_files, desc="Scanning H5 files"):
        h5_path = os.path.join(h5_dir, h5_file)
        found_labels = set()

        with h5py.File(h5_path, "r") as f:
            if "mask" not in f:
                print(f"[WARNING] mask not found in {h5_file}")
                continue

            mask_ds = f["mask"]  # h5py dataset (lazy loading)

            # Iterate slice-wise to allow early stopping
            for z in range(mask_ds.shape[2]):
                slice_mask = mask_ds[:, :, z]
                found_labels.update(np.unique(slice_mask))

                # Early stop if all labels are found
                if TARGET_LABELS.issubset(found_labels):
                    break

        file_name = os.path.splitext(h5_file)[0]
        rows.append([file_name, sorted(map(int, found_labels))])

    # Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "labels"])
        for file_name, labels in rows:
            writer.writerow([file_name, str(labels)])


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    h5_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/BraTS2024_PED"
    output_csv = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2024_ped.csv"

    scan_h5_labels_to_csv(h5_dir, output_csv)