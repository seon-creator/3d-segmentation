import os
import glob
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

"""
This preprocessing is for BraTS 2024 PED.

Folder structure (one patient case):
BraTS-PED-00001-000/
  ├─ *-t1n.nii/   → T1 (pre-contrast)
  ├─ *-t1c.nii/   → T1CE (post-contrast)
  ├─ *-t2w.nii/   → T2
  ├─ *-t2f.nii/   → FLAIR
  └─ *-seg.nii    → Segmentation mask

Processing steps:
1. Iterate over all patient folders under the given root directory.
2. Load the four MRI modalities (T1, T1CE, T2, FLAIR) and the segmentation mask.
3. Apply foreground-based intensity normalization to each modality.
4. Stack the modalities into a 4-channel volume and convert to (C, H, W, D) format.
5. Remap segmentation labels and precompute foreground voxel coordinates.
6. Save one HDF5 (.h5) file per patient.

HDF5 file structure:
- image: (4, H, W, D) float32 volume [T1, T1CE, T2, FLAIR]
- mask:  (H, W, D) int64 segmentation mask
- fg_coords_{cls}: foreground voxel coordinates for each class (optional)
- attributes: modality usage and metadata (e.g., number of channels)
"""

# =========================================================
# Intensity normalization
# =========================================================
def _normalize_volume_np(vol):
    """Percentile clipping + Z-score normalization (foreground only)"""
    nz = vol > 0
    if nz.sum() == 0:
        return vol.astype(np.float32)

    v = vol[nz]
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    m, s = v.mean(), v.std()
    if s < 1e-8:
        s = 1e-8

    out = np.zeros_like(vol, dtype=np.float32)
    out[nz] = (v - m) / s
    return out


# =========================================================
# Single patient → H5
# =========================================================
def process_single_patient(patient_dir, output_dir):
    patient_id = os.path.basename(patient_dir.rstrip("/"))
    output_path = os.path.join(output_dir, f"{patient_id}.h5")

    # modality directory mapping
    modality_dirs = {
        "t1": "t1n",
        "t1ce": "t1c",
        "t2": "t2w",
        "flair": "t2f"
    }

    def find_nii(mod):
        d = glob.glob(os.path.join(patient_dir, f"*{mod}.nii"))
        if len(d) == 0:
            raise FileNotFoundError(f"{mod} directory not found in {patient_dir}")
        nii = glob.glob(os.path.join(d[0], "*.nii"))
        if len(nii) == 0:
            raise FileNotFoundError(f"NIfTI not found in {d[0]}")
        return nii[0]

    # Load NIfTI
    t1    = nib.load(find_nii(modality_dirs["t1"])).get_fdata()
    t1ce  = nib.load(find_nii(modality_dirs["t1ce"])).get_fdata()
    t2    = nib.load(find_nii(modality_dirs["t2"])).get_fdata()
    flair = nib.load(find_nii(modality_dirs["flair"])).get_fdata()

    seg_path = glob.glob(os.path.join(patient_dir, "*-seg.nii"))
    if len(seg_path) == 0:
        raise FileNotFoundError(f"Segmentation not found in {patient_dir}")
    seg = nib.load(seg_path[0]).get_fdata()

    # Normalize
    t1    = _normalize_volume_np(t1)
    t1ce  = _normalize_volume_np(t1ce)
    t2    = _normalize_volume_np(t2)
    flair = _normalize_volume_np(flair)

    # Stack → (C, H, W, D)
    image = np.stack([t1, t1ce, t2, flair], axis=-1)
    image = np.transpose(image, (3, 0, 1, 2)).astype(np.float32)

    # Mask processing
    mask = seg.astype(np.int64)
    # mask = np.where(mask == 4, 3, mask)  # brats 2018-2021 버전의 경우 필요 (0,1,2,4) -> (0,1,2,3)

    # Foreground coords
    fg_coords_dict = {}
    for cls in [1, 2, 3]:
        coords = np.argwhere(mask == cls)
        if coords.size > 0:
            fg_coords_dict[f"fg_coords_{cls}"] = coords

    # Save HDF5
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip", compression_opts=4)
        f.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)

        for k, v in fg_coords_dict.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=1)

        f.attrs["use_4modalities"] = True
        f.attrs["num_channels"] = 4
        f.attrs["has_fg_coords"] = len(fg_coords_dict) > 0


# =========================================================
# Root directory → all patients
# =========================================================
def convert_brats_ped_nii_to_h5(root_dir, output_dir):
    patient_dirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for patient_dir in tqdm(patient_dirs, desc="Converting BraTS-PED"):
        try:
            process_single_patient(patient_dir, output_dir)
        except Exception as e:
            print(f"[ERROR] {patient_dir}: {e}")

root_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/sample"
output_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data"

convert_brats_ped_nii_to_h5(root_dir, output_dir)