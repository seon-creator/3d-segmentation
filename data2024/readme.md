BraTS2024_PED : BraTS 2024 pediatric dataset
- https://arxiv.org/abs/2404.15009
- 1개 샘플 mask가 manual이라고 하는데 시각화해보기(BraTS-PED-00085-000)
- 1개 샘플(BraTS-PED-00255-000) T1CE 모달리티가 없어서 제외함


# preprocess.py : raw data -> .h5 dataset
raw dataset structure (one patient case):
BraTS-PED-00001-000/
  ├─ *-t1n.nii/   → T1 (pre-contrast)
  ├─ *-t1c.nii/   → T1CE (post-contrast)
  ├─ *-t2w.nii/   → T2
  ├─ *-t2f.nii/   → FLAIR
  └─ *-seg.nii    → Segmentation mask

h5 data structure
HDF5 file structure:
- image: (4, H, W, D) float32 volume [T1, T1CE, T2, FLAIR]
- mask:  (H, W, D) int64 segmentation mask
- fg_coords_{cls}: foreground voxel coordinates for each class (optional)
- attributes: modality usage and metadata (e.g., number of channels)

# BraTS 2024 shape
[240, 240, 155]