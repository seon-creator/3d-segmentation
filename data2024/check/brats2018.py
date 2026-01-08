import torch
import os

# brats2018 shape 확인용
def inspect_brats2018_pt(case_dir, case_name):
    modality_fp = os.path.join(case_dir, f"{case_name}_modalities.pt")
    label_fp    = os.path.join(case_dir, f"{case_name}_label.pt")

    modalities = torch.load(modality_fp)
    labels     = torch.load(label_fp)

    print(f"[{case_name}]")
    print(f"Modalities shape: {modalities.shape}, dtype: {modalities.dtype}")
    print(f"Label shape:      {labels.shape}, dtype: {labels.dtype}")


# 예시 경로
case_dir = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data_old/brats2018_seg/BraTS2018_training_data/Brats18_2013_0_1"
case_name = "Brats18_2013_0_1"

inspect_brats2018_pt(case_dir, case_name)