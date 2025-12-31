import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================================
# 설정 부분
# ==========================================================
ROOT_DIR = "/home/work/3D_/seondeok/project/3d_segmentation/SegFormer3D/data/brats2018_seg_all_v1"
SPLIT = "test"
SAVE_ROOT = "xai/result/check"

GRID = 12   # n x n grid (6이면 36장 per figure)

# VIEW: 0 = Axial, 1 = Sagittal, 2 = Coronal
VIEW = 1   # 여기를 0/1/2로 바꾸면서 사용


# ==========================================================
# 한 슬라이스에 overlay 이미지 생성 함수
# ==========================================================
def generate_overlay(image_2d, label_2d):
    h, w = image_2d.shape
    overlay = np.zeros((h, w, 4), dtype=float)

    overlay[label_2d == 1] = [1, 0, 0, 0.4]  # NE
    overlay[label_2d == 2] = [0, 1, 0, 0.4]  # ED
    overlay[label_2d == 3] = [0, 0, 1, 0.4]  # ET

    return overlay


# ==========================================================
# 뷰에 따라 슬라이스를 꺼내는 함수
# ==========================================================
def get_slice(flair_np, label_np, idx, view):
    """
    view:
      0 -> Axial:    (H, W, D), 슬라이스 = [:, :, z]
      1 -> Sagittal: (H, W, D), 슬라이스 = [x, :, :]
      2 -> Coronal:  (H, W, D), 슬라이스 = [:, y, :]
    """
    if view == 0:  # Axial
        img2d = flair_np[:, :, idx]
        lab2d = label_np[:, :, idx]
    elif view == 1:  # Sagittal
        img2d = flair_np[idx, :, :]
        lab2d = label_np[idx, :, :]
    elif view == 2:  # Coronal
        img2d = flair_np[:, idx, :]
        lab2d = label_np[:, idx, :]
    else:
        raise ValueError(f"Invalid VIEW: {view}. Must be 0, 1, or 2.")
    return img2d, lab2d


def get_view_name(view):
    if view == 0:
        return "axial"
    elif view == 1:
        return "sagittal"
    elif view == 2:
        return "coronal"
    else:
        return "unknown"


# ==========================================================
# n×n Grid로 묶어서 figure 저장하는 함수
# ==========================================================
def save_grid_figure(flair_np, label_np, start_idx, end_idx, case_save_dir, fig_idx, view):
    cols = GRID
    rows = GRID

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(range(start_idx, end_idx)):
        img2d, lab2d = get_slice(flair_np, label_np, idx, view)

        ax = axes[i]
        ax.imshow(img2d, cmap="gray")
        ax.imshow(generate_overlay(img2d, lab2d))
        ax.set_title(f"{get_view_name(view).capitalize()} {idx}")
        ax.axis("off")

    # 나머지 빈 subplot 숨기기
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    os.makedirs(case_save_dir, exist_ok=True)

    save_path = os.path.join(case_save_dir, f"{get_view_name(view)}_grid_{fig_idx:03d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[GRID SAVED] {save_path}")


# ==========================================================
# Tensor or numpy → numpy 변환
# ==========================================================
def to_numpy(x):
    """torch.Tensor → numpy 변환, numpy면 그대로 반환"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# ==========================================================
# 메인 루틴
# ==========================================================
def main():
    csv_path = os.path.join(ROOT_DIR, f"{SPLIT}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        

        # if idx == 1:
        #     break

        data_path = row["data_path"]
        case_name = row["case_name"]
        print(f"[INFO] Case index = {idx:03d}, File name = {case_name}")

        vol_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        lab_fp = os.path.join(data_path, f"{case_name}_label.pt")

        if not (os.path.exists(vol_fp) and os.path.exists(lab_fp)):
            print(f"[WARN] Missing {case_name}, skip.")
            continue

        volume = torch.load(vol_fp)
        label = torch.load(lab_fp)

        # 안전하게 numpy 변환
        volume_np = to_numpy(volume)
        label_np = to_numpy(label)

        # label shape: (1, H, W, D) → (H, W, D)
        if label_np.ndim == 4 and label_np.shape[0] == 1:
            label_np = label_np[0]

        flair_np = volume_np[0]  # Flair 채널 사용

        # -----------------------------
        # VIEW에 따라 depth(슬라이스 개수) 결정
        # -----------------------------
        if VIEW == 0:          # Axial: [:, :, z]
            depth = label_np.shape[2]
        elif VIEW == 1:        # Sagittal: [x, :, :]
            depth = label_np.shape[0]
        elif VIEW == 2:        # Coronal: [:, y, :]
            depth = label_np.shape[1]
        else:
            raise ValueError(f"Invalid VIEW: {VIEW}. Must be 0, 1, or 2.")

        view_name = get_view_name(VIEW)
        case_save_dir = os.path.join(
            SAVE_ROOT,
            f"case_{idx:03d}",
            f"{view_name.upper()}_LABEL_GRID"
        )

        slices_per_fig = GRID * GRID
        fig_idx = 0

        for start_idx in range(0, depth, slices_per_fig):
            end_idx = min(start_idx + slices_per_fig, depth)
            save_grid_figure(flair_np, label_np, start_idx, end_idx, case_save_dir, fig_idx, VIEW)
            fig_idx += 1


if __name__ == "__main__":
    main()