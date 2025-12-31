import os # save file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from typing import List, Tuple, Dict

# 시각화
def visualize_slice_2x2(img2d, lab2d, pred2d, cam2d, slice_idx, save_dir, class_mode, modal_name, thr, method):
    # 주석달기
    gt_mask = lab2d > thr
    pred_mask = pred2d > thr

    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img2d, cmap="gray")
    plt.title(f"Original MRI ({modal_name})")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2d, cmap="gray")
    plt.imshow(gt_mask, cmap="Reds", alpha=0.40)
    plt.title(f"GT ({class_mode})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img2d, cmap="gray")

    h, w = img2d.shape
    overlay = np.zeros((h, w, 4))
    overlay[gt_only] = [1, 0, 0, 0.6]
    overlay[pred_only] = [0, 0, 1, 0.6]
    overlay[overlap] = [0, 1, 0, 0.6]

    plt.imshow(overlay)
    plt.title("GT + Prediction")
    plt.axis("off")

    plt.legend(handles=[
        Patch(color="red", label="GT only"),
        Patch(color="blue", label="Pred only"),
        Patch(color="green", label="Overlap")
    ], loc="lower right", fontsize=8, framealpha=0.7)

    plt.subplot(2, 2, 4)
    plt.imshow(img2d, cmap="gray")
    plt.imshow(cam2d, cmap="jet", alpha=0.45)
    plt.title(f"{method} ({class_mode})")
    plt.axis("off")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{class_mode}_{modal_name}_slice_{slice_idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


# Comprehensiveness
def plot_faithfulness_curve(
    k_values: List[float], 
    comp_scores: List[float], 
    suff_scores: List[float], 
    class_mode: str, 
    save_dir: str
):
    """
    Comprehensiveness 및 Sufficiency 점수의 변화 곡선 저장

    :param k_values: 사용된 k (제거 비율) 값 리스트
    :param comp_scores: 각 k 값에 대응하는 Comprehensiveness 점수 리스트
    :param suff_scores: 각 k 값에 대응하는 Sufficiency 점수 리스트
    :param class_mode: 클래스 모드 (예: "WT", "TC", "ET")
    :param save_dir: 이미지를 저장할 디렉토리 경로
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Comprehensiveness (Ablation) Plot
    # 중요 복셀을 제거하면 점수가 하락해야 하므로, 곡선이 아래로 급격히 떨어질수록 좋음
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, comp_scores, marker='o', linestyle='-', color='red', label='Comprehensiveness (Lower is better)')
    plt.title(f'Comprehensiveness Curve for {class_mode}')
    plt.xlabel('Top-k Attribution Ratio (k)')
    plt.ylabel('Score Change (Orig Score - Masked Score)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    comp_path = os.path.join(save_dir, f'{class_mode}_comprehensiveness_curve.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Saved] Comprehensiveness Curve → {comp_path}")
    # 

    # Sufficiency (Retention) Plot
    # 중요하지 않은 복셀을 제거하면 점수가 유지되어야 하므로, 곡선이 위로 평평할수록 좋음
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, suff_scores, marker='x', linestyle='--', color='blue', label='Sufficiency (Higher is better)')
    plt.title(f'Sufficiency Curve for {class_mode}')
    plt.xlabel('Top-k Attribution Ratio (k)')
    plt.ylabel('Retained Score (Orig Score - Masked Score on Complement)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    suff_path = os.path.join(save_dir, f'{class_mode}_sufficiency_curve.png')
    plt.savefig(suff_path, dpi=300)
    plt.close()
    print(f"[Saved] Sufficiency Curve → {suff_path}")
    # 

    return comp_path, suff_path


# Occlusion 전용 시각화 함수
def visualize_slice_2x2_occ(img2d, lab2d, pred2d, occ2d, slice_idx, save_dir, class_mode, modal_name, thr):
    """
    Occlusion 전용 시각화 (Blue-White-Red diverging colormap)
    값 범위: 음수(파랑), 0(흰색), 양수(빨강)
    """

    gt_mask = lab2d > thr
    pred_mask = pred2d > thr

    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    plt.figure(figsize=(10, 10))

    # -------------------------
    # (1) Original MRI
    # -------------------------
    plt.subplot(2, 2, 1)
    plt.imshow(img2d, cmap="gray")
    plt.title(f"Original MRI ({modal_name})")
    plt.axis("off")

    # -------------------------
    # (2) Ground Truth
    # -------------------------
    plt.subplot(2, 2, 2)
    plt.imshow(img2d, cmap="gray")
    plt.imshow(gt_mask, cmap="Reds", alpha=0.40)
    plt.title(f"GT ({class_mode})")
    plt.axis("off")

    # -------------------------
    # (3) GT + Pred Overlay
    # -------------------------
    plt.subplot(2, 2, 3)
    plt.imshow(img2d, cmap="gray")

    h, w = img2d.shape
    overlay = np.zeros((h, w, 4))
    overlay[gt_only] = [1, 0, 0, 0.6]     # red
    overlay[pred_only] = [0, 0, 1, 0.6]   # blue
    overlay[overlap]   = [0, 1, 0, 0.6]   # green

    plt.imshow(overlay)
    plt.title("GT + Prediction")
    plt.axis("off")
    plt.legend(handles=[
        Patch(color="red", label="GT only"),
        Patch(color="blue", label="Pred only"),
        Patch(color="green", label="Overlap")
    ], loc="lower right", fontsize=8, framealpha=0.7)


    # (4) Occlusion diverging colormap
    plt.subplot(2, 2, 4)
    plt.imshow(img2d, cmap="gray")

    vmin, vmax = float(occ2d.min()), float(occ2d.max())

    # Case 1) 값이 모두 같으면 flat map
    if vmin == vmax:
        plt.imshow(np.zeros_like(occ2d), cmap="bwr", alpha=0.55)
    else:
        # Case 2) 양/음 혼합 → 정상 diverging
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        # Case 3) 모두 양수 → center = (vmin + vmax) / 2
        elif vmin >= 0:
            mid = (vmin + vmax) / 2
            norm = TwoSlopeNorm(vmin=vmin, vcenter=mid, vmax=vmax)

        # Case 4) 모두 음수 → center = (vmin + vmax) / 2
        elif vmax <= 0:
            mid = (vmin + vmax) / 2
            norm = TwoSlopeNorm(vmin=vmin, vcenter=mid, vmax=vmax)

        plt.imshow(occ2d, cmap="bwr", norm=norm, alpha=0.55)

    plt.title(f"Occlusion Map ({class_mode})")
    plt.axis("off")

    # -------------------------
    # Save figure
    # -------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{class_mode}_{modal_name}_slice_{slice_idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[Saved OCC]:", out_path)