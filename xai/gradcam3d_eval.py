import os
import yaml
import torch
import numpy as np
import csv

from architectures.build_architecture import build_architecture
from xai.method.gradcam_3d import GradCAM3D
from xai.metrics import eval_attribution_vol
from xai.vis import plot_faithfulness_curve

from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr
)

# ============================
# 설정
# ============================
CLASS_MODE = "WT"
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"

MODALITIES_PATH = "xai/sample/brats_modalities.pt"
LABEL_PATH = "xai/sample/brats_label.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Faithfulness curve: 0.01 ~ 0.99
K_STEPS = np.arange(0.01, 0.1, 0.01).tolist()


# ============================
# CLASS_MODE → channel index
# ============================
def get_channel_index(mode: str) -> int:
    mapping = {"WT": 0, "TC": 1, "ET": 2}
    return mapping[mode]


# ============================
# Grad-CAM 실행
# ============================
def run_single_class_cam(gradcam, image, class_idx: int):
    _, cam = gradcam(image=image, class_idx=class_idx, target_mask=None)
    cam_np = cam[0, 0].detach().cpu().numpy()

    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np


# ============================
# MAIN
# ============================
def main():

    # 1) Load config & model
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.eval()

    weight_path = config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin"
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)

    # Grad-CAM
    target_layer = model.decoder.linear_fuse
    gradcam = GradCAM3D(model, target_layer)

    # 2) 샘플 데이터 로드
    modalities = torch.load(MODALITIES_PATH)
    label = torch.load(LABEL_PATH)

    if isinstance(modalities, np.ndarray):
        modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    image = modalities.unsqueeze(0).to(DEVICE)
    label = label.unsqueeze(0).to(DEVICE)

    # numpy version
    label_np = label[0].cpu().numpy()

    # WT / TC / ET 선택
    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]

    # ============================
    # 3) 초기 예측 확률 계산 (GT 내부 avg)
    # ============================
    with torch.no_grad():
        logits = model(image)
        prob_map = torch.sigmoid(logits)[0, class_idx]

        gt_mask_tensor = torch.from_numpy(gt_np).to(DEVICE).bool()

        if gt_mask_tensor.sum() > 0:
            initial_prob_gt = prob_map[gt_mask_tensor].mean().item()
        else:
            initial_prob_gt = prob_map.mean().item()

    print(f"\n[DEBUG] Initial P(X, {CLASS_MODE}) in GT region: {initial_prob_gt:.4f}")

    # ============================
    # 4) Grad-CAM 생성
    # ============================
    cam_np = run_single_class_cam(gradcam, image, class_idx)
    gradcam.remove_hooks()

    # ============================
    # 5) Grad-CAM Region Metrics
    # ============================
    metrics = eval_attribution_vol(cam_np, gt_np, ratio=0.1)

    print("\n========== Region-based Grad-CAM Metrics ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("====================================================\n")

    print(f"[DEBUG] Region Dice@0.10 = {metrics['topk_dice']:.4f}")
    print(f"[DEBUG] Region IoU@0.10  = {metrics['topk_iou']:.4f}")

    # ============================
    # 6) Faithfulness Curve 계산
    # ============================
    comp_scores = []
    suff_scores = []
    dfr_scores = []

    print("\n====== Faithfulness Curve Calculation ======")

    for k_ratio in K_STEPS:

        # --- Comprehensiveness ---
        comp, orig_score_comp, masked_score_comp = compute_comprehensiveness(
            model, image, cam_np, class_idx, gt_np, k=k_ratio
        )
        comp_scores.append(comp)

        # --- Sufficiency ---
        suff, orig_score_suff, masked_score_suff = compute_sufficiency(
            model, image, cam_np, class_idx, gt_np, k=k_ratio
        )
        suff_scores.append(suff)

        # --- DFR ---
        dfr, orig_prob_dfr, masked_prob_dfr = compute_dfr(
            model, image, cam_np, class_idx, gt_np, k=k_ratio, threshold=0.5
        )
        dfr_scores.append(int(dfr))

        # DEBUG 일부 출력
        if k_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.75]:
            print(
                f"[DEBUG] k={k_ratio:.2f} | "
                f"Comp={comp:.4f}, Suff={suff:.4f}, DFR={dfr} | "
                f"P_orig={orig_score_comp:.4f}, P_masked={masked_score_comp:.4f}"
            )

    print("\n==============================================\n")

    # ============================
    # 7) CSV 저장
    # ============================
    save_dir = "xai/result"
    os.makedirs(save_dir, exist_ok=True)

    # region metrics
    region_csv = f"{save_dir}/{CLASS_MODE}_region_metrics.csv"
    with open(region_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"[Saved] {region_csv}")

    # faithfulness curve
    curve_csv = f"{save_dir}/{CLASS_MODE}_faithfulness_curve.csv"
    with open(curve_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "comp", "suff", "dfr"])
        for k, c, s, d in zip(K_STEPS, comp_scores, suff_scores, dfr_scores):
            w.writerow([k, c, s, d])

    print(f"[Saved] {curve_csv}")

    # ============================
    # 8) Plot curves
    # ============================
    plot_faithfulness_curve(
        k_values=K_STEPS,
        comp_scores=comp_scores,
        suff_scores=suff_scores,
        class_mode=CLASS_MODE,
        save_dir=save_dir
    )

    print("XAI Analysis Completed!")


if __name__ == "__main__":
    main()