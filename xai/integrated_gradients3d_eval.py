import os
import yaml
import torch
import numpy as np
import csv

from xai.vis import visualize_slice_2x2
from architectures.build_architecture import build_architecture
from xai.method.integrated_gradients_3d import IntegratedGradients3D
from xai.metrics import eval_attribution_vol
from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr
)
from xai.vis import plot_faithfulness_curve


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 설정
# ============================
CLASS_MODE = "ET"
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"

MODALITIES_PATH = "xai/sample/brats_modalities.pt"
LABEL_PATH = "xai/sample/brats_label.pt"

VIEW = 1
SLICES = [60]
MODAL_NAMES = ["flair", "t1ce"]

# k 범위 (Grad-CAM 코드와 동일)
K_STEPS = np.arange(0.01, 0.1, 0.01).tolist()


# ============================
# CLASS_MODE → index
# ============================
def get_channel_index(mode):
    return {"WT": 0, "TC": 1, "ET": 2}[mode]


# ============================
# slice extraction
# ============================
def extract_slice(volume, z, view):
    if view == 0:
        return volume[:, :, z]
    elif view == 1:
        return volume[z, :, :]
    elif view == 2:
        return volume[:, z, :]
    else:
        raise ValueError("Invalid view")


# ============================
# MAIN
# ============================
def main():

    # ----------------------------------------------------------------
    # 1) 모델 Load
    # ----------------------------------------------------------------
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.eval()

    checkpoint = config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    # ----------------------------------------------------------------
    # 2) 데이터 Load
    # ----------------------------------------------------------------
    modalities = torch.load(MODALITIES_PATH)
    label = torch.load(LABEL_PATH)

    if isinstance(modalities, np.ndarray): modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray): label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    image = modalities.unsqueeze(0).to(DEVICE)     # (1, C, H, W, D)
    label = label.unsqueeze(0).to(DEVICE)          # (1, 3, H, W, D)

    image_np = modalities.cpu().numpy()
    label_np = label[0].cpu().numpy()

    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]

    # ----------------------------------------------------------------
    # 3) 초기 예측 확률 (GT 영역 내부)
    # ----------------------------------------------------------------
    with torch.no_grad():
        logits = model(image)
        prob_map = torch.sigmoid(logits)[0, class_idx]

        gt_mask_tensor = torch.from_numpy(gt_np).to(DEVICE).bool()

        if gt_mask_tensor.sum() > 0:
            initial_prob = prob_map[gt_mask_tensor].mean().item()
        else:
            initial_prob = prob_map.mean().item()

    print(f"\n[DEBUG] Initial P(X,{CLASS_MODE}) = {initial_prob:.4f}")

    # ----------------------------------------------------------------
    # 4) IG 계산
    # ----------------------------------------------------------------
    print("Running Integrated Gradients...")
    ig_calc = IntegratedGradients3D(model)

    ig_map, ig_raw_np = ig_calc(
        image=image,
        target_class=class_idx,
        gt_mask=gt_np,
        steps=40
    )

    ig_calc.remove_hooks()

    # ----------------------------------------------------------------
    # 5) Region-based 평가
    # ----------------------------------------------------------------
    metrics = eval_attribution_vol(ig_map, gt_np, ratio=0.1)

    print("\n========== IG Region Metrics ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=======================================\n")

    # ----------------------------------------------------------------
    # 6) Faithfulness Curve 계산
    # ----------------------------------------------------------------
    comp_scores, suff_scores, dfr_scores = [], [], []

    print("====== Computing Faithfulness Curves ======")

    for k_ratio in K_STEPS:

        # --- Comprehensiveness ---
        comp, orig_score_comp, masked_score_comp = compute_comprehensiveness(
            model, image, ig_map, class_idx, gt_np, k=k_ratio
        )
        comp_scores.append(comp)

        # --- Sufficiency ---
        suff, orig_score_suff, masked_score_suff = compute_sufficiency(
            model, image, ig_map, class_idx, gt_np, k=k_ratio
        )
        suff_scores.append(suff)

        # --- DFR ---
        dfr, orig_d, masked_d = compute_dfr(
            model, image, ig_map, class_idx, gt_np, k=k_ratio, threshold=0.5
        )
        dfr_scores.append(int(dfr))

        if k_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.75]:
            print(
                f"[DEBUG] k={k_ratio:.2f} | "
                f"Comp={comp:.4f}, Suff={suff:.4f}, DFR={dfr} | "
                f"P_orig={orig_score_comp:.4f}, P_masked={masked_score_comp:.4f}"
            )

    print("==========================================\n")

    # ----------------------------------------------------------------
    # 7) 저장
    # ----------------------------------------------------------------
    save_dir = "xai/result"
    os.makedirs(save_dir, exist_ok=True)

    # region metrics 저장
    csv_region = f"{save_dir}/{CLASS_MODE}_IG_region_metrics.csv"
    with open(csv_region, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in metrics.items():
            w.writerow([k, v])

    # curve 저장
    csv_curve = f"{save_dir}/{CLASS_MODE}_IG_faithfulness_curve.csv"
    with open(csv_curve, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "comp", "suff", "dfr"])
        for k, c, s, d in zip(K_STEPS, comp_scores, suff_scores, dfr_scores):
            w.writerow([k, c, s, d])

    # Plot curve
    plot_faithfulness_curve(
        k_values=K_STEPS,
        comp_scores=comp_scores,
        suff_scores=suff_scores,
        class_mode=f"{CLASS_MODE}_IG",
        save_dir=save_dir
    )

    print("Integrated Gradients Analysis Done!")


if __name__ == "__main__":
    main()