import os
import yaml
import torch
import numpy as np
import csv

from xai.vis import visualize_slice_2x2, plot_faithfulness_curve
from architectures.build_architecture import build_architecture
from xai.method.occlusion_3d import OcclusionSensitivity3D
from xai.metrics import eval_attribution_vol
from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr,
)

# ----------------------------------------------------------
# 설정
# ----------------------------------------------------------
CLASS_MODE = "ET"  # {"WT", "TC", "ET"}
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"

MODALITIES_PATH = "xai/sample/brats_modalities.pt"
LABEL_PATH = "xai/sample/brats_label.pt"

VIEW = 0        # 0 Axial / 1 Sagittal / 2 Coronal
BLOCK_SIZE = 8  # Occlusion 블록 크기
SLICES = [115]  # 시각화할 slice index
MODAL_NAMES = ["flair", "t1ce"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Region-based 평가에서 사용할 top-k ratio
REGION_TOPK_RATIO = 0.1

# Faithfulness curve: 0.01 ~ 0.99 (1% 간격)
K_STEPS = np.arange(0.01, 0.1, 0.01).tolist()


# ----------------------------------------------------------
# Class index / Slice Extraction
# ----------------------------------------------------------
def get_channel_index(mode):
    return {"WT": 0, "TC": 1, "ET": 2}[mode]


def extract_slice(volume, z, view):
    if view == 0:
        return volume[:, :, z]
    elif view == 1:
        return volume[z, :, :]
    elif view == 2:
        return volume[:, z, :]
    else:
        raise ValueError(f"Invalid view: {view}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():

    # 1. 모델 load
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.load_state_dict(
        torch.load(
            config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin",
            map_location=DEVICE,
        )
    )
    model.eval()

    # ⭐ OcclusionSensitivity3D 초기화
    occlusion_method = OcclusionSensitivity3D(model, block=BLOCK_SIZE)

    # 2. 입력 데이터 load
    modalities = torch.load(MODALITIES_PATH)
    label = torch.load(LABEL_PATH)

    if isinstance(modalities, np.ndarray):
        modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    image = modalities.unsqueeze(0).to(DEVICE)  # (1,C,H,W,D)
    label = label.unsqueeze(0).to(DEVICE)       # (1,3,H,W,D)

    image_np = modalities.cpu().numpy()         # (C,H,W,D)
    label_np = label[0].cpu().numpy()           # (3,H,W,D)

    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]                 # (H,W,D)

    # 3. 초기 예측 확률 (GT 내부 평균 확률 기준)
    with torch.no_grad():
        logits = model(image)                  # (1,3,H,W,D)
        prob_map = torch.sigmoid(logits)[0, class_idx]  # (H,W,D)

    gt_mask_tensor = torch.from_numpy(gt_np).to(DEVICE).bool()
    if gt_mask_tensor.sum() > 0:
        initial_prob_gt = prob_map[gt_mask_tensor].mean().item()
    else:
        initial_prob_gt = prob_map.mean().item()

    print(f"\n[DEBUG] Initial P(X, {CLASS_MODE}) in GT region: {initial_prob_gt:.4f}")

    # 예측 바이너리 마스크 (시각화용)
    pred_np = (prob_map > 0.5).detach().cpu().numpy()

    # ------------------------------------------------------
    # 4. Occlusion Sensitivity 3D 맵 생성
    # ------------------------------------------------------
    print(f"\nRunning Occlusion Sensitivity 3D (Block={BLOCK_SIZE}) ...")
    occ_map = occlusion_method(image=image, class_idx=class_idx, gt_mask=gt_np)
    # 반환 타입이 tensor일 수 있으므로 numpy로 변환
    if isinstance(occ_map, torch.Tensor):
        occ_map = occ_map.detach().cpu().numpy()

    # 값이 "성능 하락 크기"라고 가정하고, 0~1로 정규화
    occ_min, occ_max = occ_map.min(), occ_map.max()
    occ_map = (occ_map - occ_min) / (occ_max - occ_min + 1e-8)

    # ------------------------------------------------------
    # 5. Region-based Attribution Metrics (Top-k ratio 기준)
    # ------------------------------------------------------
    metrics = eval_attribution_vol(occ_map, gt_np, ratio=REGION_TOPK_RATIO)

    print("\n========== Occlusion Sensitivity 3D Evaluation (Region-based) ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("===================================================================\n")

    print(
        f"[DEBUG] Region Dice@{REGION_TOPK_RATIO:.2f} = {metrics['topk_dice']:.4f}"
    )
    print(
        f"[DEBUG] Region IoU@{REGION_TOPK_RATIO:.2f}  = {metrics['topk_iou']:.4f}"
    )

    # ------------------------------------------------------
    # 6. Faithfulness Curves (Comprehensiveness / Sufficiency / DFR)
    #    → Grad-CAM / IG와 동일한 방식 (occ_map을 중요도 맵으로 사용)
    # ------------------------------------------------------
    comp_scores = []
    suff_scores = []
    dfr_scores = []

    print("\n====== Computing Faithfulness Curves (Occlusion-based) ======")

    for k_ratio in K_STEPS:
        # Comprehensiveness: 중요 영역(top-k)을 제거했을 때 GT 내부 score 감소량
        comp, orig_score_c, masked_score_c = compute_comprehensiveness(
            model=model,
            image=image,
            cam_np=occ_map,
            class_idx=class_idx,
            gt_mask_np=gt_np,
            k=k_ratio,
        )
        comp_scores.append(comp)

        # Sufficiency: 중요 영역만 남겼을 때 score 유지 정도
        suff, orig_score_s, masked_score_s = compute_sufficiency(
            model=model,
            image=image,
            cam_np=occ_map,
            class_idx=class_idx,
            gt_mask_np=gt_np,
            k=k_ratio,
        )
        suff_scores.append(suff)

        # DFR: 중요 영역 제거 후 양성→음성으로 뒤집혔는지 여부
        dfr, orig_prob_d, masked_prob_d = compute_dfr(
            model=model,
            image=image,
            cam_np=occ_map,
            class_idx=class_idx,
            gt_mask_np=gt_np,
            k=k_ratio,
            threshold=0.5,
        )
        dfr_scores.append(int(dfr))

        # 일부 k에서만 디버깅 출력
        if k_ratio in [0.01,0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.75]:
            print(
                f"[DEBUG] k={k_ratio:.2f} | "
                f"Comp={comp:.4f}, Suff={suff:.4f}, DFR={dfr} | "
                f"P_orig={orig_score_c:.4f}, P_masked={masked_score_c:.4f}"
            )

    print("=============================================================\n")

    # ------------------------------------------------------
    # 7. CSV 저장
    # ------------------------------------------------------
    save_dir = "xai/result"
    os.makedirs(save_dir, exist_ok=True)

    # Region-based metrics
    region_csv = os.path.join(save_dir, f"{CLASS_MODE}_Occlusion_region_metrics.csv")
    with open(region_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"[Saved] {region_csv}")

    # Faithfulness curve (k, comp, suff, dfr)
    curve_csv = os.path.join(save_dir, f"{CLASS_MODE}_Occlusion_faithfulness_curve.csv")
    with open(curve_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "comp", "suff", "dfr"])
        for k, c, s, d in zip(K_STEPS, comp_scores, suff_scores, dfr_scores):
            w.writerow([k, c, s, d])
    print(f"[Saved] {curve_csv}")

    # ------------------------------------------------------
    # 8. Comprehensiveness / Sufficiency 곡선 플로팅
    #    (Grad-CAM / IG와 동일한 함수 사용)
    # ------------------------------------------------------
    plot_faithfulness_curve(
        k_values=K_STEPS,
        comp_scores=comp_scores,
        suff_scores=suff_scores,
        class_mode=f"{CLASS_MODE}_Occlusion",
        save_dir=save_dir,
    )

    print("Occlusion Sensitivity XAI Analysis Completed!")


if __name__ == "__main__":
    main()