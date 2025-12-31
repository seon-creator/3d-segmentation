import os
import yaml
import torch
import numpy as np
from xai.vis import visualize_slice_2x2
import csv

from architectures.build_architecture import build_architecture
from xai.method.occlusion_3d import OcclusionSensitivity3D
from xai.metrics import eval_attribution_vol
from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr
)

# ----------------------------------------------------------
# 설정
# ----------------------------------------------------------
CLASS_MODE = "WT"  # {"WT", "TC", "ET"}
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"

MODALITIES_PATH = "xai/sample/brats_modalities.pt"
LABEL_PATH = "xai/sample/brats_label.pt"

VIEW = 0       # 0 Axial / 1 Sagittal / 2 Coronal
BLOCK_SIZE = 4 # Occlusion 블록 크기
SLICES = [115] # 원하는 slice index
MODAL_NAMES = ["flair", "t1ce"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# Class index / Slice Extraction (기존 함수 유지)
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
    model.load_state_dict(torch.load(
        config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin",
        map_location=DEVICE
    ))
    model.eval()

    # ⭐ OcclusionSensitivity3D 클래스 초기화
    occlusion_method = OcclusionSensitivity3D(model, block=BLOCK_SIZE)

    # 2. 입력 데이터 load (+ 타입 보정)
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

    image_np = modalities.cpu().numpy()
    label_np = label[0].cpu().numpy()

    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]

    # 3. 모델 예측 (Dice 기반 Occlusion은 base_pred가 필요)
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    pred_np = (probs[0, class_idx] > 0.5).cpu().numpy()


    # -------------------
    # 4. Occlusion Sensitivity 3D 맵 생성 (클래스 호출)
    # -------------------
    print(f"\nRunning Occlusion Sensitivity 3D (Block={BLOCK_SIZE}) (Dice-based)...")
    occ_map = occlusion_method(image=image, class_idx=class_idx, gt_mask=gt_np)

    # normalize (0~1)
    occ_map = (occ_map - occ_map.min()) / (occ_map.max() - occ_map.min() + 1e-8)

    # -------------------
    # 5. 정량 평가 (eval_attribution_vol 대신 evaluate_gradcam_volume 사용 가정)
    # -------------------
    metrics = eval_attribution_vol(occ_map, gt_np, ratio=0.05)

    print("\n========== Occlusion Sensitivity 3D Evaluation (Region-based) ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=========================================================\n")

    # -------------------
    # 6. Faithfulness metrics (기존 occ_map 사용)
    # -------------------
    # Note: 이 함수들은 일반적으로 확률 변화를 측정하며, Dice 기반 Occlusion 맵의
    # Drop 값(Dice 변화)을 직접적으로 사용하지 않고, 맵의 순위(ranking)만을 이용함.
    # 따라서 Comp/Suff의 입력으로 occ_map을 그대로 사용 가능합니다.
    
    comp, comp_orig, comp_mask = compute_comprehensiveness(model, image, occ_map, class_idx)
    suff, suff_orig, suff_mask = compute_sufficiency(model, image, occ_map, class_idx)
    dfr, dfr_orig, dfr_mask = compute_dfr(model, image, occ_map, class_idx)

    print(f"Comprehensiveness (k=0.05): {comp:.4f}")
    print(f"Sufficiency (k=0.05): {suff:.4f}")
    print(f"Deletion/Flipping Ratio (DFR): {dfr:.4f}")
    
    # -------------------
    # 7. Slices Visualizations (기존 로직 유지)
    # -------------------
    save_dir = f"xai/result/{CLASS_MODE}Occlusion3D_{['axial','sagittal','coronal'][VIEW]}"

    for m_idx, modal_name in enumerate(MODAL_NAMES):
        vol = image_np[m_idx]

        for z in SLICES:
            img2d  = extract_slice(vol, z, VIEW)
            gt2d   = extract_slice(gt_np, z, VIEW)
            pred2d = extract_slice(pred_np, z, VIEW)
            diff2d = extract_slice(occ_map, z, VIEW)

            visualize_slice_2x2(
                img2d, gt2d, pred2d, diff2d,
                z, save_dir, CLASS_MODE, modal_name,
                0.5, "Occ"
            )


if __name__ == "__main__":
    main()