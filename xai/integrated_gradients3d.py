import os
import yaml
import torch
import numpy as np
from xai.vis import visualize_slice_2x2

from xai.method.integrated_gradients_3d import IntegratedGradients3D
from architectures.build_architecture import build_architecture
from xai.metrics import eval_attribution_vol
from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 설정
CLASS_MODE = "WT"  # {"WT", "TC", "ET"}
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"

MODALITIES_PATH = "xai/sample/brats_modalities.pt"
LABEL_PATH = "xai/sample/brats_label.pt"

VIEW = 1        # 0 Axial / 1 Sagittal / 2 Coronal
SLICES = [60]  # 원하는 slice index
MODAL_NAMES = ["flair", "t1ce"]


# Class index
def get_channel_index(mode):
    return {"WT": 0, "TC": 1, "ET": 2}[mode]

# slice extraction
def extract_slice(volume, z, view):
    if view == 0:
        return volume[:, :, z]
    elif view == 1:
        return volume[z, :, :]
    elif view == 2:
        return volume[:, z, :]
    else:
        raise ValueError(f"Invalid view: {view}")

# MAIN
def main():

    # 1. 모델 로드
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.eval()

    weight_path = config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin"
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    # 2. 데이터 로드
    modalities = torch.load(MODALITIES_PATH)
    label = torch.load(LABEL_PATH)

    if isinstance(modalities, np.ndarray): modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray): label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    image = modalities.unsqueeze(0).to(DEVICE)   # (1,2,H,W,D)
    label = label.unsqueeze(0).to(DEVICE)        # (1,3,H,W,D)

    image_np = modalities.cpu().numpy()
    label_np = label[0].cpu().numpy()

    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]

    # 3. 모델 예측
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    pred_np = (probs[0, class_idx] > 0.5).cpu().numpy()

    # 4. Integrated Gradients 3D 실행 (클래스 사용)
    print("\nRunning Integrated Gradients 3D...")
    
    # IntegratedGradients3D 클래스 초기화
    ig_calculator = IntegratedGradients3D(model) 
    
    # __call__ 메소드 호출
    ig_map, ig_raw_np = ig_calculator(
        image=image, 
        target_class=class_idx, 
        gt_mask=gt_np, 
        steps=40 # steps는 여기에 명시하거나 클래스 내부 기본값 사용
    ) 
    
    # 5. IG 정량 평가 (ig_map 사용)
    metrics = evaluate_gradcam_volume(ig_map, gt_np, ratio=0.05)

    print("\n========== Integrated Gradients 3D Evaluation ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("========================================================\n")
    
    # 6. 2D slice 시각화
    save_dir = f"xai/result/IG3D_{['axial','sagittal','coronal'][VIEW]}"

    for m_idx, modal_name in enumerate(MODAL_NAMES):
        vol = image_np[m_idx]

        for z in SLICES:
            img2d  = extract_slice(vol, z, VIEW)
            gt2d   = extract_slice(gt_np, z, VIEW)
            pred2d = extract_slice(pred_np, z, VIEW)
            ig2d   = extract_slice(ig_map, z, VIEW) # ig_map 사용

            visualize_slice_2x2(
                img2d, gt2d, pred2d, ig2d,
                z, save_dir, CLASS_MODE, modal_name,
                0.5, "IG"
            )
            
    ig_calculator.remove_hooks() # (옵션)
    print("Integrated Gradients Completed.")


if __name__ == "__main__":
    main()