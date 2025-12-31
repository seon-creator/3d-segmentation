import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from vis import visualize_slice_2x2
from architectures.build_architecture import build_architecture
from xai.method.gradcam_3d import GradCAM3D
from metrics import eval_attribution_vol


# ----------------------------------------
# 설정
# ----------------------------------------
CLASS_MODE = "WT"   # "WT", "TC", "ET"
CONFIG = "experiments/brats2018_21_all/ex1/config.yaml"

MODALITIES_PATH = "xai/sample/modal4/Brats18_CBICA_AAB_1_modalities.pt"
LABEL_PATH = "xai/sample/modal4/Brats18_CBICA_AAB_1_label.pt"

SLICES = [35]       # 보고 싶은 slice index들
VIEW = 1            # 0=axial, 1=sagittal, 2=coronal

# 네 개 모달리티 전체
MODAL_NAMES = ["flair", "t1", "t1ce", "t2"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# CLASS_MODE → 채널 인덱스 매핑
# ----------------------------------------
def get_channel_index(mode: str) -> int:
    mapping = {"WT": 0, "TC": 1, "ET": 2}
    return mapping[mode]


# ----------------------------------------
# (H,W,D)에서 2D slice 추출
# ----------------------------------------
def extract_slice(volume, z, view):
    if view == 0:      # Axial: (H,W,z)
        return volume[:, :, z]
    elif view == 1:    # Sagittal: (z,W,D) → (H,W)로 맞춤
        return volume[z, :, :]
    elif view == 2:    # Coronal: (H,z,D)
        return volume[:, z, :]
    else:
        raise ValueError("Invalid VIEW value (0=axial,1=sagittal,2=coronal)")


# ----------------------------------------
# Grad-CAM 계산
# ----------------------------------------
def run_single_class_cam(gradcam, image, class_idx: int):
    _, cam = gradcam(image=image, class_idx=class_idx, target_mask=None)
    cam_np = cam[0, 0].detach().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np


# ----------------------------------------
# MAIN
# ----------------------------------------
def main(slice_list=None):

    # --------------------------
    # 1) 모델 Load
    # --------------------------
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.eval()

    weight_path = config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin"
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)

    print("Loaded model weights!")

    # Grad-CAM Target Layer
    target_layer = model.decoder.linear_fuse
    gradcam = GradCAM3D(model, target_layer)

    # --------------------------
    # 2) Sample Data Load
    # --------------------------
    modalities = torch.load(MODALITIES_PATH)  # (4,H,W,D)
    label = torch.load(LABEL_PATH)            # (3,H,W,D)

    if isinstance(modalities, np.ndarray):
        modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    print("Loaded sample data!")
    print("Modalities:", modalities.shape)
    print("Label:", label.shape)

    # Batch dimension 추가
    image = modalities.unsqueeze(0).to(DEVICE)     # (1,4,H,W,D)
    label = label.unsqueeze(0).to(DEVICE)          # (1,3,H,W,D)

    image_np = modalities.cpu().numpy()            # (4,H,W,D)
    label_np = label[0].cpu().numpy()              # (3,H,W,D)

    # GT 채널 선택 (WT/TC/ET)
    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]

    # --------------------------
    # 3) 모델 Prediction
    # --------------------------
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    pred_np = (probs[0, class_idx] > 0.5).cpu().numpy()

    # --------------------------
    # 4) Grad-CAM 생성
    # --------------------------
    cam_np = run_single_class_cam(gradcam, image, class_idx)

    depth = cam_np.shape[2]
    if slice_list is None:
        slice_list = [depth // 4, depth // 2, 3 * depth // 4]

    # --------------------------
    # 5) 시각화 저장 폴더
    # --------------------------
    save_dir = os.path.join("xai/result", f"{CLASS_MODE}_GradCAM_view{VIEW}")
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------
    # 6) 모달리티별 시각화
    # --------------------------
    for m_idx, modal_name in enumerate(MODAL_NAMES):

        modal_vol = image_np[m_idx]  # (H,W,D)

        for z in slice_list:

            img2d  = extract_slice(modal_vol, z, VIEW)
            gt2d   = extract_slice(gt_np, z, VIEW)
            pred2d = extract_slice(pred_np, z, VIEW)
            cam2d  = extract_slice(cam_np, z, VIEW)

            visualize_slice_2x2(
                img2d,
                gt2d,
                pred2d,
                cam2d,
                z,
                save_dir,
                CLASS_MODE,
                modal_name,
                0.5,
                "Grad-CAM"
            )

    gradcam.remove_hooks()
    print("Grad-CAM Completed.")


if __name__ == "__main__":
    main(slice_list=SLICES)