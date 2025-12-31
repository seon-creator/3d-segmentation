import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from vis import visualize_slice_2x2

from architectures.build_architecture import build_architecture
from xai.method.gradcam_3d import GradCAM3D
from metrics import eval_attribution_vol   # 정량지표 계산

from xai.faithfulness import (
    compute_comprehensiveness,
    compute_sufficiency,
    compute_dfr
)


# 설정
CLASS_MODE = "ET"   # "WT", "TC", "ET"
CONFIG = "experiments/brats_2018_21/segformer3d_mlp/ex7/config.yaml"
MODALITIES_PATH = "xai/sample/modal2/Brats18_CBICA_AAB_1_modalities.pt"
LABEL_PATH = "xai/sample/modal2/Brats18_CBICA_AAB_1_label.pt"

SLICES = [35]

# 테스트할 모달리티만 선택
MODAL_NAMES = ["flair", "t1ce"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [ADD] VIEW 선택 기능 추가
VIEW = 1   # 0=axial, 1=sagittal, 2=coronal


# ----------------------------------------
# CLASS_MODE → 채널 인덱스 매핑
# ----------------------------------------
def get_channel_index(mode: str) -> int:
    mapping = {"WT": 0, "TC": 1, "ET": 2}
    return mapping[mode]


# ----------------------------------------
# [ADD] 3D 볼륨에서 방향(view)에 따라 2D slice 추출
# ----------------------------------------
def extract_slice(volume, idx, view):
    """
    volume: (H, W, D)
    view:
        0 → axial:    volume[:,:,idx]
        1 → sagittal: volume[idx,:,:]
        2 → coronal:  volume[:,idx,:]
    """
    if view == 0:      # axial
        return volume[:, :, idx]
    elif view == 1:    # sagittal
        return volume[idx, :, :]
    elif view == 2:    # coronal
        return volume[:, idx, :]
    else:
        raise ValueError("VIEW must be 0(axial), 1(sagittal), or 2(coronal)")


# ----------------------------------------
# Grad-CAM 실행
# ----------------------------------------
def run_single_class_cam(gradcam, image, class_idx: int):
    _, cam = gradcam(image=image, class_idx=class_idx, target_mask=None)
    cam_np = cam[0, 0].detach().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np


# ============================================================
# MAIN
# ============================================================
def main(slice_list=None):

    # 모델 load
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    model = build_architecture(config).to(DEVICE)
    model.eval()

    weight_path = config["training_parameters"]["checkpoint_save_dir"] + "/pytorch_model.bin"
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded model weights!")

    target_layer = model.decoder.linear_fuse
    gradcam = GradCAM3D(model, target_layer)

    # 데이터 로드
    modalities = torch.load(MODALITIES_PATH)
    label = torch.load(LABEL_PATH)

    if isinstance(modalities, np.ndarray):
        modalities = torch.from_numpy(modalities)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    modalities = modalities.float()
    label = label.float()

    print("Loaded sample data!")
    print("Modalities:", modalities.shape)
    print("Label:", label.shape)

    image = modalities.unsqueeze(0).to(DEVICE)
    label = label.unsqueeze(0).to(DEVICE)

    image_np = modalities.cpu().numpy()
    label_np = label[0].cpu().numpy()

    class_idx = get_channel_index(CLASS_MODE)
    gt_np = label_np[class_idx]    # (H,W,D)

    # 모델 추론
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    pred_np = (probs[0, class_idx] > 0.5).cpu().numpy()

    # Grad-CAM 생성
    cam_np = run_single_class_cam(gradcam, image, class_idx)

    depth = cam_np.shape[2]
    if slice_list is None:
        slice_list = [depth // 4, depth // 2, 3 * depth // 4]

    # 결과 저장 폴더
    save_dir = os.path.join("xai/result", CLASS_MODE + "_Grad-CAM")

    # ----------------------------------------
    # 모달리티별 시각화
    # ----------------------------------------
    for m_idx, modal_name in enumerate(MODAL_NAMES):
        modal_vol = image_np[m_idx]   # (H,W,D)

        for z in slice_list:

            # [ADD] 방향(view)에 맞춘 슬라이스 추출
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