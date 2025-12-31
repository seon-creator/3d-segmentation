# xai/faithfulness.py

import numpy as np
import torch
import torch.nn.functional as F  # [MOD] dilation용 max_pool3d 사용


# ==============================
# [MOD] 전역 하이퍼파라미터
# ==============================
# GT 마스크를 얼마나 확장(dilation)해서 ROI로 쓸지
DILATE_ITERS = 1       # 예: 3번 dilation → 병변 주변 컨텍스트까지 포함
# 삭제(deletion) 강도: 1.0이면 완전 제거, 0.5면 50%만 약화
DELETION_ALPHA = 1.0
# sufficiency에서 중요 영역 외부를 얼마나 남길지 (0이면 완전 제거)
BACKGROUND_EPS = 0


def _get_topk_mask(cam_np: np.ndarray, k: float):
    """
    cam_np: (H, W, D)
    k: top-k 비율 (0.05 = 상위 5%)

    return: binary mask (H,W,D)
    """
    cam_flat = cam_np.flatten()
    # 상위 k 비율에 해당하는 threshold
    thresh = np.quantile(cam_flat, 1 - k)
    mask_np = (cam_np >= thresh).astype(np.float32)
    return mask_np


# ==============================
# [MOD] GT 기반 ROI (dilated region) 생성
# ==============================
def _build_roi_mask(gt_mask_np: np.ndarray, dilate_iters: int = DILATE_ITERS):
    """
    gt_mask_np: (H,W,D) 0/1 GT 마스크
    dilate_iters: 몇 번 dilation을 할지 (3~5 사이 추천)

    return: roi_np (H,W,D) bool
    """
    if gt_mask_np is None:
        return None

    # (1,1,H,W,D) 텐서로 변환
    m = torch.from_numpy(gt_mask_np.astype(np.float32))[None, None, ...]  # (1,1,H,W,D)

    # 3D max-pooling으로 dilation 구현
    for _ in range(dilate_iters):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)

    roi = (m > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(bool)
    return roi


# ----------------------------------------------------------
# 1) Comprehensiveness (GT 확장 ROI 기반 + Soft Deletion)
# ----------------------------------------------------------
def compute_comprehensiveness(
    model,
    image,
    cam_np,
    class_idx,
    gt_mask_np=None,
    k=0.05,
    deletion_alpha: float = DELETION_ALPHA,  # [MOD]
    dilate_iters: int = DILATE_ITERS,        # [MOD]
):
    """
    model      : segmentation 모델
    image      : (1,C,H,W,D)
    cam_np     : (H,W,D) Grad-CAM / IG / Occlusion 맵 (0~1 정규화)
    class_idx  : 0=WT, 1=TC, 2=ET
    gt_mask_np : (H,W,D) 0/1 numpy (GT lesion mask)
    k          : top-k 비율
    deletion_alpha : 중요 영역을 얼마나 약화할지 (0~1)
                     1.0이면 완전 제거, 0.7이면 70% 감소
    dilate_iters   : GT 마스크를 주변으로 얼마나 확장할지

    return: (comprehensiveness, orig_score, masked_score)
    """

    device = image.device

    # 1) top-k important region mask (전체 CAM 기준)
    mask_np = _get_topk_mask(cam_np, k)  # (H,W,D)
    mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

    # 2) [MOD] GT 기반 ROI (dilated region)
    if gt_mask_np is not None:
        roi_np = _build_roi_mask(gt_mask_np, dilate_iters=dilate_iters)
        roi_t = torch.from_numpy(roi_np.astype(np.float32)).to(device).bool()
    else:
        roi_t = None

    with torch.no_grad():
        # ----------------------------
        # 원본 score (ROI에서 평균)
        # ----------------------------
        logits = model(image)
        prob_map = torch.sigmoid(logits)[0, class_idx]  # (H,W,D)

        if roi_t is not None:
            orig_score = prob_map[roi_t].mean().item()
        else:
            orig_score = prob_map.mean().item()

        # ----------------------------
        # [MOD] Soft Deletion
        #   X\Mk = X * (1 - alpha * M_k)
        #   alpha=1.0 → 완전 제거 (원래 코드)
        #   alpha<1   → 완전 제거 대신 "부분 제거"
        # ----------------------------
        deleted_img = image * (1.0 - deletion_alpha * mask_t)
        deleted_prob_map = torch.sigmoid(model(deleted_img))[0, class_idx]

        if roi_t is not None:
            masked_score = deleted_prob_map[roi_t].mean().item()
        else:
            masked_score = deleted_prob_map.mean().item()

    comp = orig_score - masked_score
    return comp, orig_score, masked_score


# ----------------------------------------------------------
# 2) Sufficiency (GT 확장 ROI 기반 + Soft Keep)
# ----------------------------------------------------------
def compute_sufficiency(
    model,
    image,
    cam_np,
    class_idx,
    gt_mask_np=None,
    k=0.05,
    background_eps: float = BACKGROUND_EPS,  # [MOD]
    dilate_iters: int = DILATE_ITERS,        # [MOD]
):
    """
    중요한 영역만 남겼을 때에도 모델이 충분히 예측할 수 있는지 평가.
    suff = P_orig - P_keep  (작을수록 좋음)

    background_eps:
      - CAM이 낮은 영역(= 중요하지 않다고 간주되는 영역)에 곱해줄 비율
      - 0.0이면 완전 제거 (hard keep)
      - 0.1~0.3 정도면 "배경을 살짝 남기는" soft keep
    """

    device = image.device

    # 1) top-k mask
    mask_np = _get_topk_mask(cam_np, k)
    mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

    # 2) [MOD] ROI 생성
    if gt_mask_np is not None:
        roi_np = _build_roi_mask(gt_mask_np, dilate_iters=dilate_iters)
        roi_t = torch.from_numpy(roi_np.astype(np.float32)).to(device).bool()
    else:
        roi_t = None

    with torch.no_grad():
        logits = model(image)
        prob_map = torch.sigmoid(logits)[0, class_idx]

        if roi_t is not None:
            orig_score = prob_map[roi_t].mean().item()
        else:
            orig_score = prob_map.mean().item()

        # ----------------------------
        # [MOD] Soft Sufficiency
        #   X_k = X * (eps + (1-eps)*M_k)
        #   - 중요 영역(M_k=1): factor = 1
        #   - 비중요 영역(M_k=0): factor = eps
        #   eps>0 → 완전 제거 대신 약하게 남김
        # ----------------------------
        keep_factor = background_eps + (1.0 - background_eps) * mask_t
        kept_img = image * keep_factor

        kept_prob_map = torch.sigmoid(model(kept_img))[0, class_idx]

        if roi_t is not None:
            masked_score = kept_prob_map[roi_t].mean().item()
        else:
            masked_score = kept_prob_map.mean().item()

    suff = orig_score - masked_score
    return suff, orig_score, masked_score


# ----------------------------------------------------------
# 3) Decision Flip Rate (GT 확장 ROI 기반 + Soft Deletion)
# ----------------------------------------------------------
def compute_dfr(
    model,
    image,
    cam_np,
    class_idx,
    gt_mask_np=None,
    k=0.05,
    threshold=0.5,
    deletion_alpha: float = DELETION_ALPHA,  # [MOD]
    dilate_iters: int = DILATE_ITERS,        # [MOD]
):
    """
    중요 영역을 삭제했을 때, 특정 threshold 기준으로
    "양성 → 음성"으로 뒤집히는지 여부를 확인.

    dfr = masked_prob < threshold  (bool)

    여기서 prob은 ROI(= dilated GT) 내부 평균 확률을 사용.
    """

    device = image.device

    # 1) top-k mask
    mask_np = _get_topk_mask(cam_np, k)
    mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

    # 2) [MOD] ROI
    if gt_mask_np is not None:
        roi_np = _build_roi_mask(gt_mask_np, dilate_iters=dilate_iters)
        roi_t = torch.from_numpy(roi_np.astype(np.float32)).to(device).bool()
    else:
        roi_t = None

    with torch.no_grad():
        # 원본 prob
        prob_map = torch.sigmoid(model(image))[0, class_idx]

        if roi_t is not None:
            orig_prob = prob_map[roi_t].mean().item()
        else:
            orig_prob = prob_map.mean().item()

        # [MOD] Soft Deletion 적용
        deleted_img = image * (1.0 - deletion_alpha * mask_t)
        deleted_prob_map = torch.sigmoid(model(deleted_img))[0, class_idx]

        if roi_t is not None:
            masked_prob = deleted_prob_map[roi_t].mean().item()
        else:
            masked_prob = deleted_prob_map.mean().item()

    dfr = masked_prob < threshold
    return dfr, orig_prob, masked_prob