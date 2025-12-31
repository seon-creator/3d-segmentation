import numpy as np

def topk_mask_from_cam(cam, ratio=0.05):
    """
    cam: (H,W,D) numpy, 0~1 정규화된 Grad-CAM
    ratio: 상위 몇 % voxel을 중요 영역으로 볼지 (0~1)
    """
    flat = cam.flatten()
    k = int(max(1, ratio * flat.size))
    # 상위 k 번째 값 기준으로 threshold
    thresh = np.partition(flat, -k)[-k]
    mask = (cam >= thresh).astype(np.uint8)
    return mask


def dice_score(mask1, mask2, eps=1e-6):
    """
    mask1, mask2: (H,W,D) 0/1 numpy arrays
    """
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    s1 = (mask1 > 0).sum()
    s2 = (mask2 > 0).sum()
    if s1 + s2 == 0:
        return np.nan
    return (2.0 * inter) / (s1 + s2 + eps)


def iou_score(mask1, mask2, eps=1e-6):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    uni   = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if uni == 0:
        return np.nan
    return inter / (uni + eps)


def eval_attribution_vol(cam, gt_mask, ratio=0.05):
    """
    cam    : (H,W,D) Grad-CAM (0~1)
    gt_mask: (H,W,D) GT binary mask (0/1)
    ratio  : 상위 ratio 비율을 중요 영역으로 봄
    """
    cam = cam.astype(np.float32)
    gt  = gt_mask.astype(np.uint8)

    # 1) GT 안/밖 CAM 평균
    inside_vals  = cam[gt > 0]
    outside_vals = cam[gt == 0]

    mean_in  = float(inside_vals.mean())  if inside_vals.size  > 0 else 0.0
    mean_out = float(outside_vals.mean()) if outside_vals.size > 0 else 0.0
    ratio_in_out = mean_in / (mean_out + 1e-6)

    # 2) Top-k CAM mask vs GT
    cam_topk = topk_mask_from_cam(cam, ratio=ratio)
    dice_topk = dice_score(cam_topk, gt)
    iou_topk  = iou_score(cam_topk, gt)

    # 3) Pointing game (max CAM 위치가 GT 안인지)
    max_idx_flat = cam.argmax()
    max_idx = np.unravel_index(max_idx_flat, cam.shape)
    pointing_hit = bool(gt[max_idx] > 0)

    return {
        "mean_cam_in_gt": mean_in,
        "mean_cam_out_gt": mean_out,
        "in_out_ratio": ratio_in_out,
        "topk_ratio": ratio,
        "topk_dice": dice_topk,
        "topk_iou": iou_topk,
        "pointing_hit": pointing_hit,
        "pointing_max_index": max_idx,
    }


def evaluate_occlusion_volume(occlusion_map, gt_mask, ratio=0.05):
    """
    Occlusion 맵의 국소화 성능을 평가합니다.
    (Occlusion 맵이 '성능 하락 크기'를 나타내는 중요도 맵이라고 가정)

    :param occlusion_map: (H,W,D) Occlusion 맵 (0~1)
    :param gt_mask: (H,W,D) GT binary mask (0/1)
    :param ratio: 상위 ratio 비율을 중요 영역으로 봄
    """
    occlusion_map = occlusion_map.astype(np.float32)
    gt  = gt_mask.astype(np.uint8)

    # 1) GT 안/밖 Occlusion 평균 (Occlusion 맵의 값이 GT 영역에 집중되었는지)
    inside_vals  = occlusion_map[gt > 0]
    outside_vals = occlusion_map[gt == 0]

    mean_in  = float(inside_vals.mean())  if inside_vals.size  > 0 else 0.0
    mean_out = float(outside_vals.mean()) if outside_vals.size > 0 else 0.0
    ratio_in_out = mean_in / (mean_out + 1e-6)

    # 2) Top-k Occlusion mask vs GT (가장 큰 성능 하락을 유발하는 영역이 GT와 겹치는지)
    # topk_mask_from_cam 함수를 재사용하여 마스크 생성
    occlusion_topk = topk_mask_from_cam(occlusion_map, ratio=ratio)
    dice_topk = dice_score(occlusion_topk, gt)
    iou_topk  = iou_score(occlusion_topk, gt)

    # 3) Pointing game (최대 성능 하락 복셀이 GT 안인지)
    max_idx_flat = occlusion_map.argmax()
    max_idx = np.unravel_index(max_idx_flat, occlusion_map.shape)
    pointing_hit = bool(gt[max_idx] > 0)

    return {
        "mean_occl_in_gt": mean_in,
        "mean_occl_out_gt": mean_out,
        "in_out_ratio": ratio_in_out,
        "topk_ratio": ratio,
        "topk_dice": dice_topk,
        "topk_iou": iou_topk,
        "pointing_hit": pointing_hit,
        "pointing_max_index": max_idx,
    }