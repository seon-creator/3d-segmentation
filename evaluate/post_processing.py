import torch

def suppress_small_et(preds_class: torch.Tensor, threshold: int = 500):
    """
    preds_class: (B, H, W, D), values in {0,1,2,3}
    threshold: minimum number of ET voxels to keep ET

    return:
        processed_preds_class: (B, H, W, D)
    """
    preds_class = preds_class.clone()

    B = preds_class.shape[0]
    for b in range(B):
        et_mask = (preds_class[b] == 3)
        if et_mask.sum() < threshold:
            preds_class[b][et_mask] = 0  # ET → Background

    return preds_class