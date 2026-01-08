import torch
import torch.nn as nn
from typing import Dict
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Activations
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


# 4-class (0,1,2,3) → WT, TC, ET 변환 함수 (validation 수행 시 dice score 계산)
def convert_4class_to_wt_tc_et(label_4c: torch.Tensor) -> torch.Tensor:
    """
    label_4c: (B,H,W,D), values in {0,1,2,3}

    Returns:
        (B,3,H,W,D) where channels are [WT, TC, ET]
    """
    wt = (label_4c == 1) | (label_4c == 2) | (label_4c == 3)
    tc = (label_4c == 1) | (label_4c == 3)
    et = (label_4c == 3)

    return torch.stack([wt, tc, et], dim=1).float()

################################################################################
class SlidingWindowInference:
    def __init__(self, roi: tuple, sw_batch_size: int):
        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def infer(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Returns:
            logits: (B, 4, H, W, D)
        """
        return sliding_window_inference(
            inputs=inputs,
            roi_size=self.roi,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )

    def dice_4class(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if labels.ndim == 5:
            labels = labels.squeeze(1)

        preds = torch.argmax(logits, dim=1)

        dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch"
        )

        dice_metric(
            y_pred=preds.unsqueeze(1),
            y=labels.unsqueeze(1)
        )

        return dice_metric.aggregate().mean().item() * 100

    # WT, TC, ET
    def dice_brats(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if labels.ndim == 5:
            labels = labels.squeeze(1)

        preds = torch.argmax(logits, dim=1)

        preds_wt_tc_et = convert_4class_to_wt_tc_et(preds)
        labels_wt_tc_et = convert_4class_to_wt_tc_et(labels)

        dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch"
        )

        dice_metric(preds_wt_tc_et, labels_wt_tc_et)

        return dice_metric.aggregate().mean().item() * 100


def build_metric_fn(metric_type: str, metric_arg: Dict = None):
    if metric_type == "sliding_window_inference":
        return SlidingWindowInference(
            roi=metric_arg["roi"],
            sw_batch_size=metric_arg["sw_batch_size"],
        )
    else:
        raise ValueError("must be cross sliding_window_inference!")
