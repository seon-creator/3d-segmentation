import torch
import monai
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from monai import losses

def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "crossentropy":
        return CrossEntropyLoss()

    elif loss_type == "dice":
        return DiceLoss()

    elif loss_type == "dice_focal":
        focal_weight = loss_args.get("focal_weight", 0.3) if loss_args else 0.3
        gamma = loss_args.get("gamma", 2.0) if loss_args else 2.0
        return DiceFocalLoss(focal_weight=focal_weight, gamma=gamma)

    else:
        raise ValueError("Unsupported loss type")


###########################################################################
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        # targets: (N,1,H,W,D) → (N,H,W,D)
        targets = targets.squeeze(1).long()
        loss = self._loss(predictions, targets)
        return loss
    
###########################################################################
# 라벨 0, 1, 2, 3 각각
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(
            to_onehot_y=True,   # target을 one-hot으로 변환
            softmax=True,       # predicted에 softmax 적용
            include_background=True  # 0 class(BG)도 포함
        )

    def forward(self, predicted, target):
        # predicted: (N, 4, H, W, D)
        # target:    (N, 1, H, W, D)  with {0,1,2,3}
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class DiceFocalLoss(nn.Module):
    """
    Multiclass Dice + α * Multiclass Focal Loss
    (for single-channel labels {0,1,2,3})
    """
    def __init__(self, focal_weight=0.3, gamma=2.0, include_background=True):
        super().__init__()
        self.focal_weight = focal_weight
        self.gamma = gamma

        # Multiclass Dice
        self.dice = losses.DiceLoss(
            to_onehot_y=True,        # target: (N,1,...) → one-hot
            softmax=True,            # predicted: softmax over C
            include_background=include_background,
        )

    def forward(self, predicted, target):
        """
        predicted: (N, C=4, H, W, D) logits
        target:    (N, 1, H, W, D) with {0,1,2,3}
        """

        # -------------------------
        # 1) Dice Loss (multiclass)
        # -------------------------
        dice_loss = self.dice(predicted, target)

        # -------------------------
        # 2) Multiclass Focal Loss
        # -------------------------
        # target → (N,H,W,D)
        target_ce = target.squeeze(1).long()

        # log-softmax
        log_pt = F.log_softmax(predicted, dim=1)   # (N,C,H,W,D)
        pt = torch.exp(log_pt)

        # focal weight
        focal_weight = (1 - pt) ** self.gamma

        # NLL loss with focal modulation
        focal_loss = F.nll_loss(
            focal_weight * log_pt,
            target_ce,
            reduction="mean",
        )

        # -------------------------
        # 3) Final Loss
        # -------------------------
        return dice_loss + self.focal_weight * focal_loss