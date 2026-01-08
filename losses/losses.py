import torch
import monai
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from monai import losses

def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "crossentropy":
        return CrossEntropyLoss()

    elif loss_type == "binarycrossentropy":
        return BinaryCrossEntropyWithLogits()

    elif loss_type == "dice":
        return DiceLoss()

    elif loss_type == "diceCE":
        return DiceCELoss()

    elif loss_type == "dice_focal":
        focal_weight = loss_args.get("focal_weight", 0.3) if loss_args else 0.3
        gamma = loss_args.get("gamma", 2.0) if loss_args else 2.0
        return DiceFocalLoss(focal_weight=focal_weight, gamma=gamma)

    elif loss_type == "dice_focal_weighted":
        focal_weight = loss_args.get("focal_weight", 0.1)
        gamma = loss_args.get("gamma", 2.0)
        class_weights = loss_args.get("class_weights", None)
        return WeightedDiceFocalLoss(
            focal_weight=focal_weight,
            gamma=gamma,
            class_weights=class_weights,
        )

    else:
        raise ValueError("Unsupported loss type")


###########################################################################
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


###########################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss
        
###########################################################################
# 라벨 WT, TC, ET 매핑한 경우
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss

# 라벨 0, 1, 2, 3 각각
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._loss = losses.DiceLoss(
#             to_onehot_y=True,   # target을 one-hot으로 변환
#             softmax=True,       # predicted에 softmax 적용
#             include_background=True  # 0 class(BG)도 포함
#         )

#     def forward(self, predicted, target):
#         # predicted: (N, 4, H, W, D)
#         # target:    (N, 1, H, W, D)  with {0,1,2,3}
#         loss = self._loss(predicted, target)
#         return loss


###########################################################################
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss

###########################################################################
class DiceFocalLoss(nn.Module):
    """
    Combined Dice + α * Focal Loss
    """
    def __init__(self, focal_weight=0.3, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice = losses.DiceLoss(to_onehot_y=False, sigmoid=True)
        self.focal = losses.FocalLoss(to_onehot_y=False, gamma=gamma)

    def forward(self, predicted, target):
        dice = self.dice(predicted, target)
        focal = self.focal(predicted, target)
        return dice + self.focal_weight * focal


###########################################################################
class WeightedDiceFocalLoss(nn.Module):
    """
    Dice + focal_weight * WeightedFocalLoss
    class_weights: Tensor/list of per-class weights (C,)
    """
    def __init__(self, focal_weight=0.1, gamma=2.0, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.gamma = gamma

        # Dice (기본은 unweighted)
        self.dice = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

        # class_weights → tensor 변환
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights

    def forward(self, predicted, target):
        """
        predicted: (N, C, D, H, W)
        target:    (N, C, D, H, W) or (N, 1, D, H, W)
        """

        dice_loss = self.dice(predicted, target)

        # -----------------------------
        # 1) Focal Loss 직접 구현 (class_weights 지원)
        # -----------------------------
        # MONAI FocalLoss는 weighted BCE 기반이므로,
        # weighted focal loss를 직접 구현함.
        pt = torch.sigmoid(predicted)     # (N, C, D, H, W)
        ce_loss = F.binary_cross_entropy_with_logits(
            predicted, target, reduction="none"
        )  # (N, C, D, H, W)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 클래스 가중치 적용 (C 차원에 broadcast)
        if self.class_weights is not None:
            cw = self.class_weights.to(predicted.device).view(1, -1, 1, 1, 1)
            focal_loss = focal_loss * cw

        focal_loss = focal_loss.mean()

        # -----------------------------
        # 2) Final Loss = Dice + λ * Focal
        # -----------------------------
        return dice_loss + self.focal_weight * focal_loss