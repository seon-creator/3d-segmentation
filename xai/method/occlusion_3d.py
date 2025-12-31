# xai/method/occlusion_3d.py

import torch
import numpy as np

class OcclusionSensitivity3D:
    """
    3D Occlusion Sensitivity 맵을 생성합니다.
    (분할(Segmentation) 모델의 Dice Score 변화를 기준으로 귀속 맵을 생성합니다.)
    """
    def __init__(self, model, block: int = 8, fill_value: float = 0.0):
        """
        :param model: 평가할 3D Segmentation 모델 (torch.nn.Module)
        :param block: Occlusion 블록의 크기 (block x block x block)
        :param fill_value: Occlusion 영역을 채울 값 (0.0 또는 평균값 등)
        """
        self.model = model
        self.block = block
        self.fill_value = fill_value
        self.model.eval()

    def _dice_score(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Dice Score 계산 함수"""
        inter = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        union = pred_mask.sum() + gt_mask.sum()
        return (2 * inter) / (union + 1e-6)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, class_idx: int, gt_mask: np.ndarray) -> np.ndarray:
        """
        Occlusion Sensitivity 맵을 생성하고 반환합니다.

        :param image: 입력 볼륨 (B, C, H, W, D). B=1 이어야 합니다.
        :param class_idx: 평가할 클래스 채널 인덱스
        :param gt_mask: GT 이진 마스크 (H, W, D) numpy 배열
        :return: Occlusion Sensitivity Heatmap (H, W, D)
        """
        
        # 입력 유효성 검사
        if image.ndim != 5 or image.shape[0] != 1:
            raise ValueError("Input image must be (1, C, H, W, D).")

        _, C, H, W, D = image.shape
        device = image.device
        heatmap = np.zeros((H, W, D), dtype=np.float32)

        # 1. Baseline Prediction 및 Dice Score 계산
        logits = self.model(image)
        probs = torch.sigmoid(logits)[0, class_idx].cpu().numpy()
        base_pred_mask = (probs > 0.5).astype(np.uint8)
        base_dice = self._dice_score(base_pred_mask, gt_mask)

        # 2. Sliding Occlusion
        for i in range(0, H, self.block):
            for j in range(0, W, self.block):
                for k in range(0, D, self.block):
                    
                    # Occlusion 영역 설정
                    slice_h = slice(i, min(i + self.block, H))
                    slice_w = slice(j, min(j + self.block, W))
                    slice_d = slice(k, min(k + self.block, D))

                    x_occ = image.clone()
                    # 해당 블록을 fill_value로 채움
                    x_occ[:, :, slice_h, slice_w, slice_d] = self.fill_value 

                    # 3. Occluded Prediction 및 Dice Score 계산
                    logits_occ = self.model(x_occ)
                    probs_occ = torch.sigmoid(logits_occ)[0, class_idx].cpu().numpy()
                    pred_occ_mask = (probs_occ > 0.5).astype(np.uint8)

                    occ_dice = self._dice_score(pred_occ_mask, gt_mask)
                    
                    # 4. Sensitivity (Drop in Dice Score) 계산
                    # Drop이 클수록 해당 영역이 중요함
                    drop = base_dice - occ_dice
                    
                    # Drop이 음수인 경우 (Occlusion이 오히려 성능을 높이는 경우) 0으로 설정
                    # Occlusion Sensitivity는 중요 영역을 찾는 것이 목적이므로, 성능 저하만을 관심 대상으로 함
                    drop = max(drop, 0)

                    # Heatmap에 결과 기록
                    heatmap[slice_h, slice_w, slice_d] = drop

        return heatmap