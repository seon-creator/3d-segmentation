# integrated_gradients_3d.py

import torch
import numpy as np

class IntegratedGradients3D:
    """
    3D Segmentation 모델을 위한 Integrated Gradients (IG) 구현.

    주요 기능:
    1. Baseline (보통 Zero)에서부터 입력까지의 경로를 따라 그래디언트를 적분
    2. Segmentation의 특성에 맞게 특정 클래스 (target_class)의
       GT 마스크 영역 (gt_mask)에 대해서만 그래디언트를 집중 계산
    """
    def __init__(self, model):
        """
        :param model: 3D Segmentation PyTorch 모델
        """
        self.model = model
        self.model.eval()
        self.hooks = []  # 후크 관리 (현재 IG는 후크 불필요하나 구조 통일을 위해 유지)

    def __call__(self, image: torch.Tensor, target_class: int, gt_mask: np.ndarray, steps: int = 40):
        """
        Integrated Gradients 맵을 계산

        :param image: 입력 3D 볼륨 (1, C, H, W, D) torch.Tensor
        :param target_class: 타겟 클래스 인덱스 (0=WT, 1=TC, 2=ET)
        :param gt_mask: 타겟 클래스의 Ground Truth 마스크 (H, W, D) numpy.ndarray
        :param steps: 적분에 사용할 샘플링 단계 수
        :return: (ig_map_normalized, ig_raw)
            - ig_map_normalized: 절대값 합산 후 0~1로 정규화된 3D IG 맵 (H, W, D) numpy.ndarray
            - ig_raw: 원본 IG 텐서 (1, C, H, W, D) numpy.ndarray
        """
        device = image.device
        baseline = torch.zeros_like(image).to(device)

        # 1. 경로 샘플링
        scaled_inputs = [
            baseline + (i / steps) * (image - baseline)
            for i in range(steps + 1)
        ]

        total_gradients = torch.zeros_like(image).to(device)

        # 2. GT 마스크 텐서 준비 (그래디언트 타겟 영역 지정)
        # GT mask를 torch tensor로 변환 및 차원 확장 (1, 1, H, W, D)
        gt_mask_t = torch.tensor(gt_mask, device=device).float()
        gt_mask_t = gt_mask_t.unsqueeze(0).unsqueeze(0)

        # 3. 그래디언트 계산 및 적분
        for scaled in scaled_inputs:
            scaled = scaled.clone().requires_grad_(True)
            out = self.model(scaled)  # (1, 3, H, W, D)

            # 타겟 클래스 로짓과 GT 마스크를 곱하여 관심 영역에 집중
            class_logit = out[:, target_class:target_class+1, :, :, :]  # (1, 1, H, W, D)
            loss = (class_logit * gt_mask_t).sum()  # GT 영역의 활성화 합산

            self.model.zero_grad()
            loss.backward()
            total_gradients += scaled.grad

        # 4. 최종 IG 계산: (입력 - Baseline) * 평균 그래디언트
        avg_grad = total_gradients / (steps + 1)
        ig_raw = (image - baseline) * avg_grad

        ig_raw_np = ig_raw.detach().cpu().numpy()

        # 5. 시각화 및 정량 지표용 최종 맵 계산 (채널 합산 및 정규화)
        # (1, C, H, W, D) -> (H, W, D)
        ig_map = np.abs(ig_raw_np).sum(axis=1)[0]
        
        # Normalize
        ig_map_normalized = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)

        return ig_map_normalized, ig_raw_np
    
    def remove_hooks(self):
        """
        클래스 구조 통일을 위한 더미 함수
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []