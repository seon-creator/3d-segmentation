import torch
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # forward hook
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)

        # backward hook
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()   # (B, C, D', H', W')

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()   # (B, C, D', H', W')

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, image, class_idx, target_mask=None):
        """
        image: (B, C, H, W, D)
        class_idx: 0=WT, 1=TC, 2=ET
        target_mask: (B, H, W, D) — ET 마스크 등
        """

        self.model.zero_grad()
        logits = self.model(image)     # (B, 3, H, W, D)

        B, C, H, W, D = logits.shape

        if target_mask is not None:
            target = (logits[:, class_idx] * target_mask).sum() / (target_mask.sum() + 1e-6)
        else:
            target = logits[:, class_idx].mean()

        # backward
        target.backward(retain_graph=True)

        A = self.activations          # (B, C_feat, D', H', W')
        G = self.gradients            # (B, C_feat, D', H', W')

        weights = G.mean(dim=(2, 3, 4), keepdim=True)

        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(H, W, D),
            mode="trilinear",
            align_corners=False
        )

        # normalize to 0~1
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return logits, cam