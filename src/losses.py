from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Alternative to label-smoothed CE for severe class imbalance.
    Not the paper's default (Sec 3.5 specifies label-smoothed CE), but
    exposed via configs/config.yaml -> imbalance.loss_type: 'focal'."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


class MambaUVLoss(nn.Module):
    """Primary classification loss + Sec 3.5's two novel auxiliary losses."""

    def __init__(self, cfg: dict, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        imb = cfg["imbalance"]
        loss_cfg = cfg["losses"]

        if imb["loss_type"] == "focal":
            self.primary = FocalLoss(weight=class_weights, gamma=imb["focal_gamma"])
        else:
            self.primary = nn.CrossEntropyLoss(
                weight=class_weights if imb["use_class_weighted_loss"] else None,
                label_smoothing=imb["label_smoothing"],
            )

        self.use_physics = loss_cfg["use_physics_consistency_loss"]
        self.lambda_physics = loss_cfg["lambda_physics"]
        self.use_scan_inv = loss_cfg["use_scan_invariance_loss"]
        self.lambda_scan_inv = loss_cfg["lambda_scan_invariance"]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux: Optional[dict] = None):
        loss = self.primary(logits, targets)
        logs = {"loss_cls": loss.item()}

        if aux is not None:
            if self.use_physics and "D_pred" in aux:
                l_phys = F.mse_loss(aux["D_pred"], aux["D_target"])
                loss = loss + self.lambda_physics * l_phys
                logs["loss_physics"] = l_phys.item()

            if self.use_scan_inv and "pooled_shuffled" in aux:
                l_scan = F.mse_loss(aux["pooled_shuffled"], aux["pooled_ordered"])
                loss = loss + self.lambda_scan_inv * l_scan
                logs["loss_scan_inv"] = l_scan.item()

        logs["loss_total"] = loss.item()
        return loss, logs
