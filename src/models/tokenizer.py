import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dme import DegradationMapEstimator


class GhostModule(nn.Module):
    """Standard GhostNet-style module: a small primary conv produces
    'intrinsic' feature maps, and cheap depthwise convs generate the
    remaining 'ghost' feature maps from them -- used in place of a plain
    3x3 conv in the patch embedding stage (Sec 3.4 compression, applied
    here from the start since the model is meant to be lightweight
    end-to-end)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 ratio: int = 2, dw_size: int = 3, stride: int = 1):
        super().__init__()
        self.out_ch = out_ch
        init_ch = math.ceil(out_ch / ratio)
        new_ch = init_ch * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.GELU(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, dw_size, 1, dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch]


class GhostPatchEmbed(nn.Module):
    """3x3 (Ghost) conv over the full-resolution image, then non-overlapping
    P x P patches are flattened and linearly projected to d_model."""

    def __init__(self, patch_size: int, d_model: int, pre_channels: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.ghost = GhostModule(in_ch=3, out_ch=pre_channels, kernel_size=3, dw_size=3, stride=1)
        self.proj = nn.Linear(pre_channels * patch_size * patch_size, d_model)

    def forward(self, img: torch.Tensor):
        B, C, H, W = img.shape
        feat = self.ghost(img)                                   # (B, pre_channels, H, W)
        P = self.patch_size
        patches = F.unfold(feat, kernel_size=P, stride=P)         # (B, pre_channels*P*P, L)
        patches = patches.transpose(1, 2)                          # (B, L, pre_channels*P*P)
        tokens = self.proj(patches)                                # (B, L, d_model)
        return tokens, (H // P, W // P)


class PhysicsOrderedTokenizer(nn.Module):
    """Combines the DME + Ghost patch embedding, sorts tokens by ascending
    degradation, injects turbidity-conditioned register tokens, and applies
    training-time local-rank perturbation (Sec 3.3)."""

    def __init__(self, patch_size: int, d_model: int, use_physics_order: bool = True,
                 use_register_tokens: bool = True, local_perturb_frac: float = 0.15,
                 pre_channels: int = 16):
        super().__init__()
        self.d_model = d_model
        self.use_physics_order = use_physics_order
        self.use_register_tokens = use_register_tokens
        self.local_perturb_frac = local_perturb_frac
        self.num_register_tokens = 2 if use_register_tokens else 0

        self.dme = DegradationMapEstimator(patch_size=patch_size)
        self.patch_embed = GhostPatchEmbed(patch_size, d_model, pre_channels)

        if use_register_tokens:
            self.reg_mean_proj = nn.Linear(1, d_model)
            self.reg_std_proj = nn.Linear(1, d_model)

    def _local_perturb(self, order: torch.Tensor, window: int = 5) -> torch.Tensor:
        """Swap ~local_perturb_frac of tokens with a neighbour within a small
        window of their sorted rank -- prevents over-reliance on a perfect
        ordering (Sec 3.3)."""
        B, L = order.shape
        perturbed = order.clone()
        num_swaps = max(1, int(self.local_perturb_frac * L))
        for b in range(B):
            positions = torch.randperm(L, device=order.device)[:num_swaps]
            for pos in positions.tolist():
                lo, hi = max(0, pos - window // 2), min(L - 1, pos + window // 2)
                partner = torch.randint(lo, hi + 1, (1,), device=order.device).item()
                perturbed[b, pos], perturbed[b, partner] = (
                    perturbed[b, partner].clone(), perturbed[b, pos].clone()
                )
        return perturbed

    def _shuffle_fraction(self, order: torch.Tensor, frac: float) -> torch.Tensor:
        """Used by the scan-order-invariance loss: randomly reorder `frac`
        of the tokens, breaking the physics-guided sequence."""
        B, L = order.shape
        shuffled = order.clone()
        num_shuffle = max(1, int(frac * L))
        for b in range(B):
            idx = torch.randperm(L, device=order.device)[:num_shuffle]
            shuffled[b, idx] = shuffled[b, idx][torch.randperm(num_shuffle, device=order.device)]
        return shuffled

    def forward(self, img: torch.Tensor, shuffle_frac: float = 0.0):
        """
        Returns:
          tokens:   (B, L + num_register_tokens, d_model)
          D_sorted: (B, L) degradation score per patch token, same order as
                    the patch-token part of `tokens`
          grid_hw:  (nH, nW)
        """
        B = img.shape[0]
        D, grid_hw = self.dme(img)                  # (B, nH, nW)
        D_flat = D.view(B, -1)
        L = D_flat.shape[1]

        patch_tokens, _ = self.patch_embed(img)      # (B, L, d_model)

        if self.use_physics_order:
            order = torch.argsort(D_flat, dim=1, descending=False)
        else:
            order = torch.arange(L, device=img.device).unsqueeze(0).expand(B, -1).clone()

        if self.training and self.local_perturb_frac > 0 and shuffle_frac == 0.0:
            order = self._local_perturb(order)

        if shuffle_frac > 0.0:
            order = self._shuffle_fraction(order, shuffle_frac)

        gather_idx = order.unsqueeze(-1).expand(-1, -1, self.d_model)
        tokens_sorted = torch.gather(patch_tokens, 1, gather_idx)
        D_sorted = torch.gather(D_flat, 1, order)

        if self.use_register_tokens:
            mean_D = D_flat.mean(dim=1, keepdim=True)
            std_D = D_flat.std(dim=1, keepdim=True)
            reg1 = self.reg_mean_proj(mean_D).unsqueeze(1)
            reg2 = self.reg_std_proj(std_D).unsqueeze(1)
            tokens_sorted = torch.cat([reg1, reg2, tokens_sorted], dim=1)

        return tokens_sorted, D_sorted, grid_hw
