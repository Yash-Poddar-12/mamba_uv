"""
Degradation Map Estimator (DME) -- Sec 3.3.

Fully analytical, zero learned parameters. Given an image in [0, 1],
scores every non-overlapping P x P patch with a degradation value in
[0, 1] by fusing three physics-grounded signals:

  1. Channel-ratio (Beer-Lambert): red attenuates faster than blue with
     propagation distance, so a low mean-R / mean-B ratio in a patch
     signals depth-induced colour corruption.
  2. Local contrast energy: backscatter raises local mean intensity while
     suppressing local variance, so low std/mean indicates haze.
  3. Dark channel prior: in a clean natural patch, at least one channel
     has near-zero intensity somewhere in the patch. A uniformly elevated
     per-channel minimum indicates a suspended-particle scattering veil.

The three signals are min-max normalised per-image and combined with a
fixed weighted sum, then re-normalised to [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationMapEstimator(nn.Module):
    def __init__(self, patch_size: int = 16, eps: float = 1e-6,
                 weights: tuple = (1 / 3, 1 / 3, 1 / 3)):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.w1, self.w2, self.w3 = weights

    @staticmethod
    def _minmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
        return (x - x_min) / (x_max - x_min + eps)

    @torch.no_grad()  # analytical, no gradient needed w.r.t. its own (nonexistent) params
    def forward(self, img: torch.Tensor):
        """
        img: (B, 3, H, W) float tensor in [0, 1]
        returns:
          D:        (B, nH, nW) degradation score per patch, in [0, 1]
          grid_hw:  (nH, nW)
        """
        B, C, H, W = img.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, (
            f"Image size {(H, W)} must be divisible by patch_size={P}"
        )
        nH, nW = H // P, W // P

        patches = F.unfold(img, kernel_size=P, stride=P)      # (B, C*P*P, L)
        L = patches.shape[-1]
        patches = patches.view(B, C, P * P, L)                 # (B, C, P*P, L)

        # ---- 1) Beer-Lambert channel ratio ----
        mean_r = patches[:, 0].mean(dim=1)                     # (B, L)
        mean_b = patches[:, 2].mean(dim=1)
        ratio = mean_r / (mean_b + self.eps)
        s_ratio = 1.0 - self._minmax(ratio, dim=-1, eps=self.eps)

        # ---- 2) Local contrast energy ----
        gray = patches.mean(dim=1)                             # (B, P*P, L)
        patch_mean = gray.mean(dim=1)
        patch_std = gray.std(dim=1)
        contrast = patch_std / (patch_mean + self.eps)
        s_contrast = 1.0 - self._minmax(contrast, dim=-1, eps=self.eps)

        # ---- 3) Dark channel prior ----
        min_per_channel = patches.min(dim=2).values            # (B, C, L)
        dark_channel = min_per_channel.min(dim=1).values       # (B, L)
        s_dark = self._minmax(dark_channel, dim=-1, eps=self.eps)

        D = self.w1 * s_ratio + self.w2 * s_contrast + self.w3 * s_dark
        D = self._minmax(D, dim=-1, eps=self.eps)              # (B, L)

        return D.view(B, nH, nW), (nH, nW)


def beer_lambert_degrade(
    clean_img: torch.Tensor,
    depth: float = 3.0,
    turbidity: float = 0.5,
    veil_color: tuple = (0.55, 0.75, 0.9),
) -> torch.Tensor:
    """
    Standalone synthetic underwater image formation model, used by
    src/data/synthetic_augment_openai.py to make LLM/diffusion-generated
    images physically consistent with real degraded images before they
    join the training set (Sec 3.5).

        I_deg = I_clean * exp(-c * depth) + veil * (1 - exp(-turbidity * depth))

    with per-channel attenuation c decaying red fastest, blue slowest.

    clean_img: (..., 3, H, W) float tensor in [0, 1]
    returns:   same shape, degraded image in [0, 1]
    """
    attenuation = torch.tensor([0.45, 0.25, 0.12], device=clean_img.device).view(3, 1, 1)
    veil = torch.tensor(veil_color, device=clean_img.device).view(3, 1, 1)

    transmission = torch.exp(-attenuation * depth)             # (3,1,1)
    backscatter = veil * (1 - torch.exp(torch.tensor(-turbidity * depth)))

    degraded = clean_img * transmission + backscatter
    return degraded.clamp(0.0, 1.0)
