"""
Selective SSM block used by the Mamba-UV encoder (Sec 3.3).

Two backends:
  1. mamba_ssm.Mamba -- the official CUDA-kernel implementation. Fast, but
     only builds on Linux/WSL2 with a matching NVCC + PyTorch/CUDA version
     (see README "Installing mamba-ssm").
  2. NaiveSelectiveSSM -- a pure-PyTorch reimplementation of the same
     selective-scan recurrence (input-dependent Δ, B, C gating). Runs
     anywhere (incl. CPU/Windows) but loops sequentially over the sequence
     length, so it is noticeably slower. Use this to get a model training
     end-to-end before you've built the CUDA kernel, then switch over.

Both expose the same interface: forward(x) with x of shape (B, L, d_model)
-> (B, L, d_model).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveSelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x, delta, A, Bm, Cm):
        # x, delta: (B, L, d_inner); A: (d_inner, d_state); Bm, Cm: (B, L, d_state)
        B, L, d_inner = x.shape
        d_state = A.shape[1]

        deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_inner, d_state))
        deltaB_x = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)

        h = x.new_zeros(B, d_inner, d_state)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y_t = torch.einsum("bds,bs->bd", h, Cm[:, t])
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x_in, res = self.in_proj(x).chunk(2, dim=-1)              # (B, L, d_inner) x2

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]       # causal truncation
        x_conv = F.silu(x_conv.transpose(1, 2))                    # (B, L, d_inner)

        x_dbl = self.x_proj(x_conv)
        delta, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_conv, delta, A, Bm, Cm)
        y = y + x_conv * self.D
        y = y * F.silu(res)
        return self.out_proj(y)


def build_ssm(d_model: int, d_state: int, d_conv: int, expand: int, prefer_mamba_ssm: bool = True):
    """Returns (module, using_official_kernel: bool)."""
    if prefer_mamba_ssm:
        try:
            from mamba_ssm import Mamba
            return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand), True
        except ImportError:
            pass
    return NaiveSelectiveSSM(d_model, d_state, d_conv, expand), False


class MambaUVBlock(nn.Module):
    """One SSM block: selective scan -> channel-mixing MLP, both pre-norm
    with residual connections (Sec 3.3)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, mlp_ratio: int = 4, prefer_mamba_ssm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm, self.using_official_kernel = build_ssm(d_model, d_state, d_conv, expand, prefer_mamba_ssm)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
