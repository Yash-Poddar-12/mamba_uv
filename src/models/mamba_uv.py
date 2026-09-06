import torch
import torch.nn as nn

from src.models.tokenizer import PhysicsOrderedTokenizer
from src.models.ssm_block import MambaUVBlock


class PhysicsConsistencyDecoder(nn.Module):
    """Lightweight decoder, discarded after training, that reconstructs the
    DME's per-token degradation score from the final block's features
    (Sec 3.5, physics consistency auxiliary loss). Operates as a 1D conv
    over the physics-ordered token sequence -- the sequence order already
    matches the corresponding D_sorted target, so no un-sorting is needed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        x = patch_tokens.transpose(1, 2)          # (B, d_model, L)
        out = self.decoder(x).squeeze(1)           # (B, L)
        return torch.sigmoid(out)


class MambaUV(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.tokenizer = PhysicsOrderedTokenizer(
            patch_size=cfg["data"]["patch_size"],
            d_model=m["d_model"],
            use_physics_order=m["use_physics_order"],
            use_register_tokens=m["use_register_tokens"],
            local_perturb_frac=m["local_perturb_frac"],
        )
        self.blocks = nn.ModuleList([
            MambaUVBlock(
                d_model=m["d_model"], d_state=m["d_state"], d_conv=m["d_conv"],
                expand=m["expand"], mlp_ratio=m["mlp_ratio"],
                prefer_mamba_ssm=cfg["mamba_backend"]["prefer_mamba_ssm"],
            )
            for _ in range(m["n_blocks"])
        ])
        self.use_idwp = m["use_idwp"]
        self.num_register_tokens = self.tokenizer.num_register_tokens

        self.classifier = nn.Linear(m["d_model"], cfg["data"]["num_classes"])
        self.physics_decoder = PhysicsConsistencyDecoder(m["d_model"])

    def _encode(self, img: torch.Tensor, shuffle_frac: float = 0.0):
        tokens, D_sorted, grid_hw = self.tokenizer(img, shuffle_frac=shuffle_frac)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens, D_sorted, grid_hw

    def _pool(self, tokens: torch.Tensor, D_sorted: torch.Tensor):
        patch_tokens = tokens[:, self.num_register_tokens:, :]
        if self.use_idwp:
            weights = (1.0 - D_sorted).unsqueeze(-1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            pooled = (patch_tokens * weights).sum(dim=1)
        else:
            pooled = patch_tokens.mean(dim=1)
        return pooled, patch_tokens

    def forward(self, img: torch.Tensor, return_aux: bool = False,
                scan_invariance_shuffle_frac: float = 0.0):
        tokens, D_sorted, _ = self._encode(img)
        pooled, patch_tokens = self._pool(tokens, D_sorted)
        logits = self.classifier(pooled)

        if not return_aux:
            return logits

        aux = {
            "D_target": D_sorted,
            "D_pred": self.physics_decoder(patch_tokens),
        }

        if scan_invariance_shuffle_frac > 0.0:
            tokens_shuf, D_sorted_shuf, _ = self._encode(img, shuffle_frac=scan_invariance_shuffle_frac)
            pooled_shuf, _ = self._pool(tokens_shuf, D_sorted_shuf)
            aux["pooled_ordered"] = pooled
            aux["pooled_shuffled"] = pooled_shuf

        return logits, aux

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
