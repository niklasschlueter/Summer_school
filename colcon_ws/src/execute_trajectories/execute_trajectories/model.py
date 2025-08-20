import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

class BlockNormalizer24:
    
    def __init__(self, idx_ft):
        self.idx_ft = np.array(idx_ft)
        self.mean = None
        self.std = None

    def fit(self, X):
        Xc = X.copy()
        ft = Xc[:, self.idx_ft]
        lo = np.percentile(ft, 1, axis=0)
        hi = np.percentile(ft, 99, axis=0)
        Xc[:, self.idx_ft] = np.clip(ft, lo, hi)
        self.mean = Xc.mean(axis=0)
        self.std  = Xc.std(axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ============== Normalization ==============
class BlockNormalizer:
    """Per-feature z-score with optional robust clipping specifically for FT.
    Exposes transform(x), inverse_transform(x), and can be saved/loaded.
    """
    def __init__(self, idx_q, idx_qd, idx_eef, idx_ft):
        self.idx_q = np.array(idx_q)
        self.idx_qd = np.array(idx_qd)
        self.idx_eef = np.array(idx_eef)
        self.idx_ft = np.array(idx_ft)
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        Xc = X.copy()
        # Robust clip FT to 1st-99th percentile before stats
        ft = Xc[:, self.idx_ft]
        lo = np.percentile(ft, 1, axis=0)
        hi = np.percentile(ft, 99, axis=0)
        ft = np.clip(ft, lo, hi)
        Xc[:, self.idx_ft] = ft

        self.mean = Xc.mean(axis=0)
        self.std = Xc.std(axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        return (X - self.mean) / self.std

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean

    def state_dict(self):
        return {
            'idx_q': self.idx_q.tolist(),
            'idx_qd': self.idx_qd.tolist(),
            'idx_eef': self.idx_eef.tolist(),
            'idx_ft': self.idx_ft.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }

    @staticmethod
    def from_state_dict(sd):
        norm = BlockNormalizer(sd['idx_q'], sd['idx_qd'], sd['idx_eef'], sd['idx_ft'])
        norm.mean = np.array(sd['mean'], dtype=np.float32)
        norm.std = np.array(sd['std'], dtype=np.float32)
        return norm



class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ln = nn.LayerNorm(dim)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.net(self.ln(x))
        return x + self.do(y)


class ILPolicy(nn.Module):
    """
    Input x: [q(6), qd(6), eef_pos(3), ft(6)] -> shape (B, 21)
    Output: q_des (6)
    """
    def __init__(self, input_dim: int = 21, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.0, output_dim: int = 6):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            
        )
        # self.head = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.Linear(hidden, output_dim),
        #     nn.Tanh())

        # Xavier init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = nn.GELU()(self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        y = self.head(h)
        return y


# -------- small utils ----------
def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=2, act=nn.SiLU, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), act(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(init_linear)
    def forward(self, x): return self.net(x)

class GatedResBlock(nn.Module):
    """Pre-LN gated residual FFN: x + (V * sigmoid(G))."""
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.SiLU(),
            nn.Linear(mult * dim, 2 * dim),
        )
        self.do = nn.Dropout(dropout)
        self.apply(init_linear)
    def forward(self, x):
        h = self.ff(self.norm(x))
        v, g = h.chunk(2, dim=-1)
        return x + self.do(v * torch.sigmoid(g))


# -------- improved policy ----------
class ILPolicyV2(nn.Module):
    """
    Feature-aware residual MLP with optional delta head.

    Default input layout (24 dims): [q(0:6), qd(6:12), tau(12:18), ft(18:24)]
    Output: q_cmd(6)  (absolute by default; delta if predict_delta=True)
    """
    def __init__(
        self,
        input_dim: int = 24,
        output_dim: int = 6,
        hidden: int = 512,
        layers: int = 5,
        dropout: float = 0.05,
        max_delta: float = 0.15,      # rad per step, used if predict_delta=True
        group_slices: Optional[dict[str, slice]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_delta = max_delta

        # Default group layout for 24-d native features
        if group_slices is None and input_dim == 24:
            group_slices = {
                "q":   slice(0, 6),
                "qd":  slice(6, 12),
                "tau": slice(12, 18),
                "ft":  slice(18, 24),
            }
        self.group_slices = group_slices

        # Per-block encoders → project each group to a common width, then concat
        enc_width = hidden // (len(group_slices) if group_slices else 1)
        self.encoders = nn.ModuleDict()
        if group_slices is None:
            self.encoders["x"] = MLP(input_dim, hidden, hidden, depth=2, dropout=dropout)
            trunk_in = hidden
        else:
            for name, sl in group_slices.items():
                self.encoders[name] = MLP(sl.stop - sl.start, max(64, enc_width), enc_width, depth=2, dropout=dropout)
            trunk_in = enc_width * len(group_slices)

        # Trunk: gated residual blocks (pre-LN)
        blocks = []
        # project to trunk width if needed
        if trunk_in != hidden:
            self.proj = nn.Linear(trunk_in, hidden)
            init_linear(self.proj)
        else:
            self.proj = nn.Identity()
        for _ in range(layers):
            blocks.append(GatedResBlock(hidden, mult=2, dropout=dropout))
        self.trunk = nn.Sequential(*blocks)
        self.trunk_norm = nn.LayerNorm(hidden)

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
        self.apply(init_linear)

        # Optional bounds for clamping (set via set_action_bounds)
        self.register_buffer("a_low", None, persistent=False)
        self.register_buffer("a_high", None, persistent=False)

    def set_action_bounds(self, low: torch.Tensor, high: torch.Tensor):
        """low/high are 1D tensors of shape (output_dim,) in radians."""
        self.a_low = low.clone().detach().to(next(self.parameters()).device)
        self.a_high = high.clone().detach().to(next(self.parameters()).device)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_slices is None:
            h = self.encoders["x"](x)
        else:
            hs = []
            for name, sl in self.group_slices.items():
                hs.append(self.encoders[name](x[..., sl]))
            h = torch.cat(hs, dim=-1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) with D=input_dim (24 by default).
        returns: (B, 6) joint angles (absolute) or delta (applied to q) depending on predict_delta flag.
        """
        h = self._encode(x)
        h = self.trunk_norm(self.trunk(h))
        out = self.head(h)  # (B,6)


        y = out

        # Optional joint-limit clamp if bounds were set
        if self.a_low is not None and self.a_high is not None:
            y = torch.max(torch.min(y, self.a_high), self.a_low)

        return y


# keep your config (updated default input_dim=24)
@dataclass
class TrainConfig:
    input_dim: int = 24
    output_dim: int = 6
    hidden: int = 512
    layers: int = 5
    dropout: float = 0.05
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 60
    grad_clip: float = 1.0
    amp: bool = False
    num_workers: int = 4
    seed: int = 0
    val_ratio: float = 0.2
    predict_delta: bool = False
    max_delta: float = 0.15
    noise_std: float = 0.01
    control_mode: str = "position_delta"
    

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "TrainConfig":
        with open(path, 'r') as f:
            return TrainConfig(**json.load(f))
