from __future__ import annotations

import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    act = str(name).strip().lower()
    if act in {"relu"}:
        return nn.ReLU()
    if act in {"gelu"}:
        return nn.GELU()
    if act in {"silu", "swish"}:
        return nn.SiLU()
    if act in {"mish"}:
        return nn.Mish()
    if act in {"tanh"}:
        return nn.Tanh()
    if act in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if act in {"elu"}:
        return nn.ELU(alpha=1.0)
    if act in {"selu"}:
        return nn.SELU()
    if act in {"softplus"}:
        return nn.Softplus()
    if act in {"identity", "none", "linear"}:
        return nn.Identity()
    raise ValueError(
        "activation must be one of: "
        "relu, gelu, silu, mish, tanh, leaky_relu, elu, selu, softplus, identity"
    )


class MLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = []
        d = int(in_dim)
        for i in range(depth - 1):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(_make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
