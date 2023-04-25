from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

norm_types: Tuple[nn.Module, ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)


def init_default(m: nn.Module, func: Callable = nn.init.kaiming_normal_) -> None:
    """Initialize `m` weights with `func` and set bias to 0."""
    if hasattr(m, "weight"):
        func(m.weight)
    if hasattr(m, "bias") and hasattr(m.bias, "data"):
        m.bias.data.fill_(0)


def requires_grad(m: nn.Module) -> bool:
    """Check if the first parameter of `m` requires grad or not"""
    ps = list(m.parameters())
    return ps[0].requires_grad if ps else False


def cond_init(m: nn.Module, func: Callable) -> None:
    """Initialize the non-batchnorm layers of `m` with `init_default`"""
    if not requires_grad(m):
        return None

    if not isinstance(m, norm_types):
        init_default(m, func)
    elif isinstance(m, norm_types):
        if m.affine:
            m.bias.data.fill_(1e-3)
            m.weight.data.fill_(1.0)


def apply_init(model: nn.Module, func: Callable = nn.init.kaiming_normal_) -> None:
    """Initialize all non-batchnorm layers of `model` with `func`"""
    childeren = model.children()
    if isinstance(model, nn.Module):
        cond_init(model, func=func)

    for child in childeren:
        apply_init(child, func)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(
        self, size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    ) -> None:
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mp(x), self.ap(x)], dim=1)
