"""CSRNet crowd density regression network (dilated VGG16 backbone).

Ported from the original CSRNet implementation (leeyeehoo/CSRNet-pytorch,
MIT licensed) as wrapped by the LWCC project (tersekmatija/lwcc, MIT
licensed). Provided as an alternative backend to DM-Count -- CSRNet is
slightly less accurate on the published benchmarks but is a useful
cross-check / fallback model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FRONTEND_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
_BACKEND_CFG = [512, 512, 512, 256, 128, 64]


def _make_layers(cfg, in_channels=3, dilation=False) -> nn.Sequential:
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate),
                nn.ReLU(inplace=True),
            ]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    name = "CSRNet"

    def __init__(self):
        super().__init__()
        self.frontend = _make_layers(_FRONTEND_CFG)
        self.backend = _make_layers(_BACKEND_CFG, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)
