"""DM-Count crowd density regression network (VGG19 backbone).

Architecture ported from the original DM-Count implementation
(cvlab-stonybrook/DM-Count, MIT licensed) as wrapped by the LWCC project
(tersekmatija/lwcc, MIT licensed). DM-Count predicts a per-pixel density
map whose sum over the image is the estimated crowd count; unlike a person
detector it does not need to resolve individual, possibly heavily occluded
people, which is why density-regression models are the standard choice for
very dense crowds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_CFG_E = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]


def _make_vgg_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DMCount(nn.Module):
    """VGG19-based density regressor used by DM-Count."""

    name = "DM-Count"

    def __init__(self):
        super().__init__()
        self.features = _make_vgg_layers(_CFG_E)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.reg_layer(x)
        return self.density_layer(x)
