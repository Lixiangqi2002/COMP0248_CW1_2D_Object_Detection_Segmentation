from typing import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepLabHeadV3Plus(nn.Module):
    """
    DeepLabV3+ Head
    """

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # self.backbone = backbone
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def forward(self, feature : OrderedDict, target=None):
        # print("target:", target)
        
        processed_targets = []
        if target is not None:
            for t in target:
                masks = t['masks'].long()
                # unique_values = np.unique(masks)
                # print("Unique values in target:", unique_values)
                target_tensor = torch.argmax(masks, dim=0)
                processed_targets.append(target_tensor)
            target = torch.stack(processed_targets, dim=0)
            # print("target shape:", processed_targets.shape) # [batch_size, H, W]
            # unique_values = np.unique(target)
            # print("Unique values in target:", unique_values)

        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        if target is None:
            # print("no target")
            return self.classifier(torch.cat([low_level_feature, output_feature], dim=1)), 0.0
        else:
            # print("target")
            class_weights = torch.tensor([0.07, 2.15, 5.42, 4.77, 304.47, 7.26], dtype=torch.float32, device=target.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            seg_out = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
            seg_out = seg_out.to(target.device)
            # print("seg_out shape:", seg_out.shape)
    
            # print("target shape:", target.shape)

            target = F.interpolate(target.unsqueeze(1).float(), size=(256, 256), mode="nearest").squeeze(1).long()

            # print("target shape:", target.shape)
            seg_loss = criterion(seg_out, target)
            # print("seg_loss:", seg_loss)
            # print(seg_out.shape)
            return seg_out, seg_loss
  

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    """
    Standard DeepLab Head
    """

    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Depthwise Separable Convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    """ ASPP Component: Standard 3x3 Convolution + BatchNorm + ReLU """

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )


class ASPPPooling(nn.Sequential):
    """ ASPP Component: Global Pooling """

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling)
    """

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        ]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    """
    Recursively convert standard convolutions to Atrous Separable Convolutions
    """

    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(
            module.in_channels, module.out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.bias)

    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))

    return new_module