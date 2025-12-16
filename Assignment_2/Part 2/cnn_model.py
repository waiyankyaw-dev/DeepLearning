from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()

        # Feature extractor: reduced VGG from the slides
        self.features = nn.Sequential(
            # 1. conv: in=3, out=64, k=3x3, s=1, p=1
            ConvBlock(n_channels, 64),

            # 2. maxpool: k=3x3, s=2, p=1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 3. conv: in=64, out=128
            ConvBlock(64, 128),

            # 4. maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 5. conv: in=128, out=256
            ConvBlock(128, 256),

            # 6. conv: in=256, out=256
            ConvBlock(256, 256),

            # 7. maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 8. conv: in=256, out=512
            ConvBlock(256, 512),

            # 9. conv: in=512, out=512
            ConvBlock(512, 512),

            # 10. maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 11. conv: in=512, out=512
            ConvBlock(512, 512),

            # 12. conv: in=512, out=512
            ConvBlock(512, 512),

            # 13. maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #for CIFAR-10 (32x32 input), after these pools the spatial size is 1x1,
        # so the feature map is [batch, 512, 1, 1] -> 512 features.
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.features(x)           # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)      # flatten -> [B, 512]
        out = self.classifier(x)       # logits -> [B, 10]
        return out
