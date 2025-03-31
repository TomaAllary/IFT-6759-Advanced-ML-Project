import torch
import torch.nn as nn
import os

"""
Here inputs will have the form (n+m) X 1536
where n is the lenght of the encoded audio and m the lenght of the encoded text.

Since n & m varies for each conversation, we shall use padding and have a fixed size 
corresponding to the longest length of (n+m). Or something close and chop longer conversations.
"""

"""
ResNet CNN architecture here
Explanation: https://medium.com/@siddheshb008/resnet-architecture-explained-47309ea9283d
Paper: Deep Residual Learning for Image Recognition -> arXiv:1512.03385
"""

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, first_conv_stride=1):

        super(ResNetBlock, self).__init__()

        self.relu = nn.ReLU()

        # FIRST HALF BLOCK #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_conv_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # make sure dimension are the same for the skip connections:
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=first_conv_stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # SECOND HALF BLOCK #
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # make sure dimension are the same for the skip connections:
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # FIRST HALF BLOCK #
        x_shortcut1 = x

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.shortcut1(x_shortcut1)
        x = self.relu(x)

        # SECOND HALF BLOCK #
        x_shortcut2 = x

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = x + self.shortcut2(x_shortcut2)
        x = self.relu(x)

        return x


class ResNetCNN(nn.Module):
    def __init__(self, num_emotion=6):
        super(ResNetCNN, self).__init__()

        # CONV formula
        # out = ( (in + 2*pad - kernel_size) / stride ) + 1
        # *we don't use padding here*

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()


        self.layer1 = ResNetBlock(64, 64, first_conv_stride=1)  # (32 - 1)/1 + 1 = out = 32
        self.layer2 = ResNetBlock(64, 128, first_conv_stride=2)  # (32 - 1)/2 + 1 = out = 16
        self.layer3 = ResNetBlock(128, 256, first_conv_stride=2)  # (16 - 1)/2 + 1 = out = 8
        self.layer4 = ResNetBlock(256, 512, first_conv_stride=2)  # (8 - 1)/2 + 1 = out = 4

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # pool each 4x4 img -> out = 10, 512
        self.linear1 = nn.Linear(512, 1024)
        self.ff_dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1024, 512)
        self.ff_dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(512, num_emotion)

    def forward(self, images):

        # First conv + Blocks
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Shape for FC layers
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        # FC Layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.ff_dropout1(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.ff_dropout2(x)
        x = self.linear3(x)

        # return logits
        return x

