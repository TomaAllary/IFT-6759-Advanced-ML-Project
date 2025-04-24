import torch
from torch import optim
from torch import nn
from tqdm import tqdm



"""
Here inputs will have the form (n+m) X 1536
where n is the lenght of the encoded audio and m the lenght of the encoded text.

Since n & m varies for each conversation, we shall use padding and have a fixed size 
corresponding to the longest length of (n+m). Or something close and chop longer conversations.

"""

import torch
from torch import optim
from torch import nn
from tqdm import tqdm



"""
Here inputs will have the form (n+m) X 1536
where n is the lenght of the encoded audio and m the lenght of the encoded text.

Since n & m varies for each conversation, we shall use padding and have a fixed size 
corresponding to the longest length of (n+m). Or something close and chop longer conversations.
"""

class CNNClassifier(nn.Module):

    def __init__(self, activation_function='relu', num_emotions=7):
        super(CNNClassifier, self).__init__()
        self.is_processing_v1 = False;
        self.fc_size = 6144 if self.is_processing_v1 else 258048

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(8)
        self.pool4 = nn.AvgPool2d(2, 2)

        self.drop2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.fc_size, num_emotions)

        self.activation = nn.ReLU()
        if activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.drop2d(x)
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.drop2d(x)
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.drop2d(x)
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))

        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


######################################### Version 2 #########################################
"""
In the end, it wasn't better. 
But this model did not overfit. Train acc and Val acc were similar (but lower than V1)
"""
class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout_value=0.2, activation_function='leaky_relu'):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.dropout2d = nn.Dropout2d(dropout_value)

        self.activation = nn.ReLU()
        if activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout2d(x)
        return x

class CNNClassifier2(nn.Module):

    def __init__(self, activation_function='relu', dropout_value=0.5, fc_size=32, num_blocks=4, num_emotions=7):
        super(CNNClassifier2, self).__init__()

        self.first_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.second_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.Sequential(
            *[CNNBlock(in_channels=64, out_channels=64, kernel_size=3, dropout_value=dropout_value, activation_function=activation_function) for _ in range(num_blocks)]
        )

        self.final_conv = nn.Conv2d(64, fc_size, kernel_size=3, stride=1, padding=1)
        self.final_bn = nn.BatchNorm2d(fc_size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(fc_size, num_emotions)

        self.activation = nn.ReLU()
        if activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.first_conv(x)
        x = self.second_conv(x)

        x = self.blocks(x)

        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.activation(x)
        x = self.global_pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class CNNClassifier3(nn.Module):

    def __init__(self, activation_function='relu', num_emotions=7):
        super(CNNClassifier, self).__init__()
        self.is_processing_v1 = False;
        self.fc_size = 6144 if self.is_processing_v1 else 258048

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(4, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(4, 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(4, 2)

        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.AvgPool2d(4, 2)

        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(8)
        self.pool5 = nn.AvgPool2d(4, 2)

        self.drop2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6000, num_emotions)

        self.activation = nn.ReLU()
        if activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.drop2d(x)
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.drop2d(x)
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.drop2d(x)
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))
        x = self.drop2d(x)
        x = self.pool5(self.activation(self.bn5(self.conv5(x))))

        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x