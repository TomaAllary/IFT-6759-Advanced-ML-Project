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

        # Define constants
        self.max_encoding_lenght = 1536
        self.num_encoded_features = 1536 # always 1536

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

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6144, num_emotions)

        self.activation = nn.ReLU()
        if activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))

        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x