"This module contains the CNN model"


import torch
import torch.nn as nn
import torch.nn.functional as F

class GrayscaleCNN(nn.Module):
    def __init__(self):
        super(GrayscaleCNN, self).__init__()
        
        # Define the CNN architecture for grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the input to the fully connected layer
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # 12x12 is from (100/2^3) size after pooling
        self.fc2 = nn.Linear(256, 3)  # 3 classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 12 * 12)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
