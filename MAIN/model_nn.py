"Module for storing the model"
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Base model with 5 layers"""
    def __init__(self, in_features=10000, h1=2000, h2=500,
                 h3=200, h4=50, h5=10,
                 out_features=3, dropout_p=0.5):
        """Initialize an instance of the model"""
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_features)
        self.dropout = nn.Dropout(p=dropout_p)  # Dropout layer

    def forward(self, x):
        """Forward method for the layers in the model"""
        # Using softmax in the output layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
