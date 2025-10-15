import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 200)
        self.layer4 = nn.Linear(200, 100)
        self.layer5 = nn.Linear(100, 1)

    # x_in is a tensor of shape (X, 7)
    def forward(self,x_in):
        x = self.layer1(x_in)
        x = F.relu(x)
        x = self.layer2(x) # must be Tensor, not ReLU
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)

        return x