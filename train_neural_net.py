import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(l1_input_size, l1_output_size  )
        self.relu = nn.ReLu()
        self.layer2 = nn.Linear(l1_output_size, l2_output_size)

    def forward(self,x_in):
        x = self.layer1(x_in)
        x = self.relu(x)
        x = self.layer2(x)
        return x