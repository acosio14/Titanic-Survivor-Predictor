import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 200)
        self.layer4 = nn.Linear(200, 100)
        self.layer5 = nn.Linear(100, 1)

    def forward(self,x_in):
        x = self.layer1(x_in)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        feature = self.data[index]
        target = self.targets[index]

        return feature, target

    def __len__(self):
        return len(self.data)


def convert_df_to_tensor(data_df):
    """ Convert dataframe to tensor."""
    data_np = data_df.to_numpy().astype(np.float32)
    
    return torch.from_numpy(data_np)


def train_pytorch_model(dataloader, num_epochs):
    mps_device = torch.device("mps")

    pytorch_model = NeuralNetwork().to(mps_device)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        pytorch_model.train()
        total_loss = 0
        for features, target in dataloader:
            # Move data to device.
            features_mps = features.to(mps_device)
            target_mps = target.to(mps_device)
            
            # Forward pass.
            outputs = pytorch_model(features_mps)
            loss = criterion(outputs,target_mps)

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
