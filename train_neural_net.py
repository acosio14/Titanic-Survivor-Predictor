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


def evaluate_pytorch_model(model, loss_fcn, test_dataloader):
    total_loss = 0
    model.eval()
    
    with torch.no_grad():
        for val_inputs, val_targets in test_dataloader:
            
            val_inputs_mps = val_inputs.to(torch.device("mps"))
            val_targets_mps = val_targets.to(torch.device("mps"))

            y_pred_val= model(val_inputs_mps)
            loss = loss_fcn(y_pred_val, val_targets_mps)
            total_loss += loss
    
    return total_loss / len(test_dataloader)


def train_pytorch_model(train_dataloader, test_dataloader, num_epochs):
    mps_device = torch.device("mps")
    best_avg_loss = 1000000
    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    pytorch_model = NeuralNetwork().to(mps_device)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    loss_fcn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        pytorch_model.train()
        total_loss = 0
        for features, target in train_dataloader:
            # Move data to device.
            features_mps = features.to(mps_device)
            target_mps = target.to(mps_device)
            
            # Forward pass.
            y_pred = pytorch_model(features_mps)
            loss = loss_fcn(y_pred,target_mps)

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_val_loss = evaluate_pytorch_model(pytorch_model, loss_fcn, test_dataloader)
        
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {total_loss / len(train_dataloader)}")
        print(f"Val Loss: {avg_val_loss}")
        print()
        
        train_loss_list.append(total_loss / len(train_dataloader))
        val_loss_list.append(avg_val_loss.cpu().numpy())
        epoch_list.append(epoch)

        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            best_epoch = epoch
            best_model = pytorch_model

    return best_model, best_epoch, best_avg_loss, train_loss_list, val_loss_list, epoch_list


def save_model(model, epoch, timestamp):
    
    torch.save(model.state_dict(), f"trained_models/model_epoch{epoch}_{timestamp}.pth")

