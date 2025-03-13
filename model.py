# model.py (Small ResNet-like NN for TicTacToe)
import torch
import torch.nn as nn

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.res_block = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.policy_head = nn.Linear(64, 9)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x + self.res_block(x)  # Residual connection
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value