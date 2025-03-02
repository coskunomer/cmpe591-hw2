import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, N_ACTIONS):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(6, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, N_ACTIONS) 
        
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = self.dropout1(x) 
        x = torch.relu(self.fc2(x)) 
        x = self.dropout2(x)  
        x = self.fc3(x) 
        return x
