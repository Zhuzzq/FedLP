import sys
sys.path.append('.')

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
# from utils.options import args_parser
import copy
from torchinfo import summary
from thop import profile

class EnhancedMLP(nn.Module):
    def __init__(self):
        super(EnhancedMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512) 
        self.fc2 = nn.Linear(512, 128) 
        self.fc3 = nn.Linear(128, 128) 
        self.fc4 = nn.Linear(128, 64) 
        self.fc5 = nn.Linear(64, 32) 
        self.fc6 = nn.Linear(32, 10) 
    
    def forward(self, x):
        out = x.view(-1, 28*28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        return out


class MLPComplexity(nn.Module):
    def __init__(self, complexity):
        super(MLPComplexity, self).__init__()

        self.cplx = complexity

        self.fc1 = nn.Linear(28*28, 512) 

        if self.cplx == 1:
            self.fc = nn.Linear(512, 10)
        
        elif self.cplx == 2:
            self.fc2 = nn.Linear(512, 128) 
            self.fc = nn.Linear(128, 10)

        elif self.cplx == 3:
            self.fc2 = nn.Linear(512, 128) 
            self.fc3 = nn.Linear(128, 128) 
            self.fc = nn.Linear(128, 10)

        elif self.cplx == 4:
            self.fc2 = nn.Linear(512, 128) 
            self.fc3 = nn.Linear(128, 128) 
            self.fc4 = nn.Linear(128, 64) 
            self.fc = nn.Linear(64, 10)
        
        elif self.cplx == 5:
            self.fc1 = nn.Linear(28*28, 512) 
            self.fc2 = nn.Linear(512, 128) 
            self.fc3 = nn.Linear(128, 128) 
            self.fc4 = nn.Linear(128, 64) 
            self.fc5 = nn.Linear(64, 32) 
            self.fc6 = nn.Linear(32, 10) 
    
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))

        if self.cplx == 1:
            x = self.fc(x)
            return x

        x = F.relu(self.fc2(x))

        if self.cplx == 2:
            x = self.fc(x)
            return x

        x = F.relu(self.fc3(x))

        if self.cplx == 3:
            x = self.fc(x)
            return x

        x = F.relu(self.fc4(x))

        if self.cplx == 4:
            x = self.fc(x)
            return x
        
        x = self.fc5(x)

        if self.cplx == 5:
            x = self.fc6(x)
            return x


def download_hetero_net_mlp(global_model, local_model, cplx):   # download
    local_update = copy.deepcopy(local_model.state_dict())
    for name in local_update.keys():	
        if 'fc.' in name:
            continue
        else:
            local_update[name] = global_model.state_dict()[name]
    local_model.load_state_dict(local_update)
    return local_model


if __name__ == '__main__':
    for i in [1,2,3,4,5]:
        model = MLPComplexity(i)
        print(model)