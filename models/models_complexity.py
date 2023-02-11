import sys
sys.path.append('.')

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from utils.options import args_parser
import copy
from torchinfo import summary
from thop import profile

class EnhancedCNNCifar(nn.Module):
    def __init__(self, args):
        super(EnhancedCNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.global_fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        # x = F.relu(self.global_fc1(x))
        x = self.global_fc2(x)

        # return F.softmax(x, dim=1)
        return x


class CNNCifarComplexity(nn.Module):
    def __init__(self, args, complexity):
        super(CNNCifarComplexity, self).__init__()

        self.cplx = complexity

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2)

        if self.cplx == 1:
            self.fc1 = nn.Linear(32 * 16 * 16, 128) 
            self.fc = nn.Linear(128, args.num_classes)
        
        elif self.cplx == 2:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.fc2 = nn.Linear(64 * 8 * 8, 128)
            self.fc = nn.Linear(128, args.num_classes)

        elif self.cplx == 3:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.fc2 = nn.Linear(64 * 8 * 8, 128)
            self.fc = nn.Linear(128, args.num_classes)

        elif self.cplx == 4:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128 )
            self.fc3 = nn.Linear(128 * 4 * 4, 128)
            self.fc = nn.Linear(128, args.num_classes)
        
        elif self.cplx == 5:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)  
            self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
            self.global_fc2 = nn.Linear(128, args.num_classes)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)

        if self.cplx == 1:
            x = x.view(-1, 32 * 16 * 16)
            x = self.fc1(x)
            x = self.fc(x)
            return x

        x = self.bn3(F.relu(self.conv3(x)))

        if self.cplx == 2:
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)

        if self.cplx == 3:
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn5(F.relu(self.conv5(x)))

        if self.cplx == 4:
            x = self.pool(x)
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc3(x)
            x = self.fc(x)
            return x
        
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(x)

        if self.cplx == 5:
            x = x.view(-1, 128 * 4 * 4)
            x = self.global_fc1(x)
            x = self.global_fc2(x)
            return x


def init_cplx_dict(args, flag):  # flag is the preference for complexity
    idx_cplx_dict={}                     # {idxs_user: complexity  -->  0: 5, 1: 1, 2: 4, ...}
    prob = [[0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6]]
    for i in range(args.num_users):
        idx_cplx_dict[i] = int( np.random.choice([1,2,3,4,5], 1, p = prob[flag-1]) )
    return idx_cplx_dict

def download_hetero_net(global_model, local_model, cplx):   # download
    local_update = copy.deepcopy(local_model.state_dict())
    if cplx == 5:
        for name in local_update.keys():
            local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    else:
        for name in local_update.keys():	
            if 'fc' in name:
                continue
            else:
                local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    return local_model

# def generate_cplx_net(args, global_model, user_idx, idx_cplx_dict, device='cuda'):   # download
#     cplx = idx_cplx_dict[user_idx]
#     local_model = CNNCifarComplexity(args, cplx).to(device)
#     local_update = copy.deepcopy(local_model.state_dict())
#     if cplx == 5:
#         for name in local_update.keys():
#             local_update[name] = global_model.state_dict()[name]
#         local_model.load_state_dict(local_update)
#     else:
#         for name in local_update.keys():	
#             if 'fc' in name:
#                 continue
#             else:
#                 local_update[name] = global_model.state_dict()[name]
#         local_model.load_state_dict(local_update)
#     return local_model

if __name__=="__main__":
    args = args_parser()
    input = torch.randn(1, 3, 32, 32) 
    for i in [1,2,3,4,5]:
        print('--------------------------------------------------------------------------------')
        model = CNNCifarComplexity(args,i)
        summary(model, (1, 3, 32, 32))
        macs, params = profile(model, inputs=(input, ))
