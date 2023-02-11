#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import random


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def fed_hetero(local_models, global_model):  # upload
    '''
    local_models is a list of state_dict()
    global_model is a model ( class EnhancedCNNCifar )
    '''

    global_update = copy.deepcopy(global_model.state_dict())

    count = {}

    for layer, layer_param in global_update.items():
        count[layer] = 0    
        for model in local_models:
            for name, param in model.items():
                if name == layer:
                    count[layer] += 1
                    break    
    

    for layer, layer_param in global_update.items():
        if count[layer] != 0:
            layer_param.zero_()
            for model in local_models:
                for name, param in model.items():
                    if name == layer:
                        layer_param += param  #deepcopy
                        break
            global_update[layer] = torch.div(layer_param, count[layer])
        
    return global_update


def fed_dropout(local_models, global_model, users, p_list):  # upload
    '''
    local_models is a list of state_dict()
    global_model is a model ( class EnhancedCNNCifar )
    '''

    global_update = copy.deepcopy(global_model.state_dict())

    count = {}
    for name in global_model.state_dict().keys():
        count[name] = 0

    p_dropout = []
    for user in users:
        p_dropout.append(p_list[user])

    for layer, layer_param in global_update.items():
        temp_param = torch.zeros_like(layer_param)
        for idx, model in enumerate(local_models):
            if random.random() < p_dropout[idx]:
                count[layer] += 1
                temp_param += model[layer]
        if count[layer] != 0:
            global_update[layer] = torch.div(temp_param, count[layer])
       
    return global_update