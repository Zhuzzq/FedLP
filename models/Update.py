#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy


class DatasetSplit(Dataset):            # Define __getitem__() method, so that we can fetch specific data in the dataset.
    def __init__(self, dataset, idxs):  # Here 'idxs' are ndarray.
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):                  # Define the length of the class, control when the iterate in line 41 stops.
        return len(self.idxs)

    def __getitem__(self, item):        # idxs contain indexs of data in label order in dataset, length of 600 per list.
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batchsize, shuffle=True)

    def train(self, net):
        net.train()

        # train and update
        
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        epoch_loss = []
        for iter in range(self.args.epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):   # Iterate in the order of __getitem__(item)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # main_fed.py
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        # main_fed_cplx.py
        # return net, sum(epoch_loss) / len(epoch_loss)

def download_sampling(w_local, w_glob, prob):
    w_temp = copy.deepcopy(w_local)
    for layer in w_glob.keys():
        if random.random() < prob:
            w_temp[layer] = w_glob[layer]
    return w_temp
