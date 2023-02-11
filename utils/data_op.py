#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
import pandas as pd


def dataset_stat(dataset, cl=10):
    d_labels = [int(y) for x, y in dataset]
    st = pd.value_counts(d_labels)
    cls_idx = sorted(st.keys())
    # print(cls_idx)
    count = [0] * cl
    for i in cls_idx:
        count[i] = st[i]
    return cls_idx, count


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.labels = [self.dataset[int(i)][1] for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def iid_data(dataset, num_users, sample_num=0):
    """
    random sampling to generate i.i.d. data
    return: dict of image index
    """
    if sample_num == 0:
        num_items = int(len(dataset) / num_users)
    else:
        num_items = sample_num
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    num_imgs = 500  # mnist 300?
    num_shards = int(len(dataset) / num_imgs)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.targets.numpy()
    labels = dataset.targets[0:num_shards * num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign  shards/client
    # num_items = int(len(idx_shard) / num_users)
    num_items = 2
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_items, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mixed_noniid(dataset, num_users, ratio):
    """
    Sample mixed non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :non-i.i.d ratio
    :return:
    """
    # iid sample
    # noniid_num = int(len(dataset) * ratio)
    # print(noniid_num)

    # num_items = int((len(dataset) - noniid_num) / num_users)
    # dict_users, all_idxs = {}, [i for i in range(noniid_num, len(dataset))]
    iid_num = int(len(dataset) * (1 - ratio))
    print(iid_num)

    iidnum_items = int(iid_num / num_users)
    dict_users, all_idxs = {}, [i for i in range(iid_num)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, iidnum_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = np.array(list(dict_users[i]))

    # non-iid sample
    # num_items = int(num_shards / num_users)
    num_items = 2
    num_imgs = 500 - iidnum_items // num_items  # mnist 300? cifar 500
    # num_shards = int(noniid_num / num_imgs)
    num_shards = int(len(dataset) / num_imgs)
    idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets[0:num_shards * num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_items, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        # if i == 7:
        #     rand_set = list(rand_set)
        #     rand_set[-1] = idx_shard[-1]

        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


# def cifar_mixed_noniid(dataset, num_users, ratio):
#     """
#     Sample mixed non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :non-i.i.d ratio
#     :return:
#     """
#     # iid sample
#     noniid_num = int(len(dataset) * ratio)
#     print(noniid_num)

#     num_items = int((len(dataset) - noniid_num) / num_users)
#     dict_users, all_idxs = {}, [i for i in range(noniid_num, len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#         dict_users[i] = np.array(list(dict_users[i]))

#     # non-iid sample
#     num_imgs = 500  # mnist 600?
#     num_shards = int(noniid_num / num_imgs)
#     idx_shard = [i for i in range(num_shards)]
#     # dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.targets[0:noniid_num]

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign  shards/client
#     # num_items = int(len(idx_shard) / num_users)
#     num_items = 2
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, num_items, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return dict_users


# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 100, 500
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(dataset.targets)

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return dict_users


def get_dataset(name, iid, num_usr, ratio=0.5):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if name == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if iid == 1:
            # Sample IID user data
            user_groups = iid_data(train_dataset, num_usr, 1000)
        elif iid == 2:
            user_groups = mixed_noniid(train_dataset, num_usr, ratio)
        else:
            user_groups = noniid(train_dataset, num_usr)

    elif name == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        if name == 'mnist':
            train_dataset = datasets.MNIST(root="data", train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(root="data", train=False, download=True,
                                          transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(root="data", train=True, download=True,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(root="data", train=False, download=True,
                                                 transform=apply_transform)

        # sample training data amongst users
        if iid == 1:
            # Sample IID user data from Mnist
            user_groups = iid_data(train_dataset, num_usr, 1000)
            # user_groups = iid_data(train_dataset, num_usr, int((1 - ratio) * len(train_dataset) / num_usr) + 400 * 10)
        elif iid == 2:
            user_groups = mixed_noniid(train_dataset, num_usr, ratio)
        else:
            user_groups = noniid(train_dataset, num_usr)

    return train_dataset, test_dataset, user_groups
