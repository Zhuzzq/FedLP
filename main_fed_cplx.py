#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import time

from utils.sampling import get_iid, get_noniid, get_mixed_noniid, get_dirichlet_noniid, dataset_stats
from utils.options import args_parser, details
from models.Update import LocalUpdate, download_sampling
from models.Nets import MLP, CNNMnist, CNNCifar, ResidualBlock, ResNetCifar
from models.Fed import FedAvg, fed_hetero, fed_dropout
from models.test import test_img
from models.models_complexity import EnhancedCNNCifar, CNNCifarComplexity, init_cplx_dict


# if __name__ == '__main__':

def mainfunc(droprate, iid):

    start = time.time()

    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = iid
    args.up_droprate = droprate
    # args.up_droprate = 0.5
    details(args)

    if torch.cuda.is_available():
        print("Using GPU")
        print("Current GPU id:", torch.cuda.current_device())
        print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Cannot find GPU, return to CPU")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('./data/fmnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=trans_mnist)           
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized dataset')
    
    if args.iid == 1:
        dict_users = get_iid(dataset_train, args.num_users)
    elif args.iid == 0:
        dict_users = get_noniid(dataset_train, args.num_users)
    elif args.iid == 2:
        dict_users = get_mixed_noniid(dataset_train, args.num_users, args.mixrate)
    elif args.iid == 3:
        dict_users = get_dirichlet_noniid(dataset_train, args.num_users, alpha=args.alpha, sample_num=1000 * args.num_users)
    else:
        exit('Error: unrecognized distribution')
    img_size = dataset_train[0][0].shape

    # visualize data distribution
    # dataset_stats(dict_users, dataset_train, args)

    
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        # net_glob = CNNCifar(args=args).to(args.device)
        net_glob = EnhancedCNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = ResNetCifar(ResidualBlock, [2, 2, 2]).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(input_size=784, hidden_size=500, num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    loss_test = []
    acc_test = []

    # fed cplx
    # idx_cplx_dict = init_cplx_dict(args)
    idx_cplx_dict={}
    for i in range(args.num_users):
        idx_cplx_dict[i]=5

    # fed dropout
    p_list = np.random.choice([args.up_droprate], args.num_users, replace=True)
    print(p_list)
   
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.round):
        net_glob.train()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    
            w_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # model_c = generate_cplx_net(args, net_glob, idx, idx_cplx_dict)
            # w_local, loss = local.train(net=copy.deepcopy(model_c).to(args.device))
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w_local)
            else:
                w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # w_glob = FedAvg(w_locals)
        # w_glob = fed_hetero(copy.deepcopy(w_locals), copy.deepcopy(net_glob))
        w_glob = fed_dropout(copy.deepcopy(w_locals), copy.deepcopy(net_glob), idxs_users, p_list)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)   

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg))
        loss_train.append(loss_avg)

        #print accuracy
        acc, loss = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc)) 
        acc_test.append(acc)
        loss_test.append(loss)

    duration = time.time() - start

    print('\nTotal Run Time: {0:0.4f}'.format(duration))

    # save the data
    np_loss_train = np.array(loss_train)
    np_loss_test = np.array(loss_test)
    np_acc_test = np.array(acc_test)
    np.save('./save/fed_homo_DR[{}]_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_losstest[{}].npy'
                .format(args.up_droprate, args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, loss), np_loss_test)
    np.save('./save/fed_homo_DR[{}]_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].npy'
                .format(args.up_droprate, args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc), np_acc_test)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test_loss')
    plt.savefig('./save/fed_homo_DR[{}]_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_losstest[{}].png'
                .format(args.up_droprate, args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, loss))
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.title('Test Accuracy vs. Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_homo_DR[{}]_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].png'
                .format(args.up_droprate, args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc))


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
