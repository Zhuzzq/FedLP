#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset
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

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(input_size=784, hidden_size=500, num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net_glob.parameters(), lr=0.001)    

    list_loss = []
    acc_test = []
    
    for round in range(args.round):
        net_glob.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(round+1, args.round, batch_idx * len(data), len(train_loader.dataset), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

        # test
        test_acc, test_loss = test(net_glob, test_loader)
        acc_test.append(test_acc)

    # plot loss or acc
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.title('Test Accuracy vs. Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.round))
    
