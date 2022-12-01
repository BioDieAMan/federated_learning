#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import LEGO_iid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

        """
        This part is the LEGO dataset code!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

    elif args.dataset == 'LEGO':
        data_dir = '../data/LEGO brick images v1/'
        classes = os.listdir(data_dir)
        print('The number of classes in the dataset is: ' + str(len(classes)))

        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            #transforms.Resize(100),  # resize shortest side to 100 pixels
            #transforms.CenterCrop(100),  # crop longest side to 100 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(data_dir, transform=train_transform)
        img, label = dataset[100]
        torch.manual_seed(20)
        test_size = len(dataset) // 5
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = LEGO_iid(train_dataset, args.num_users)




    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        meanValue = torch.mean(w_avg[key])
        stdValue = torch.std(w_avg[key],unbiased=False)
        newW = (w_avg[key]-meanValue)/stdValue
        w_avg[key]=newW
        for i in range(1, len(w)):
            meanValue = torch.mean(w[i][key])
            stdValue = torch.std(w[i][key],unbiased=False)
            newW = (w[i][key]-meanValue)/stdValue
            w_avg[key] += newW
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
