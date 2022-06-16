# codebase from hyperspherical prototype networks, Pascal Mettes, NeurIPS2019
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms


import argparse
import numpy as np
from PIL import Image
import pickle
from utils import *
import os
import matplotlib as mpl
mpl.use('Agg')
import random

################################################################################
# General helpers.
################################################################################

#
# Transform a CSV file into a correct Torch Tensor
#
def csv_torch(df):

    # Creating Tuples
    cols = df.columns
    rows = np.unique(df.index)
    samps = []

    for i in rows:
        s  = df.loc[int(i),:]
        for j in cols:
            t = torch.Tensor(np.array(s[str(j)]))
            tup = (t, int(i))
            samps.append(tup)
    return samps

#
# Count the number of learnable parameters in a model.
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
# Get the desired optimizer.
#
def get_optimizer(optimname, params, learning_rate, momentum, decay):
    if optimname == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    elif optimname == "adadelta":
        optimizer = optim.Adadelta(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "rmsprop":
        optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay, momentum=momentum)
    elif optimname == "asgd":
        optimizer = optim.ASGD(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adamax":
        optimizer = optim.Adamax(params, lr=learning_rate, weight_decay=decay)
    else:
        print('Your option for the optimizer is not available, I am loading SGD.')
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)

    return optimizer


################################################################################
# ImageNet helpers
################################################################################

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=(X_train), # lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        mean=mean_image)



def load_data(input_file):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']
    # mean_image = d['mean']

    # x = np.divide(x,np.float32(255)
    # mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]

    # x -= mean_image

    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    return x, y

def get_train_dat(p):

    path = p
    counter = 0

    for i in os.listdir(path):
        counter += 1
        total_path = path + i

        x,y = load_data(total_path)

        if counter == 1:
            x_train, y_train = x, y
        else:
            x_train = np.append(x_train, x, axis = 0)
            y_train = y_train + y
    return x_train, y_train

def create_tuples(x,y):
    samps = []

    for i in np.arange(0,len(x),1):
        t = torch.Tensor(x[i]).view(3,32,32)
        tup = (t, int(y[i]))
        samps.append(tup)
    return samps

def get_test_dat(p):
    x_test , y_test = load_data(p)
    dat_test = create_tuples(x_test, y_test)
    return dat_test

def create_imnet_train_test(path_train, path_test, batch_size, kwargs):
    x_train, y_train = get_train_dat(path_train)
    dat_train = create_tuples(x_train, y_train)
    trainset = torch.utils.data.DataLoader(dat_train, batch_size = batch_size, shuffle = True, **kwargs)

    x_test,y_test = load_data(path_test)
    dat_test = create_tuples(x_test, y_test)
    testset = torch.utils.data.DataLoader(dat_test, batch_size = batch_size, shuffle = True, **kwargs)

    return trainset, testset


################################################################################
# Standard dataset loaders.
################################################################################
def load_dataset(dataset_name, n_c, dims, samp_size, basedir, batch_size, kwargs):
    if dataset_name == 'cifar100':
        return load_cifar100(basedir, batch_size, kwargs)
    elif dataset_name == 'cifar10':
        return load_cifar10(basedir, batch_size, kwargs)
    elif dataset_name == 'cub':
        return load_cub(basedir, batch_size, kwargs)
    elif dataset_name == 'syndat':
        return load_syndat(basedir, n_c, dims, samp_size, batch_size, kwargs)  
    elif dataset_name == 'pthimnet':
        return load_pth_imnet(basedir)  
    elif dataset_name == 'imnet':
        return load_imnet(basedir, batch_size, kwargs)  
    else:
        raise Exception('Selected dataset is not available.')


def load_cifar100(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=True,
                          transform=transforms.Compose([
                              transforms.RandomCrop(32, 4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize,
                          ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                          ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def load_cifar10(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, 4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                         ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def load_cub(basedir, batch_size, kwargs):
    # Correct basedir.
    basedir += "cub/"

    # Normalization.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Train loader.
    train_data = datasets.ImageFolder(basedir + "train/", transform=transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Test loader.
    test_data = datasets.ImageFolder(basedir + "test/", transform=transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize]))

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return trainloader, testloader

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def load_syndat(basedir, n_c, dims, samp_size, batch_size, kwargs):
    # Correct basedir.
    basedir += "syndat/"

    path_train = basedir +  ("/%dc%dd%ds_train.csv" % (n_c, dims, samp_size))
    path_test = basedir +  ("/%dc%dd%ds_test.csv" % (n_c, dims, samp_size))
    train = pd.read_csv(path_train, index_col=0)
    test = pd.read_csv(path_test, index_col=0)

    dat_train = csv_torch(train)
    dat_test = csv_torch(test)

    trainloader = torch.utils.data.DataLoader(dat_train, batch_size = batch_size, shuffle = True, **kwargs)
    testloader = torch.utils.data.DataLoader(dat_test, batch_size = batch_size, shuffle = True, **kwargs)

    return trainloader, testloader


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# Pre created one already in torch format
def load_pth_imnet(basedir):
    # Correct basedir.
    basedir += "torch_imnet/"

    train_path = basedir + 'imnet_train.pth'
    test_path = basedir + 'imnet_test.pth'

    trainloader = torch.load(train_path)
    testloader = torch.load(test_path)

    return trainloader, testloader

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# create a new one
def load_imnet(basedir,  batch_size, kwargs):
    # Correct basedir.
    basedir += "imnet/"

    train_path = basedir + 'Imagenet32_train/'
    test_path = basedir + 'val_data'

    trainloader, testloader  = create_imnet_train_test(path_train= train_path, path_test= test_path, batch_size = batch_size, kwargs = kwargs)

    return trainloader, testloader
