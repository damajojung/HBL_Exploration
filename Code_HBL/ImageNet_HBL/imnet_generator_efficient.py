import argparse
import numpy as np
from PIL import Image
import pickle
from utils import *
import os
import matplotlib as mpl
mpl.use('Agg')
import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# %matplotlib inline

import numpy as np
import argparse
import math
import os
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


# the code below simple hides some warnings we don't want to see
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)



def parse_args():
    parser = argparse.ArgumentParser(description="Creating ImageNet Data")
    parser.add_argument('-train_path', dest="train", default='/home/jungd/HBL/ImageNet_HBL/data/imnet/Imagenet32_train/', type=str)
    parser.add_argument('-test_path', dest="test", default='/home/jungd/HBL/ImageNet_HBL/data/imnet/val_data', type=str)
    parser.add_argument('-bs', dest="batch_size", default=10, type=int)
    parser.add_argument('-is', dest="img_size", default=32, type=int)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    args = parser.parse_args()
    return args



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def create_tuples(x,y, im_size):
    samps = []

    for i in np.arange(0,len(x),1):
        t = torch.Tensor(x[i]).view(3,224,224)
        tup = (t, int(y[i]))
        samps.append(tup)
    return samps
    
def create_tuples_test(x,y, im_size):
    samps = []

    for i in np.arange(0,len(x),1):
        t = torch.Tensor(x[i]).view(3,224,224)
        tup = (t, int(y[i]))
        samps.append(tup)
    return samps


def load_data_train(input_file, im_size):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = np.divide(x,np.float32(255))
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]

    x -= mean_image

    img_size = im_size
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)) #.transpose(0, 3, 1, 2)

    return x, y

def load_data_test(input_file, im_size):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.divide(x,np.float32(255))

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]

    img_size = im_size
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)) #.transpose(0, 3, 1, 2)

    return x, y

def get_train_dat(p, im_size):

    path = p
    counter = 0

    transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224])
            ])

    for i in os.listdir(path):
        counter += 1
        print(counter)
        total_path = path + i

        x,y = load_data_train(total_path, im_size)

        if counter == 1:
            x_train, y_train = x, y
        else:
            x_train = np.append(x_train, x, axis = 0)
            y_train = y_train + y

    print('x_test shape:', x_train.shape)
    # x_train = np.array([transform(Image.fromarray(np.uint8((i)*255))) for i in x_train])
    x_train = np.array([transform(i) for i in x_train])
    return x_train, y_train

def get_test_dat(p, im_size):
    if im_size == 32:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]) 
            ]
        )

    if im_size == 64:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]) 
            ]
        )
    
    x_test, y_test = load_data_test(p, im_size)
    print('x_test shape:', x_test.shape)
    x_test = np.array([transform(i) for i in x_test])
    return x_test, y_test

def create_imnet_train_test(path_train, path_test, batch_size, im_size):
    x_train, y_train = get_train_dat(path_train, im_size)
    dat_train = create_tuples(x_train, y_train, im_size)
    trainset = torch.utils.data.DataLoader(dat_train, batch_size = batch_size, shuffle = False)

    x_test,y_test = get_test_dat(path_test, im_size)
    dat_test = create_tuples_test(x_test, y_test, im_size)
    testset = torch.utils.data.DataLoader(dat_test, batch_size = batch_size, shuffle = False)

    return trainset, testset



if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create both datasets
    trainset, testset = create_imnet_train_test(path_train= args.train,
                                                     path_test=args.test,
                                                     batch_size = args.batch_size,
                                                     im_size = args. img_size)

    # Save datasets 
    current_path = os.getcwd()
    dat_path = (current_path + '/data/')

    dir = f'torch_imnet_{args.img_size}'
    final_dat_path = (dat_path + dir)
    if not os.path.exists(final_dat_path):
        os.mkdir(final_dat_path)
    torch.save(trainset, os.path.join(final_dat_path, "imnet_train.pth"))
    torch.save(testset, os.path.join(final_dat_path, "imnet_test.pth"))

