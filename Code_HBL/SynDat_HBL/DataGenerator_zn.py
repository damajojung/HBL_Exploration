
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pathlib

# the code below simple hides some warnings we don't want to see
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

sns.set_style('darkgrid')


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Dynthetic Data")
    parser.add_argument('-c', dest="n_c", default=10, type=int)
    parser.add_argument('-d', dest="dims", default=3, type=int)
    parser.add_argument('-sd', dest="sd_samp", default=0.1, type=float)
    parser.add_argument('-ss', dest="samp_size", default=100, type=int)
    parser.add_argument('-t', dest="test_prop", default=0.2, type=float)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-csv', dest="to_csv", default=True, type=bool)
    args = parser.parse_args()
    return args



def data_generator(n_c, dims, sd_samp, samp_size, test_prop, to_csv):
    l = np.arange(0,n_c,1)
    centers = pd.DataFrame()

    # Creating the Centers
    for i in l:
        c = np.random.uniform(low=-1.0, high=1.0, size=dims)
        centers[str(i)] = c

    ######## Training Data

    print('Start of Creation of Trainin Data')
    index = []
    samps1 = []
    s_final = []

    for i in np.arange(0,centers.shape[1],1):
        if i % 100 == 0:
            print(i)
        for j in np.arange(0,centers.shape[0],1):
            s = np.random.normal(loc=centers[str(i)][j], scale= sd_samp, size= samp_size)
            samps1.append(s)
            if len(samps1) == dims:
                samps1 = np.array(samps1)
                s_m = np.mean(samps1, axis = 0)
                s_sd = np.std(samps1, axis = 0)
                zn = np.divide((samps1 - s_m * np.ones(samp_size)),s_sd)
                s_final.append(zn)
                samps1 = []

    # Normalise Data    
    train = np.array(s_final)
    m_val = np.abs(train).max() 
    train = np.divide(train, m_val)
        
    print('We are going to the data frame creation.')
    train = pd.DataFrame(np.array(train).reshape(centers.shape[1] * args.dims,samp_size))
    print('Data frame creation successfully finished.')

    for i in centers.columns:
        t = np.repeat(str(i),dims)
        for j in t:
            index.append(j)
    train.index = index
    # train = train.sample(frac=1)

    current_path = os.getcwd()
    dat_path = (current_path + '/data/')

    if to_csv:
        dir = 'syndat_zn'
        final_dat_path = (dat_path + dir)
        if not os.path.exists(final_dat_path):
            os.mkdir(final_dat_path)
        train.to_csv(os.path.join(final_dat_path, "%dc%dd%ds_train.csv" % (args.n_c, args.dims, args.samp_size)))

    print('End of Creation of Trainin Data')

    ######## Testing Data

    samp_size = int(samp_size * test_prop)
    print('This is the Test samp_size:', samp_size)

    index = []
    samps1 = []
    s_final = []

    for i in np.arange(0,centers.shape[1],1):
        if i % 100 == 0:
            print(i)
        for j in np.arange(0,centers.shape[0],1):
            s = np.random.normal(loc=centers[str(i)][j], scale= sd_samp, size= samp_size)
            samps1.append(s)
            if len(samps1) == dims:
                samps1 = np.array(samps1)
                s_m = np.mean(samps1, axis = 0)
                s_sd = np.std(samps1, axis = 0)
                zn = np.divide((samps1 - s_m * np.ones(samp_size)),s_sd)
                s_final.append(zn)
                samps1 = []

    # Normalise Data    
    test = np.array(s_final)
    m_val = np.abs(test).max() 
    test = np.divide(test, m_val)
        
    print('We are going to the data frame creation.')
    test = pd.DataFrame(np.array(test).reshape(centers.shape[1] * args.dims,samp_size))
    print('Data frame creation successfully finished.')
    

    for i in centers.columns:
        t = np.repeat(str(i),dims)
        for j in t:
            index.append(j)
    test.index = index
    # test = test.sample(frac=1)

    # if to_csv:
    #     path = os.getcwd() # Get the current path where this file is stored
    #     path = path + '/' + str(n_c) + 'c' + str(dims) + 'd' + str(samp_size) + 's' + '_test' + '.csv'
    #     test.to_csv(path_or_buf=path, index=True)

    current_path = os.getcwd()
    dat_path = (current_path + '/data/')

    if to_csv:
        dir = 'syndat_zn'
        final_dat_path = (dat_path + dir)
        if not os.path.exists(final_dat_path):
            os.mkdir(final_dat_path)
        test.to_csv(os.path.join(final_dat_path, "%dc%dd%ds_test.csv" % (args.n_c, args.dims, args.samp_size)))

    print('End of Creation of Testing Data')
    
    return train, test



if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()

    # # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dat_train, dat_test = data_generator(n_c = args.n_c,
     dims = args.dims,
      sd_samp = args.sd_samp,
       samp_size = args.samp_size,
        test_prop = args.test_prop,
        to_csv = args.to_csv)


# python DataGenerator_zn.py -c 1000 -d 300 -sd 1.25 -ss 1000 -t 0.2 -s 42 -csv True
