
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time


import pathlib

sns.set_style('darkgrid')

# the code below simple hides some warnings we don't want to see
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import random
import math
import os
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score

def parse_args():
    parser = argparse.ArgumentParser(description="Creating Dynthetic Data")
    parser.add_argument('-c', dest="n_c", default=10, type=int)
    parser.add_argument('-d', dest="dims", default=3, type=int)
    parser.add_argument('-sd', dest="sd_samp", default=0.1, type=float)
    parser.add_argument('-ss', dest="samp_size", default=100, type=int)
    parser.add_argument('-t', dest="test_prop", default=0.2, type=float)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-di', dest="dir", default='syn_data', type=str)
    parser.add_argument('-e', dest="ep", default=3, type=int)
    parser.add_argument('-lr', dest="lr", default=0.001, type=float)
    parser.add_argument('-bs', dest="bs", default=10, type=int)
    parser.add_argument('-drop', dest="drop", default=2, type=int)
    parser.add_argument('-dr', dest="dr", default=0.2, type=float)
    args = parser.parse_args()
    return args

#####################
### Create the correct Torch format
#####################

def csv_torch(df):

    # Creating Tuples
    cols = (df.columns)
    rows = np.unique(df.index)

    samps = []

    for i in rows:
        s  = df.loc[int(i),:]
        for j in cols:
            t = torch.Tensor(np.array(s[str(j)]))
            tup = (t, int(i))
            samps.append(tup)
    return samps

#####################
### Forward Function for the Neural Network - With Acc and Loss
#####################

def fwd_pass(X, y, train = True): # Specify whether to update the weights or not
    correct = 0; total = 0

    if train:
        net.zero_grad()

    outputs = net(X)

    for idx, i in enumerate(outputs):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

    acc = round(correct/total, 3)
    loss = loss_function(outputs, y) 

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

#####################
### Getting the Validation Accuracie and Loss - TEST
#####################

def test_vals(size = 10):
    counter = 0
    acc_list = []
    loss_list = []

    with torch.no_grad():
        for data in testset:
            counter += 1
            X, y = data
            val_acc, val_loss = fwd_pass(X.view(-1, args.dims).to(device),y.to(device), train = False) 
            acc_list.append(val_acc)
            loss_list.append(val_loss.item())
            if counter == size:
                break

    return float(np.mean(acc_list)), float(np.mean(loss_list))


#####################
### Getting F1 Scores
#####################

def f1_score(p,m):
    target =  [] 
    preds =  [] 

    with torch.no_grad():
        for data in testset:
            X, y = data
            X, y = X.to(device), y.to(device) 
            output = net(X.view(-1,args.dims))
            for idx, i in enumerate(output):
                preds.append(int(torch.argmax(i)))
                target.append(int(y[idx]))

    preds = torch.tensor(preds)
    target = torch.tensor(target)

    # accuracy = Accuracy()
    # accuracy(preds, target)

    f1 = F1Score(num_classes= args.n_c)
    f_score = [float(f1(preds, target))]

    dir = 'results_f1'
    new_path = os.path.join(p, dir)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    name = os.path.join(new_path, f"%dc%dd%ds_F1_{m}.txt" % (args.n_c, args.dims, args.samp_size))
    with open(name, 'w') as f:
        for line in f_score:
            f.write('F1-Score:' + str(line))
            f.write('\n')


#####################
### Creating the Plots
#####################


def create_acc_loss_graph(model_name, p):
    dir = 'NN_results'
    contents = open(os.path.join(os.getcwd(),dir, f"{MODEL_NAME}.log"), 'r').read().split('\n')

    times= []
    accuracies = []
    losses = []
    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(',')

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

            
    fig = plt.figure(figsize=(12,8))

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex = ax1)

    ax1.plot(times, accuracies, label = 'acc')
    ax1.plot(times, val_accs, label = 'val_acc')
    ax1.legend(loc=2)

    ax2.plot(times, losses, label = 'loss')
    ax2.plot(times, val_losses, label = 'val_loss')
    ax2.legend(loc=2)

    dir = 'results_plots'
    new_path = os.path.join(p, dir)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    plt.savefig(os.path.join(new_path, f'{MODEL_NAME}.png'))



#####################
### Start of Programm
#####################



if __name__ == "__main__":
    # Parse user arguments
    args = parse_args()

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = os.path.join(os.getcwd(), args.dir)
    path_train = path +  ("/%dc%dd%ds_train.csv" % (args.n_c, args.dims, args.samp_size))
    path_test = path +  ("/%dc%dd%ds_test.csv" % (args.n_c, args.dims, args.samp_size))
    train = pd.read_csv(path_train, index_col=0)
    test = pd.read_csv(path_test, index_col=0)

    dat_train = csv_torch(train)
    dat_test = csv_torch(test)

    trainset = torch.utils.data.DataLoader(dat_train, batch_size = args.bs, shuffle = True)
    testset = torch.utils.data.DataLoader(dat_test, batch_size = args.bs, shuffle = True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(args.dims, 300) # Input Layer
            self.fc1_bn = nn.BatchNorm1d(300)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(300, 300) # Hidden Layer 1
            self.fc2_bn = nn.BatchNorm1d(300)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc3 = nn.Linear(300, 300) # Hidden Layer 2
            self.fc3_bn = nn.BatchNorm1d(300)
            self.relu2 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(300, args.n_c) # Output Layer
            self.dropout = nn.Dropout(args.dr)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc1_bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.fc2_bn(x)
            x = self.relu1(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.fc3_bn(x)
            x = self.relu2(x)
            x = self.fc4(x)
            return F.log_softmax(x, dim=1) # We want to sum the classes - across columns

    # GPU Settings
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    net = Net().to(device)

    learning_rate = args.lr
    loss_function = nn.CrossEntropyLoss() # PeBusePenalty
    optimizer = optim.Adam(net.parameters(), lr = learning_rate) # get_optimizer

    EPOCHS = args.ep
    MODEL_NAME = f"model-{int(time.time())}"

    def train(net, model, ep, lr):
        MODEL_NAME = model
        BATCH_SIZE = trainset.batch_size
        EPOCHS = ep
        learning_rate = lr

        dir = 'NN_results'
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        name = os.path.join(dir, f"{MODEL_NAME}.log")

        with open(name, "a") as f:
            for epoch in range(EPOCHS):
                print(epoch)
                c = 0

                # Learning rate decay.
                if epoch == args.drop:
                    learning_rate *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                for data in trainset:
                    c += BATCH_SIZE
                    X, y = data
                    batch_X = X
                    batch_y = y

                    batch_X, batch_y = batch_X.to(device), batch_y.to(device) 

                    acc, loss = fwd_pass(batch_X, batch_y, train=True)

                    if c % 100 == 0:
                        val_acc, val_loss = test_vals(10)
                        f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")


    train(net, MODEL_NAME, EPOCHS, learning_rate)

    p = os.path.join(os.getcwd(), "NN_results")

    f1_score(p, MODEL_NAME)

    create_acc_loss_graph(MODEL_NAME, p)


# python NN_SynData_2.py -c 10 -d 300 -sd 3 -ss 1000 -t 0.2 -s 300 -di syn_data -e 3 -lr 1e-3 -bs 100 -drop 2 -dr 0.2
