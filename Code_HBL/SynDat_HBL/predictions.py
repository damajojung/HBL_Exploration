import argparse
import math
import os
import time
import random


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils

from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score

from helper import pmath
from helper.helper import get_optimizer, load_dataset 
from helper.hyperbolicLoss import PeBusePenalty
from models.cifar import resnet as resnet_cifar
from models.cifar import densenet as densenet_cifar
from models.cub import resnet as resnet_cub
from models.syndat import fullcon as fullcon_syndat
from models.syndat import fullcon_selu as fullcon_syndat_selu
from models.syndat import fullcon_lrelu as fullcon_syndat_lrelu
from models.syndat import fullcon_200 as fullcon_syndat_200


def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument("--data_name", dest="data_name", default="cifar100",
                        choices=["cifar100", "cifar10", "cub", "syndat"], type=str)  # choose tha name of the dataset

    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("--logdir", dest="logdir", default="", type=str)
    parser.add_argument("--loss", dest="loss_name", default="PeBuseLoss", type=str)
    parser.add_argument("-mpath", dest="mpath", default="models_10d-10c", type=str)
    parser.add_argument("-mname", dest="mname", default = '1653033082_mult0.0_dec5e-05_r0.3.pt', type=str)

    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-cv", dest="clip_value", default=15.0, type=float)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float) # LR decay
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("-p", dest="penalty", default='dim', type=str)  # choose penalty in loss
    parser.add_argument("--mult", dest="mult", default=0.1, type=float) # Regularisation Term for HBL
    parser.add_argument("--curv", dest="curv", default=1.0, type=float)

    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=500, type=int)
    parser.add_argument("--drop2", dest="drop2", default=1000, type=int)
    parser.add_argument("--do_decay", dest="do_decay", default=False, type=bool)

    # New HBL  dims = 10, n_c = 10, dr = 0.2
    parser.add_argument('-nc', dest="n_c", default=10, type=int)
    parser.add_argument('-d', dest="dims", default=10, type=int)
    parser.add_argument('-sd', dest="sd_samp", default=0.1, type=float)
    parser.add_argument('-ss', dest="samp_size", default=100, type=int)
    parser.add_argument('-t', dest="test_prop", default=0.2, type=float)
    parser.add_argument('-di', dest="dir", default='syn_data', type=str)
    parser.add_argument('-lr', dest="lr", default=0.001, type=float)
    parser.add_argument('-bs', dest="bs", default=10, type=int)
    parser.add_argument('-drop', dest="drop", default=2, type=int)
    parser.add_argument('-dr', dest="dr", default=0.2, type=float) # Dropout Rate in NN

    parser.add_argument('-r_val', dest="r_val", default=0.7, type=float) # r Value for exp. Clipping
    parser.add_argument('-os', dest="os", default=1, type=int) # Observed Class for Loss and Plot



    args = parser.parse_args()
    return args


#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 4, 'pin_memory': True}

    path = os.getcwd()
    path = path + '/saved_models/'
    model_dir = args.mpath
    model_name = args.mname
    model_path = path + '/' + model_dir  + '/' + model_name

    print(model_path)

    # hpnfile name is like prototypes-xd-yc.npy : x : dimension of prototype, y: number of classes
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])

    # Load the polars and update the trainy labels.
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()

    # calculate radius of ball
    # This part is useful when curvature is not 1.
    curvature = args.curv
    radius = 1.0 / math.sqrt(curvature)
    classpolars = classpolars * radius

    # Data
    # Load data.
    batch_size = args.batch_size
    trainloader, testloader = load_dataset(dataset_name = args.data_name,
         basedir = args.datadir,
             batch_size = batch_size,
                 kwargs = kwargs,
                    n_c = args.n_c,
                    dims = args.dims,
                    samp_size = args.samp_size) 

    # Model
    model = fullcon_syndat.fullcon(dims = args.dims, output_dims = args.output_dims, dr = args.dr, polars = classpolars)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    # Go over all batches.
    c = 1.0
    tz = torch.zeros(args.n_c).cuda() # For the predictions uniform distribution plot
    norm_vals = [] # For the distribution of the norms plot
    counter = 0
    memory = 0
    with torch.no_grad():
        for bidx, (data, target) in enumerate(testloader):

            ## Investigate a specific Sample
            if args.os in target and memory == 0:
                memory += bidx
                print(f'Target located: bidx {memory}')

            if bidx == memory and memory != 0:
                # print(target)
                indextarget = (target == args.os).nonzero(as_tuple=True)[0][0].item()
                target_class = (target[indextarget])
                target_class = model.polars[target_class].view(1,args.output_dims)
                data_class = (data[indextarget]).view(1, args.dims)

                target_class = torch.autograd.Variable(target_class).cuda() # Target
                data_class = torch.autograd.Variable(data_class).cuda() # Sample

                # # Compute outputs and losses.
                model.eval()
                output_class = model(data_class)
                model.train()
                # print('output_class', output_class.shape) # torch.Size([12, 10])

                # # Clip exp map 
                outputnorm_class = torch.norm(output_class, dim= -1) 
                # print('outputnorm_class', outputnorm_class.shape) # torch.Size([12])
                clipped_output_class = torch.multiply(torch.minimum(torch.tensor(1.0).cuda(), (args.r_val / outputnorm_class).view(-1, 1)), output_class)
                # print('clipped_output_class', clipped_output_class.shape) # torch.Size([12, 10])
                output_exp_map_class = pmath.expmap0(clipped_output_class, c=c)
                # print('output_exp_map_class', output_exp_map_class) # torch.Size([12, 10])
                # print('target_class', target_class) # torch.Size([12])
                loss_function_class = initialized_loss(output_exp_map_class, target_class)


            # From here onwards regular code.
            # Data to device.
            data = torch.autograd.Variable(data).cuda() # torch.Size([200, 300])
            target = target.cuda(non_blocking=True)
            target = torch.autograd.Variable(target)
            target_loss = model.polars[target]

            # Forward.
            output = model(data).float() # torch.Size([200, 10])
            output_exp_map = pmath.expmap0(output, c=c) # torch.Size([200, 10])

            # Save the exp. map coordinated for circle plots
            if args.output_dims == 2:
                if counter == 0:
                    save_exp_map = output_exp_map
                    targets = target
                else:
                    save_exp_map = torch.cat((save_exp_map, output_exp_map))
                    targets = torch.cat((targets, target))


            # Calculate Norm of x^H
            for i in output_exp_map:
                outputnorm = torch.norm(i, dim= -1)
                norm_vals.append(torch.norm(outputnorm).item())
            

            output = model.predict(output_exp_map).float() # torch.Size([200, 10])
            output = output.max(1, keepdim=True)[1].view(output.shape[0],)
            tz += torch.bincount(output, minlength = args.n_c)
            counter += 1

######################################################################################
    # Save everything

    # Save model     
    current_path = os.getcwd()
    exp_path = (current_path + '/experiments_results')
    class_dims = args.mpath.split('models_')[1]

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    dir = f'{class_dims}'
    final_model_path = (exp_path + '/' + dir)
    if not os.path.exists(final_model_path):
        os.mkdir(final_model_path)
    
    dir_uni = 'uni_plots'
    dir_dist_norm = 'dist_norm'

    if args.output_dims == 2:
        dir_exp_out_data = 'exp_out_data'

    end_uni_path = final_model_path + '/' + dir_uni
    end_dist_norm_path = final_model_path + '/' + dir_dist_norm

    if args.output_dims == 2:
        end_exp_out_data = final_model_path + '/' + dir_exp_out_data

    if not os.path.exists(end_uni_path):
        os.mkdir(end_uni_path)

    if not os.path.exists(end_dist_norm_path):
        os.mkdir(end_dist_norm_path)

    if args.output_dims == 2:
        if not os.path.exists(end_exp_out_data):
            os.mkdir(end_exp_out_data)


    torch.save(tz, os.path.join(end_uni_path, f"{model_name}"))
    torch.save(torch.tensor(norm_vals), os.path.join(end_dist_norm_path, f"{model_name}"))
    if args.output_dims == 2:
        torch.save(save_exp_map, os.path.join(end_exp_out_data, f"data_{model_name}"))
        torch.save(targets, os.path.join(end_exp_out_data, f"target_{model_name}"))

###################################################################################### Playground

    # print('tz:', tz)
    # print('norm_vals:', norm_vals)
    # print('Length norm vec:', len(norm_vals))
    # print('mean norm vals:', np.mean(norm_vals))
    # print('Test has been successful.')

# 10d-10c
# python predictions.py --data_name syndat --datadir data/ -nc 10 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-10d-10c.npy -mpath models_10d-10c -mname 1653033082_mult0.0_dec5e-05_r0.3.pt

######################################################################## Area of interest

# 100d-100c
# python predictions.py --data_name syndat --datadir data/ -nc 100 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-100d-100c.npy -mpath models_100d-100c -mname 1653049123_mult0.0_dec5e-05_r0.3.pt

# 1000d-100c
# python predictions.py --data_name syndat --datadir data/ -nc 100 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-100c.npy -mpath models_1000d-100c -mname 1653050249_mult0.0_dec5e-05_r0.3.pt

# 100d-1000d
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-100d-1000c.npy -mpath models_100d-1000c -mname 1653050341_mult0.0_dec5e-05_r0.3.pt

# 1000d-1000c
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653050401_mult0.0_dec5e-05_r0.3.pt

### Learning Rate 1e-5

# 1000d-1000c - Learning Rate 1e-5
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653315834_mult0.0_dec5e-05_lr1e-05_r0.01.pt

# 1000d-1000c - Learning Rate 1e-5 & r 0.02
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653379222_mult0.0_dec5e-05_lr1e-05_r0.02.pt

######### Investigate Skewness

# # 1000d-1000c - OS 768 and seed 1
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653578692_mult0.0_dec5e-05_lr0.001_r0.3.pt

# # 1000d-1000c - OS 768 and but r 0.02
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653578783_mult0.0_dec5e-05_lr0.001_r0.02.pt

# # 1000d-1000c - OS 10 - otherwise the same
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653580326_mult0.0_dec5e-05_lr0.001_r0.3.pt

# !!!!! new data !!!!
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653586329_mult0.0_dec5e-05_lr0.001_r0.3.pt

# No Clipping
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-1000d-1000c.npy -mpath models_1000d-1000c -mname 1653632570_mult0.0_dec5e-05_lr0.001_r0.3.pt

########################################################################

######################################################################## Circle Plots 2D

# 1000 Classes

# 0.01
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-1000c.npy -mpath models_2d-1000c -mname 1653206820_mult0.0_dec5e-05_r0.01.pt

# 0.3
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-1000c.npy -mpath models_2d-1000c -mname 1653125460_mult0.0_dec5e-05_r0.3.pt

# 0.7
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-1000c.npy -mpath models_2d-1000c -mname 1653125460_mult0.0_dec5e-05_r0.7.pt

# 5.0
# python predictions.py --data_name syndat --datadir data/ -nc 1000 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-1000c.npy -mpath models_2d-1000c -mname 1653125460_mult0.0_dec5e-05_r5.0.pt


########################################################################

# 100 Classes

# 0.01
# python predictions.py --data_name syndat --datadir data/ -nc 100 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-100c.npy -mpath models_2d-100c -mname 1653145939_mult0.0_dec5e-05_r0.01.pt

# 0.7
# python predictions.py --data_name syndat --datadir data/ -nc 100 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-100c.npy -mpath models_2d-100c -mname 1653146258_mult0.0_dec5e-05_r0.7.pt

# 5.0
# python predictions.py --data_name syndat --datadir data/ -nc 100 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-100c.npy -mpath models_2d-100c -mname 1653146509_mult0.0_dec5e-05_r5.0.pt



########################################################################


# 10 Classes

# 0.01
# # python predictions.py --data_name syndat --datadir data/ -nc 10 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-10c.npy -mpath models_2d-10c -mname 1653143722_mult0.0_dec5e-05_r0.01.pt

# 0.7
# # python predictions.py --data_name syndat --datadir data/ -nc 10 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-10c.npy -mpath models_2d-10c -mname 1653143784_mult0.0_dec5e-05_r0.7.pt

# 5.0
# # python predictions.py --data_name syndat --datadir data/ -nc 10 -s 200 -d 300 -ss 1000 --hpnfile prototypes/prototypes-2d-10c.npy -mpath models_2d-10c -mname 1653143851_mult0.0_dec5e-05_r5.0.pt






