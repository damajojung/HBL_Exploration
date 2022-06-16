import argparse
import math
import os
import wandb
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


def main_train(model, trainloader, optimizer, initialized_loss, ep, tz, c=1.0):
    # Set mode to training.
    model.train()
    avgloss, avgloss_class, avglosscount, newloss, acc, newacc = 0., 0., 0., 0., 0., 0.
    memory = 0
    memory_2d = 0
    tz = tz # For the predictions uniform distribution plot
    # tz = torch.zeros(args.n_c).cuda() # For the predictions uniform distribution plot
    # tz_target = torch.zeros(args.n_c).cuda() # For the predictions uniform distribution plot

    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        # bidx: Number of batches -> reaches from 0 - len(trainloader)
        # data: btach_size tensors
        # Target: A tensor containing numbers from 0-100 (sice its cifar 100, I assume these
        # are the numbers of the class - basically the target and the y in my code)

        ##############
        ####### Analysing Loss of specific Classes
        # indextarget = (target == args.os).nonzero(as_tuple=True)
        # target_class = (target[indextarget])
        # target_class = model.polars[target_class]
        # data_class = (data[indextarget])

        # target_class = torch.autograd.Variable(target_class).cuda()
        # data_class = torch.autograd.Variable(data_class).cuda()

        # # Compute outputs and losses.
        # model.eval()
        # output_class = model(data_class) # torch.Size([12, 10])
        # model.train()

        # # Clip exp map 
        # outputnorm_class = torch.norm(output_class, dim= -1) # torch.Size([12])
        # clipped_output_class = torch.multiply(torch.minimum(torch.tensor(1.0).cuda(), (args.r_val / outputnorm_class).view(-1, 1)), output_class) # torch.Size([12, 10])
        # output_exp_map_class = pmath.expmap0(clipped_output_class, c=c) # torch.Size([12, 10])
        # loss_function_class = initialized_loss(output_exp_map_class, target_class)

        #######
        #####################

        ##############
        ####### Analysing Loss of specific Sample from a specific Class

        if args.os in target and memory == 0:
            memory += bidx
            print(f'Target located: bidx {memory}')

        if bidx == memory and memory != 0:
            indextarget = (target == args.os).nonzero(as_tuple=True)[0][0].item()
            target_class = (target[indextarget])
            target_class = model.polars[target_class].view(1,args.output_dims)
            data_class = (data[indextarget]).view(1, args.dims)

            target_class = torch.autograd.Variable(target_class).cuda() # Target # torch.Size([12])
            data_class = torch.autograd.Variable(data_class).cuda() # Sample

            # # Compute outputs and losses.
            model.eval()
            output_class = model(data_class) # torch.Size([12, 10]
            model.train()

            # # Clip exp map 
            outputnorm_class = torch.norm(output_class, dim= -1) # torch.Size([12])
            clipped_output_class = torch.multiply(torch.minimum(torch.tensor(1.0).cuda(), (args.r_val / outputnorm_class).view(-1, 1)), output_class) # torch.Size([12, 10])
            output_exp_map_class = pmath.expmap0(clipped_output_class, c=c)# torch.Size([12, 10])
            loss_function_class = initialized_loss(output_exp_map_class, target_class)

            wandb.log({f"Loss Class {args.os}": loss_function_class})

        #######
        ####################
        # Data to device.
        target_tmp = target.cuda()
        # tz_target += torch.bincount(target_tmp.view(target_tmp.shape[0]), minlength = args.n_c)

        if args.output_dims == 2: # Saving the exp. outputs for 2d-plots
            if args.n_c == 1000:
                if args.os in target and memory_2d == 0:
                    memory_2d += bidx
                    target_index = (target == args.os).nonzero(as_tuple=True)[0][0].item() # Grab Index of Observed Sample of certain Class (-os = observed sample)
                elif bidx == memory_2d and memory_2d != 0:
                    target_index = (target == args.os).nonzero(as_tuple=True)[0][0].item()
            elif bidx == 0: # This works well for 10 and 100 classes
                target_index = (target == args.os).nonzero(as_tuple=True)[0][0].item() # Grab Index of Observed Sample of certain Class (-os = observed sample)

        target = model.polars[target]
        data = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()
        # print('Shape of data:', data.shape)
        # print('Shape of data[0]:', data[0].shape)
        # print('Shape of target:', target[0].shape)
        # Compute outputs and losses.
        output = model(data)

        ######################## Clipping
        # Clip exp map 
        outputnorm = torch.norm(output, dim= -1)
        clipped_output = torch.multiply(torch.minimum(torch.tensor(1.0).cuda(), (args.r_val / outputnorm).view(-1, 1)), output)
        output_exp_map = pmath.expmap0(clipped_output, c=c)
        ########################

        # If no clipping is wanted
        # output_exp_map = pmath.expmap0(output, c=c)
        # End no clipping wanted

        # If 2d Prototypes - Safe coordinates for Plotting
        if args.output_dims == 2:
            if args.n_c == 1000 and memory_2d != 0:
                with open(f"{log_path}/{model_name}.log", "a") as f:
                    f.write(f"{model_name},{round(time.time(),3)},{round(float(output_exp_map[target_index][0]),4)},{round(float(output_exp_map[target_index][1]),4)}\n")
            elif bidx == 0 and (args.n_c == 10 or args.n_c == 100):
                with open(f"{log_path}/{model_name}.log", "a") as f:
                    f.write(f"{model_name},{round(time.time(),3)},{round(float(output_exp_map[target_index][0]),4)},{round(float(output_exp_map[target_index][1]),4)}\n")

        ##########################

        # target = target.view(args.batch_size, args.n_c ,1) 
        # output_exp_map = output_exp_map.view(args.batch_size, args.n_c, 1)
        loss_function = initialized_loss(output_exp_map, target) # (Data, prototype)

        # Backpropagation.
        optimizer.zero_grad()
        loss_function.backward()
        # plot_grad_flow(model.named_parameters(), ep, bidx, l)
        # Gradient Clipping
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm= args.clip_value) # clip_grad_value_ = clip_value (only against exploding gradients)
        optimizer.step()

        # loss_function = torch.tensor with one value
        # loss_function.item() = just the float

        # Regular loss
        avgloss += loss_function.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount

        # Specific loss for one Class
        # avgloss_class += loss_function_class.item()
        # newloss_class = avgloss_class / avglosscount

        # Accuracy
        output = model.predict(output_exp_map).float()
        pred = output.max(1, keepdim=True)[1]
        tz += torch.bincount(pred.view(pred.shape[0]), minlength = args.n_c)
        acc += pred.eq(target_tmp.view_as(pred)).sum().item()
    

    trainlen = len(trainloader.dataset)
    newacc = acc / float(trainlen)
    # wandb.log({f"Class Loss {args.os}": newloss_class}) # Analysing Loss of specific Classes
    wandb.log({"loss": newloss})
    wandb.log({"acc": newacc})


    # I am returning new loss to show in the tensorboard!
    return newacc, newloss, tz


def main_test(model, testloader, initialized_loss, c=1.0):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    acc = 0
    loss = 0

    # Go over all batches.
    with torch.no_grad():
        for data, target in testloader:
            # Data to device.
            data = torch.autograd.Variable(data).cuda()
            target = target.cuda(non_blocking=True)
            target = torch.autograd.Variable(target)
            target_loss = model.polars[target]

            # Forward.
            output = model(data).float()
            output_exp_map = pmath.expmap0(output, c=c)

            output = model.predict(output_exp_map).float()
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target.view_as(pred)).sum().item()

            loss += initialized_loss(output_exp_map, target_loss.cuda())

    # Print results.
    testlen = len(testloader.dataset)

    avg_acc = acc / float(testlen)
    avg_loss = loss / float(testlen)

    wandb.log({"val_acc": avg_acc})
    wandb.log({"val_loss": avg_loss})

    return avg_acc, avg_loss


def plot_grad_flow(named_parameters, ep, iter, len_dat):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    if ep == (args.epochs - 1) and iter == len_dat - 1:
        plt.bar(np.arange(len(max_grads)), max_grads.cpu(), alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads.cpu(), alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers.cpu(), rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        log_dir = os.path.join('./runs/' + 'gradient_plot')
        plt.savefig(log_dir + '/grad.png')

def get_exp_plots(run):
    # matplotlib.use('Agg')
    curr_path =  os.getcwd()
    plot_path = os.path.join(curr_path + '/exp_plots')

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    contents = open(run, 'r').read().split('\n')
    if contents[-1] == '':
        contents = contents[:-1]

    names= []
    times = []
    x = []
    y = []

    for c in contents:
        name, time, x_value, y_value = c.split(',')
        names.append(str(name))
        times.append(float(time))
        x.append(float(x_value))
        y.append(float(y_value))

    # Defining Arrow Colors
    NUM_COLORS = len(x)
    cm = plt.get_cmap('gist_rainbow')
    colors_arr = []
    for i in range(NUM_COLORS):
        colors_arr.append(cm(i//1*1.0/NUM_COLORS))

    
    # Detailed Plot
    x = np.array(x)
    y = np.array(y)
    annotations = np.arange(0, len(x),1)

    colormap = plt.cm.get_cmap('gist_rainbow') # 'plasma' or 'viridis' or 'jet'
    colors = colormap(annotations)

    fig = plt.figure(figsize=(15,8))
    plt.subplot(2,2,2)

    plt.scatter(x, y, s= 200, color = 'black')
    plt.scatter(model.polars[args.os][0], model.polars[args.os][1], color = 'red', marker = 'X', s = 100)
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=0, vmax=NUM_COLORS)
    cbar = fig.colorbar(sm)
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width = 0.004, color = colors_arr)
    for i, label in enumerate(annotations):
        if i % 2 == 0:
            plt.annotate(label, (x[i], y[i]), fontsize=25)
    plt.title(f'Plot of Class {args.os}')

    # Circle Plot
    plt.subplot(2,2,1)

    # ax.set_aspect(1)
    theta = np.linspace(-np.pi, np.pi, 200)
    plt.plot(np.sin(theta), np.cos(theta), c = 'black')

    plt.scatter(x, y, s= 20, color = 'black')
    plt.scatter(0,0, color = 'black', marker = "x")
    plt.scatter(model.polars[args.os][0], model.polars[args.os][1], color = 'red', marker = 'X', s = 100)
    # sm = plt.cm.ScalarMappable(cmap=colormap)
    # sm.set_clim(vmin=0, vmax=NUM_COLORS)
    # plt.colorbar(sm)
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width = 0.004, color = colors_arr)
    for i, label in enumerate(annotations):
        if i % 2 == 0:
            plt.annotate(label, (x[i], y[i]), fontsize=25)
    plt.title(f'Poincaré Ball - Class {args.os}')
    plt.savefig(f'{plot_path}/{model_name}_{args.n_c}.png')
    
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument("--data_name", dest="data_name", default="cifar100",
                        choices=["cifar100", "cifar10", "cub", "syndat", "syndatzn"], type=str)  # choose tha name of the dataset

    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("--logdir", dest="logdir", default="", type=str)
    parser.add_argument("--loss", dest="loss_name", default="PeBuseLoss", type=str)

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

    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    #     print("Running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("Running on the CPU")

    hb_file = str(args.hpnfile.split("/")[-1].split(".")[0])
    model_name = f'{int(time.time())}_mult{args.mult}_dec{args.decay}_lr{args.learning_rate}_r{args.r_val}'
    wandb.init(project= 'sandbox_' + hb_file , entity="damajo", name = model_name)

    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 4, 'pin_memory': True}

    do_decay = args.do_decay
    curvature = args.curv

    # I want to use tensorboard to check the loss changes
    log_dir = os.path.join('./runs/' + args.data_name, args.logdir)
    writer = SummaryWriter(log_dir=log_dir)

    # Set the random seeds.
    # seed = args.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    # Load data.
    batch_size = args.batch_size
    trainloader, testloader = load_dataset(dataset_name = args.data_name,
         basedir = args.datadir,
             batch_size = batch_size,
                 kwargs = kwargs,
                    n_c = args.n_c,
                    dims = args.dims,
                    samp_size = args.samp_size) 

    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)

    curr_path =  os.getcwd()
    log_path = os.path.join(curr_path + '/model_logs')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Load the polars and update the trainy labels.
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()


    # calculate radius of ball
    # This part is useful when curvature is not 1.
    radius = 1.0 / math.sqrt(curvature)
    classpolars = classpolars * radius

    # hpnfile name is like prototypes-xd-yc.npy : x : dimension of prototype, y: number of classes
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])
    print('This is the output_dims:', args.output_dims)

    # Load the model.
    if (args.data_name == "cifar100") or (args.data_name == "cifar10"):
        if args.network == "resnet32":
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
        elif args.network == "densenet121":
            model = densenet_cifar.DenseNet121(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
    elif args.data_name == "cub":
        if args.network == "resnet32":
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
    elif args.data_name == "syndat" or args.data_name == "syndatzn":
        if args.network == "fullcon":
            print('The fully connected Network has been chosen for the Synthetic Data.')
            model = fullcon_syndat.fullcon(dims = args.dims, output_dims = args.output_dims, dr = args.dr, polars = classpolars)
            print('Model has been activated.')
        elif args.network == "fullcon_selu":
            print('The fully connected SELU Network has been chosen for the Synthetic Data.')
            model = fullcon_syndat_selu.fullcon(dims = args.dims, output_dims = args.output_dims, dr = args.dr, polars = classpolars)
            print('Model has been activated.')
        elif args.network == "fullcon_lrelu":
            print('The fully connected Leaky RELU Network has been chosen for the Synthetic Data.')
            model = fullcon_syndat_lrelu.fullcon(dims = args.dims, output_dims = args.output_dims, dr = args.dr, polars = classpolars)
            print('Model has been activated.')
        elif args.network == "fullcon_200":
            print('The fully connected 200 Network has been chosen for the Synthetic Data.')
            model = fullcon_syndat_200.fullcon(dims = args.dims, output_dims = args.output_dims, dr = args.dr, polars = classpolars)
            print('Model has been activated.')
    else:
        raise Exception('Selected dataset is not available.')

    model = model.to(device)
    print('First time model initialization.')

    # Observe Gradients and Parameters
    wandb.watch(model, log='all')

    # Load the optimizer.
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate, args.momentum, args.decay)

    # Initialize the loss functions.
    choose_penalty = args.penalty
    f_loss = PeBusePenalty(args.output_dims, penalty_option=choose_penalty, mult=args.mult).cuda()

    # Main loop.
    testscores = []
    learning_rate = args.learning_rate
    tz = torch.zeros(args.n_c).cuda() # For the predictions uniform distribution plot
    for i in range(args.epochs):
        print(i)

        # Learning rate decay.
        if i in [args.drop1, args.drop2] and do_decay:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Train and test.
        acc, loss, tz = main_train(model, trainloader, optimizer, f_loss, i, tz, c=curvature)


        ######
        ### Save tz and tz_target
        print('Saving process initialised.')
        current_path = os.getcwd()
        exp_path = (current_path + '/experiments_results')
        class_dims = hb_file

        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        dir = f'{class_dims}'
        final_model_path = (exp_path + '/' + dir)
        if not os.path.exists(final_model_path):
            os.mkdir(final_model_path)

        dir_bincount = 'bincount'
        end_dir_bincount = final_model_path + '/' + dir_bincount

        if not os.path.exists(end_dir_bincount):
            os.mkdir(end_dir_bincount)

        torch.save(tz, os.path.join(end_dir_bincount, f"{model_name}_tz_{i}.pt"))
        # torch.save(tz_target, os.path.join(end_dir_bincount, f"{model_name}_tz_target_{i}.pt"))
        ####### End safe tz


        # add the train loss to the tensorboard writer
        # writer.add_scalar("Loss/train", loss, i)
        # writer.add_scalar("Accuracy/train", acc, i)

        if i != 0 and (i % 1 == 0 or i == args.epochs - 1): # i % 10
            test_acc, test_loss = main_test(model, testloader, f_loss, c=curvature)

            testscores.append([i, test_acc])

            # writer.add_scalar("Loss/test", test_loss, i)
            # writer.add_scalar("Accuracy/test", test_acc, i)

    if args.output_dims == 2:
        fig = get_exp_plots(f"{log_path}/{model_name}.log")
    
    # Save model     
    current_path = os.getcwd()
    model_path = (current_path + '/saved_models')
    hb_file = str(hb_file.split('prototypes-')[1])

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    dir = f'models_{hb_file}'
    final_model_path = (model_path + '/' + dir)
    if not os.path.exists(final_model_path):
        os.mkdir(final_model_path)
    torch.save(model.state_dict(), os.path.join(final_model_path, f"{model_name}.pt"))

    writer.flush()
    writer.close()

# wandb login afd5dc6da7d5259f81a22c49850a846472147d88
# python HBL.py --data_name syndat -e 10 -s 200 -r adam -l 1e-3 -c 5e-5 --mult 0.0 --datadir data/ --resdir runs/output_dir/syndat/ --hpnfile prototypes/prototypes-1000d-100c.npy --logdir test --do_decay True --drop1 10 --drop2 13 --seed 300 -nc 100 -d 300 -ss 1000 -n fullcon -cv 15 -os 1 -r_val 0.3

# For Load and Safe Experiments - 10 classes 10 dimensions
# python HBL.py --data_name syndat -e 1 -s 200 -r adam -l 1e-4 -c 5e-5 --mult 0.0 --datadir data/ --resdir runs/output_dir/syndat/ --hpnfile prototypes/prototypes-10d-10c.npy --logdir test --do_decay True --drop1 20 --drop2 25 --seed 300 -nc 10 -d 300 -ss 1000 -n fullcon -cv 15 -os 5 -r_val 0.3






