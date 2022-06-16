import argparse
import math
import os
import wandb
import torch.nn as nn
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as torch_models

from helper import pmath
from helper.helper import get_optimizer, load_dataset 
from helper.hyperbolicLoss import PeBusePenalty
from models.cifar import resnet as resnet_cifar
from models.cifar import densenet as densenet_cifar
from models.cub import resnet as resnet_cub
from models.syndat import fullcon as fullcon_syndat
from models.syndat import fullcon_selu as fullcon_syndat_selu
from models.syndat import fullcon_lrelu as fullcon_syndat_lrelu
from models.imnet import resnet_18_pre_trained as resnet_imnet
# from models.imnet.resnet_18_pre_trained import get_model, predict
# from models.imnet import resnet_18 as resnet_imnet_18


def main_train(model, trainloader, optimizer, initialized_loss, c=1.0):
    # Set mode to training.
    model.train()
    avgloss, avglosscount, newloss, acc, newacc = 0., 0., 0., 0., 0.

    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        # bidx: Number of batches -> reaches from 0 - len(trainloader)
        # dara: btach_size tensors
        # Target: A tensor containing numbers from 0-100 (sice its cifar 100, I assume these
        # are the numbers of the class - basically the target and the y in my code)

        # Data to device.
        target_tmp = target.cuda()
        target = model.polars[target]
        data = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()
        # Compute outputs and losses.
        output = model(data)

        # Clip exp map 
        outputnorm = torch.norm(output, dim=-1)
        clipped_output = torch.multiply(torch.minimum(torch.tensor(1.0).cuda(), (args.r_val / outputnorm).view(-1, 1)), output)
        output_exp_map = pmath.expmap0(clipped_output, c=c)
        loss_function = initialized_loss(output_exp_map, target)

        # Backpropagation.
        optimizer.zero_grad()
        loss_function.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm= args.clip_value) # clip_grad_value_ = clip_value
        optimizer.step()

        avgloss += loss_function.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount

        output = model.predict(output_exp_map).float()
        pred = output.max(1, keepdim=True)[1]
        acc += pred.eq(target_tmp.view_as(pred)).sum().item()

    trainlen = len(trainloader.dataset)
    newacc = acc / float(trainlen)

    wandb.log({"loss": newloss})
    wandb.log({"acc": newacc})

    # I am returning new loss to show in the tensorboard!
    return newacc, newloss


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


def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument("--data_name", dest="data_name", default="cifar100",
                        choices=["cifar100", "cifar10", "cub", "syndat", 'pthimnet', 'imnet', 'pthimnet32', 'pthimnet64'], type=str)  # choose tha name of the dataset

    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("--logdir", dest="logdir", default="", type=str)
    parser.add_argument("--loss", dest="loss_name", default="PeBuseLoss", type=str)

    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-cv", dest="clip_value", default=15.0, type=float)
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
    # parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-di', dest="dir", default='syn_data', type=str)
    # parser.add_argument('-e', dest="ep", default=3, type=int)
    parser.add_argument('-lr', dest="lr", default=0.001, type=float)
    parser.add_argument('-bs', dest="bs", default=10, type=int)
    parser.add_argument('-drop', dest="drop", default=2, type=int)
    parser.add_argument('-dr', dest="dr", default=0.2, type=float)

    parser.add_argument('-r_val', dest="r_val", default=0.7, type=float)

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

    hb_file = str(args.hpnfile.split("/")[-1].split(".")[0].split('prototypes-')[1])
    model_name = f'{int(time.time())}_mult{args.mult}_dec{args.decay}_r{args.r_val}'
    wandb.init(project= f'imnet_{hb_file}', entity="damajo", name = model_name)

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
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    if (args.data_name == "cifar100") or (args.data_name == "cifar10"):                         # CIFAR
        if args.network == "resnet32":
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
        elif args.network == "densenet121":
            model = densenet_cifar.DenseNet121(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
    elif args.data_name == "cub":                                                               # CUB
        if args.network == "resnet32":
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
    elif args.data_name == "syndat":                                                            # Synthetic Data
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
    elif (args.data_name == "imnet") or (args.data_name == "pthimnet") or (args.data_name == "pthimnet32") or (args.data_name == "pthimnet64"):   # ImageNet
        if args.network == "resnet32":
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
        elif args.network == "resnet18":
            print('Resnet-18 is selected for ImageNet.')
            # resnet_imnet_18.ResNet(18, args.output_dims, 1, classpolars) # Option 1
            model = resnet_imnet.ResNet18(args.output_dims, classpolars)
            print('Model has been activated.')
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
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
    for i in range(args.epochs):
        print(i)

        # Learning rate decay.
        if i in [args.drop1, args.drop2] and do_decay:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Train and test.
        acc, loss = main_train(model, trainloader, optimizer, f_loss, c=curvature)

        # add the train loss to the tensorboard writer
        # writer.add_scalar("Loss/train", loss, i)
        # writer.add_scalar("Accuracy/train", acc, i)

        if i != 0 and (i % 2 == 0 or i == args.epochs - 1): # i % 10
            test_acc, test_loss = main_test(model, testloader, f_loss, c=curvature)

            testscores.append([i, test_acc])

            # writer.add_scalar("Loss/test", test_loss, i)
            # writer.add_scalar("Accuracy/test", test_acc, i)
    
        
    # Save model 
    current_path = os.getcwd()
    model_path = (current_path + '/saved_models')

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