import argparse
import numpy as np
from PIL import Image
import pickle
from utils import *
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# Test script

# Change this one to check other file



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


def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument("--i", dest="input_file", default="'/Users/dj/Desktop/MT_Desktop/ImageNet/Imagenet32_train'/train_data_batch_1", type=str)  # choose tha name of the dataset
    parser.add_argument("--g", dest="gen_images", default = True, action='store_true')
    parser.add_argument("--s", dest="sorted_histogram", default= True, action='store_true')

    args = parser.parse_args()
    return args


def load_data(input_file):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    return x, y

if __name__ == '__main__':
    args = parse_args()
    # input_file, gen_images, hist_sorted  = parse_args()
    x, y = load_data(args.input_file)

    # Lets save all images from this file
    # Each image will be 3600x3600 pixels (10 000) images

    blank_image = None
    curr_index = 0
    image_index = 0

    print('First image in dataset:')
    print(x[curr_index])

    if not os.path.exists('res'):
        os.makedirs('res')

    if args.gen_images:
        for i in range(x.shape[0]):
            if curr_index % 10000 == 0:
                if blank_image is not None:
                    print('Saving 10 000 images, current index: %d' % curr_index)
                    blank_image.save('res/Image_%d.png' % image_index)
                    image_index += 1
                blank_image = Image.new('RGB', (36*100, 36*100))
            x_pos = (curr_index % 10000) % 100 * 36
            y_pos = (curr_index % 10000) // 100 * 36

            blank_image.paste(Image.fromarray(x[curr_index]), (x_pos + 2, y_pos + 2))
            curr_index += 1

        blank_image.save('res/Image_%d.png' % image_index)

    graph = [0] * 1000

    for i in range(x.shape[0]):
        # Labels start from 1 so we have to subtract 1
        graph[y[i]-1] += 1

    if args.sorted_histogram:
        graph.sort()
        
    x = [i for i in range(1000)]
    ax = plt.axes()
    plt.bar(x = x, height=graph, color='darkblue', edgecolor='darkblue') # left=x
    ax.set_xlabel('Class', fontsize=20)
    ax.set_ylabel('Samples', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('res/Samples.pdf', format='pdf', dpi=1200)

    #  python Imagenet_Test.py --i /Users/dj/Desktop/MT_Desktop/ImageNet/Imagenet32_train/train_data_batch_1 
