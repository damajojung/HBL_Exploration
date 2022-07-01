# HBL-Exploration

Within the folder `HBL_Code`, one can find sub-folders for the experiments conducted for the Softmax, synthetic data and ImageNet data respectively. 
The code should be used the same as in: https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning 

## How to use?

The first step is to learn the ideal prototypes. One has to specify the amount of classes as well as output dimensions. This can be done as follows:

```
python prototype_learning.py -d 1000 -c 1000
```

Where `-d` stands for dimensions and `-c` amount of classes.

## Main Calculations

Once the prototypes are ready, one can use the main code as follows:

```
python HBL.py --data_name syndat -e 30 -s 200 -r adam -l 1e-3 -c 5e-5 --mult 0.0 --datadir data/ --resdir runs/output_dir/syndat/ --hpnfile prototypes/prototypes-1000d-1000c.npy --logdir test --do_decay True --drop1 20 --drop2 25 --seed 1 -nc 1000 -d 300 -ss 1000 -n fullcon -r_val 0.3 
```
* `-e` - Amount of training epochs
* `-s` - Batch size
* `-r` - Optimiser
* `-l` - Learning rate
* `-c`- Weight decay
* `--mult` - Penalisation term
* `--datadir` - Path to data
* `--hpnfile` - Ideal prototypes
* `drop1` - Training epoch where learning rate decay takes place
* `drop2` - Training epoch where learning rate decay takes place
* `--seed` - Seed
* `-nc` - Number of Classes
* `-d` - Dimensions used for the data
* `-ss` - Sample size per class
* `-n` - Which neural network should be used?
* `r_val` - r-value used for clipping
