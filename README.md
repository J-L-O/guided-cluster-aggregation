# Guided Cluster Aggregation

This repository contains the code for our paper **Guided Cluster Aggregation: 
A Hierarchical Approach to Generalized Category Discovery**.

## Installation

The dependencies can be installed via conda:

```conda env create -f environment.yaml```

To activate the environment run

```conda activate gca```

## Data

We use the follwoing datasets in our paper:

- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-100](https://image-net.org/) (subset of regular ImageNet)
- [CUB200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

## Base Models

We use two pretrainings in our paper: 
[GCD](https://github.com/sgvaze/generalized-category-discoveryL) and
[PromptCAL](https://github.com/sheng-eatamath/PromptCAL).
While PromptCAL offers pretrained models for download, GCD does not, so 
you will have to train the GCD models yourself.

## Usage

We provide experiment configs in the `./config/experiment` folder.
The naming scheme is `./config/experiment/{pretraining}_{dataset}.yaml`.
We'll use the `./config/experiment/promptcal_cub.yaml` config as an example.
To reproduce the results from our paper follow these steps:

### 1. Obtain the correct pretrained model

In this case just use the link from the PromptCAL repository. 
Place it in the directory specified in the config. 
The paths given in the config file are relative to the `./src` folder, so in 
our case the full path of the pretrained model would be 
`./pretrained/GCD/promptcal_cub.ckpt`.

### 2. Compute the nearest neighbors.

To do so, open a shell in the `./src` folder
and run 

```python evaluate.py trainer=debug experiment=promptcal_cub datamodule.data_dir=[data location] lightning_module.knn_file=null```

The nearest neighbor file will be saved in the 
`./src/evaluation_results/promptcal_cub` folder.
Place it in the directory specified in the config, in this case
`./neighbors/cub_promptcal_finetuned.npy`.

### 3. Run the experiment

To do so, open a shell in the `./src` folder
and run 

```python train.py trainer=train experiment=promptcal_cub datamodule.data_dir=[data location]```

This will run the model using tensorboard logging.
You can also use wandb by adding 
`trainer/logger=wandb trainer.logger.entity=[entity] trainer.logger.project=[project]` 
to the command.

You can also run the training on a slurm cluster by adapting
`./config/cluster/example.yaml` and `./config/slurm_config/gca_gcd_example.yaml`.
Then run

```python train.py cluster=example trainer=train experiment=promptcal_cub datamodule.data_dir=[data location]```


### Acknowledgements

Most of the dataset loading code is based on 
[GCD](https://github.com/sgvaze/generalized-category-discoveryL).
The code for the VPT-Vit is based on 
[PromptCAL](https://github.com/sheng-eatamath/PromptCAL), as well as some code 
regarding the experiments with less labeled data (e.g. CIFAR100 c10l50).
Both repositories are licensed under the MIT license.
