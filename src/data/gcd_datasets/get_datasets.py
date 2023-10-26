from data.gcd_datasets.data_utils import MergedDataset

from data.gcd_datasets.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.gcd_datasets.herbarium_19 import get_herbarium_datasets
from data.gcd_datasets.stanford_cars import get_scars_datasets
from data.gcd_datasets.imagenet import get_imagenet_100_datasets
from data.gcd_datasets.cub import get_cub_datasets
from data.gcd_datasets.fgvc_aircraft import get_aircraft_datasets

from data.gcd_datasets.cifar import subsample_classes as subsample_dataset_cifar
from data.gcd_datasets.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.gcd_datasets.stanford_cars import subsample_classes as subsample_dataset_scars
from data.gcd_datasets.imagenet import subsample_classes as subsample_dataset_imagenet
from data.gcd_datasets.cub import subsample_classes as subsample_dataset_cub
from data.gcd_datasets.fgvc_aircraft import subsample_classes as subsample_dataset_air

import numpy as np
from copy import deepcopy
import pickle
import os


sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'cifar100_10': subsample_dataset_cifar,
    'cifar100_25': subsample_dataset_cifar,
    'cifar100_50': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar100_10': get_cifar_100_datasets,
    'cifar100_25': get_cifar_100_datasets,
    'cifar100_50': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets
}


def get_datasets(dataset_name, dataset_root, train_transform, test_transform,
                 train_classes, unlabeled_classes, prop_train_labels):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    all_labeled_classes = list(train_classes)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(
        root=dataset_root,
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=all_labeled_classes,
        prop_train_labels=prop_train_labels,
        split_train_val=False
    )

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(
        labelled_dataset=deepcopy(datasets['train_labelled']),
        unlabelled_dataset=deepcopy(datasets['train_unlabelled']),
    )

    val_dataset_train_unlabelled = deepcopy(datasets['train_unlabelled'])
    val_dataset_train_labelled = deepcopy(datasets['train_labelled'])
    val_dataset_train_unlabelled.transform = test_transform
    val_dataset_train_labelled.transform = test_transform

    val_dataset_train = MergedDataset(
        labelled_dataset=val_dataset_train_labelled,
        unlabelled_dataset=val_dataset_train_unlabelled,
    )

    val_dataset_val = MergedDataset(
        labelled_dataset=[],  # Hack to make the entire dataset unlabelled
        unlabelled_dataset=deepcopy(datasets['test']),
    )

    return train_dataset, val_dataset_train, val_dataset_val


def get_class_splits(dataset_name, use_ssb_splits=False, osr_split_dir=None):
    # -------------
    # GET CLASS SPLITS
    # -------------

    num_labeled_classes_default = {
        'cifar10': 5,
        'cifar100': 80,
        'cifar100_10': 10,
        'cifar100_25': 25,
        'cifar100_50': 50,
        'tinyimagenet': 100,
        'herbarium_19': 341,
        'imagenet_100': 50,
        'scars': 98,
        'aircraft': 50,
        'cub': 100,
        'chinese_traffic_signs': 28
    }

    num_labeled_classes = num_labeled_classes_default[dataset_name]

    if dataset_name == 'cifar10':

        image_size = 32
        train_classes = range(num_labeled_classes)
        unlabeled_classes = range(num_labeled_classes, 10)

    elif dataset_name == 'cifar100':

        image_size = 32
        train_classes = range(num_labeled_classes)
        unlabeled_classes = range(num_labeled_classes, 100)

    elif dataset_name == 'cifar100_10':
        np.random.seed(0)
        image_size = 32
        ### use random generated classes
        # args.train_classes = np.random.choice(np.arange(100), size=10, replace=False).tolist()
        train_classes = [18, 49, 67, 16, 72, 14, 39, 47, 35, 88]
        unlabeled_classes = list(set(range(100)) - set(train_classes))

    elif dataset_name == 'cifar100_25':
        np.random.seed(0)
        image_size = 32
        train_classes = np.random.choice(np.arange(100), size=25, replace=False).tolist()
        unlabeled_classes = list(set(range(100)) - set(train_classes))

    elif dataset_name == 'cifar100_50':
        np.random.seed(0)
        image_size = 32
        train_classes = np.random.choice(np.arange(100), size=50, replace=False).tolist()
        unlabeled_classes = list(set(range(100)) - set(train_classes))

    elif dataset_name == 'tinyimagenet':

        image_size = 64
        train_classes = range(num_labeled_classes)
        unlabeled_classes = range(num_labeled_classes, 200)

    elif dataset_name == 'herbarium_19':

        image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        train_classes = class_splits['Old']
        unlabeled_classes = class_splits['New']

    elif dataset_name == 'imagenet_100':

        image_size = 224
        train_classes = range(num_labeled_classes)
        unlabeled_classes = range(num_labeled_classes, 100)

    elif dataset_name == 'scars':

        image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            train_classes = range(num_labeled_classes)
            unlabeled_classes = range(num_labeled_classes, 196)

    elif dataset_name == 'aircraft':

        image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            train_classes = range(num_labeled_classes)
            unlabeled_classes = range(num_labeled_classes, 100)

    elif dataset_name == 'cub':

        image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            train_classes = range(num_labeled_classes)
            unlabeled_classes = range(num_labeled_classes, 200)

    elif dataset_name == 'chinese_traffic_signs':

        image_size = 224
        train_classes = range(num_labeled_classes)
        unlabeled_classes = range(num_labeled_classes, 56)

    else:

        raise NotImplementedError

    return image_size, train_classes, unlabeled_classes
