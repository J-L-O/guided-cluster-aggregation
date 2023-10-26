from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.datasets.knn import KNNDataset
from data.datasets.wrappers import DictWrapperDataset
from data.gcd_datasets.get_datasets import get_datasets, get_class_splits


class GCDDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        osr_split_dir: str,
        use_ssb_splits: bool,
        dataset_name: str,
        prop_train_labels: float,
        knn_file: str,
        num_neighbors: int,
        train_transform: torch.nn.Module,
        val_transform: torch.nn.Module,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.use_ssb_splits = use_ssb_splits
        self.osr_split_dir = osr_split_dir
        self.dataset_name = dataset_name
        self.prop_train_labels = prop_train_labels
        self.knn_file = knn_file
        self.num_neighbors = num_neighbors
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = {}
        self.val_key_list = ["anchors", "targets", "uq_indices", "is_labeled"]

    def setup(self, stage: Optional[str] = None):
        image_size, known_classes, novel_classes = get_class_splits(
            self.dataset_name,
            use_ssb_splits=self.use_ssb_splits,
            osr_split_dir=self.osr_split_dir,
        )

        train_dataset, val_dataset_train, val_dataset_val = get_datasets(
            self.dataset_name,
            self.data_dir,
            self.train_transform,
            self.val_transform,
            known_classes,
            novel_classes,
            self.prop_train_labels,
        )

        train_dataset = KNNDataset(
            train_dataset,
            self.knn_file,
            self.num_neighbors,
        )
        val_dataset_train = KNNDataset(
            val_dataset_train,
            self.knn_file,
            self.num_neighbors,
        )
        val_dataset_val = DictWrapperDataset(
            val_dataset_val,
            self.val_key_list,
        )

        self.datasets["train"] = train_dataset
        self.datasets["val"] = [val_dataset_train, val_dataset_val]
    
    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if self.num_workers > 0 else False,
            )
            for dataset in self.datasets["val"]
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if self.num_workers > 0 else False,
            )
            for dataset in self.datasets["val"]
        ]
