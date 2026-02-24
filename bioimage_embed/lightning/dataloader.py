from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split
from typing import Tuple
from functools import partial
import numpy as np


class StratifiedSampler(WeightedRandomSampler):
    def __init__(self, dataset, replacement=True):
        # Get the labels (targets) from the dataset
        self.targets = np.array([dataset[i][1] for i in range(len(dataset))])

        # Count the occurrences of each class
        class_counts = np.bincount(self.targets)

        # Calculate the weight for each class (inverse of frequency)
        class_weights = 1.0 / class_counts

        # Assign weights to each sample based on its class
        sample_weights = class_weights[self.targets]

        # Initialize the parent class (WeightedRandomSampler) with sample weights
        super().__init__(
            weights=sample_weights,
            num_samples=len(self.targets),
            replacement=replacement,
        )


# https://stackoverflow.com/questions/74931838/cant-pickle-local-object-evaluationloop-advance-locals-batch-to-device-pyto
class Collator:
    def collate_filter_for_none(self, batch):
        """
        Collate function that filters out None values from the batch.

        Args:
            batch: The batch to be filtered.

        Returns:
            The filtered batch.
        """
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def __call__(self, incoming):
        # do stuff with incoming
        return self.collate_filter_for_none(incoming)


class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling dataset loading and splitting.

    Attributes:
        dataset: The dataset to be handled.
        batch_size: The size of each batch.
        num_workers: The number of workers for data loading.
        pin_memory: Whether to use pinned memory for data loading.
        drop_last: Whether to drop the last incomplete batch.
        collate_fn: The function to use for collating data into batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        drop_last: bool = False,
        # sampler=None,
        sampler=StratifiedSampler,
    ):
        """
        Initializes the DataModule with the given dataset and parameters.

        Args:
            dataset: The dataset to be handled.
            batch_size: The size of each batch. Default is 32.
            num_workers: The number of workers for data loading. Default is 4.
            pin_memory: Whether to use pinned memory for data loading. Default is False.
            drop_last: Whether to drop the last incomplete batch. Default is False.
            collate_fn: The function to use for collating data into batches. Default is None.
        """
        super().__init__()
        self.dataset = dataset
        # TODO this is a hack to get the dataset to work with the dataloader
        self.data_loader_settings = {
            "batch_size": batch_size,
            # batch_size:32,
            "num_workers": num_workers,
            "pin_memory": True,
            "shuffle": False,
            "sampler": sampler,
            "drop_last": True,
            # "collate_fn": self.collate_wrapper(self.collate_filter_for_none),
            # "collate_fn": self.collate_filter_for_none,
        }

    def get_dataset(self):
        return self.dataset

    def train_dataloader(self):
        return self.init_dataloader(
            self.train_dataset,
            shuffle=False,
            sampler=self.sampler(self.train_dataset),
        )

    def val_dataloader(self):
        return self.init_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.init_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.init_dataloader(self.dataset, shuffle=False)

    def init_dataloader(self, dataset, shuffle=False, sampler=None):
        """
        Initializes a dataloader for the given dataset.

        Args:
            dataset: The dataset to be loaded.
            shuffle: Whether to shuffle the dataset. Default is False.

        Returns:
            The dataloader for the dataset.
        """
        return (
            self.dataloader(
                dataset,
                shuffle=shuffle,
                sampler=sampler,
            )
            if dataset
            else None
        )


def valid_indices(dataset):
    valid_indices = []
    # Iterate through the dataset and apply the transform to each image
    for idx in range(len(dataset)):
        try:
            image, label = dataset[idx]
            # If the transform works without errors, add the index to the list of valid indices
            valid_indices.append(idx)
        except Exception as e:
            # A better way to do with would be with batch collation
            print(f"Error occurred for image {idx}: {e}")

    # Create a Subset using the valid indices
    subset = torch.utils.data.Subset(dataset, valid_indices)
    return subset
