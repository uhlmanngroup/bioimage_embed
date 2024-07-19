import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple
from functools import partial


class SimpleCustomBatch:
    def __init__(self, dataset):
        self.dataset = dataset

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.dataset = self.dataset.pin_memory()
        return self


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn=None,
    ):
        super().__init__()
        self.dataset = dataset
        collate_fn = collate_fn if collate_fn else self.collate_filter_for_none
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.setup()

    def collate_filter_for_none(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def setup(self, stage=None):
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.splitting(self.dataset)

    def splitting(
        self, dataset: Dataset, split_train=0.8, split_val=0.1, seed=42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        train_size = int(split_train * dataset_size)
        val_size = int(split_val * dataset_size)
        test_size = dataset_size - train_size - val_size

        if train_size + val_size + test_size != dataset_size:
            raise ValueError(
                "The splitting ratios do not add up to the length of the dataset"
            )

        torch.manual_seed(seed)
        train_indices, val_indices, test_indices = random_split(
            indices, [train_size, val_size, test_size]
        )

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset

    def get_dataset(self):
        return self.dataset

    def train_dataloader(self):
        return self.init_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.init_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.init_dataloader(self.test_dataset, shuffle=False)

    def init_dataloader(self, dataset, shuffle=False):
        return (
            self.dataloader(
                dataset,
                shuffle=shuffle,
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
