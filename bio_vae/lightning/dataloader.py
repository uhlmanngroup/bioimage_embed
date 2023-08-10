import pytorch_lightning as pl
from .. import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import math
from iteround import saferound


class SimpleCustomBatch:
    def __init__(self, dataset):
        self.dataset = dataset

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.dataset = self.dataset.pin_memory()
        return self


class DatamoduleGlob(pl.LightningDataModule):
    
    def collate_wrapper(self,batch):
        return SimpleCustomBatch(batch)

    def __init__(
        self, glob_str, batch_size=32, num_workers=2**8, sampler=None, **kwargs
    ):
        super().__init__()
        self.glob_str = glob_str
        self.batch_size = batch_size
        self.dataset = datasets.DatasetGlob(glob_str, **kwargs)
        self.data_loader_settings = {
            "batch_size": batch_size,
            # batch_size:32,
            "num_workers": num_workers,
            "pin_memory": True,
            "shuffle": False,
            "sampler": sampler,
            # "collate_fn": self.collate_wrapper(self.collate_filter_for_none),
            # "collate_fn": self.collate_filter_for_none,
        }

    def get_dataset(self):
        return self.dataset


    def splitting(self, dataset, split_train=0.8, split_val=0.1, seed=42):
        if len(dataset) < 3:
            return dataset, dataset, dataset

        train_share = int(len(dataset) * split_train)
        val_share = int(len(dataset) * split_val)
        test_share = len(dataset) - train_share - val_share

        # Ensure that the splits add up correctly
        if train_share + val_share + test_share != len(dataset):
            raise ValueError("The splitting ratios do not add up to the length of the dataset")

        torch.manual_seed(seed)  # for reproducibility

        train, val, test = random_split(
            dataset,
            [train_share, val_share, test_share]
        )

        return train, val, test

    def setup(self, stage=None):
        self.train, self.val,  self.test = self.splitting(self.dataset)

        # self.test = self.get_dataloader(test)
        # self.predict = self.get_dataloader(predict)
        # self.train = self.get_dataloader(train)
        # self.val = self.get_dataloader(val)

    def test_dataloader(self):
        return DataLoader(self.test, **self.data_loader_settings)

    def train_dataloader(self):
        return DataLoader(self.train, **self.data_loader_settings)

    def val_dataloader(self):
        return DataLoader(self.val, **self.data_loader_settings)

    def predict_dataloader(self):
        return DataLoader(self.dataset, **self.data_loader_settings)

    # def teardown(self, stage: Optional[str] = None):
    #     # Used to clean-up when the run is finished
    #     ...

    def collate_filter_for_none(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
