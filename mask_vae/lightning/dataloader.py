import pytorch_lightning as pl
from .. import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import math
from iteround import saferound


class DatamoduleGlob(pl.LightningDataModule):
    def __init__(self, glob_str, batch_size=32, **kwargs):
        super().__init__()
        self.glob_str = glob_str
        self.batch_size = batch_size
        self.dataset = datasets.DatasetGlob(glob_str, **kwargs)
        self.data_loader_settings = {
            "batch_size": batch_size,
            # batch_size:32,
            "num_workers": 2**4,
            "pin_memory": True,
            "shuffle": False,
            "collate_fn": self.collate_filter_for_none,
        }

    def get_dataset(self):
        return self.dataset

    def splitting(self, dataset, split=0.8, seed=42):
        spliting_shares = [
            len(dataset) * split * split,  # train
            len(dataset) * split * (1 - split),  # test
            len(dataset) * split * (1 - split),  # predict
            len(dataset) * (1 - split) * (1 - split),  # val
        ]

        train, test, predict, val = random_split(
            dataset,
            list(map(int, saferound(spliting_shares, places=0))),
        )

        return test, train, predict, val

    def setup(self, stage=None):
        self.test, self.train, self.predict, self.val = self.splitting(self.dataset)

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
        return DataLoader(self.predict, **self.data_loader_settings)

    # def teardown(self, stage: Optional[str] = None):
    #     # Used to clean-up when the run is finished
    #     ...

    def collate_filter_for_none(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # def get_dataloader(
    #     self,
    #     dataset,
    #     batch_size=32,
    #     num_workers=2**4,
    #     pin_memory=True,
    #     shuffle=True,
    #     collate_fn=collate_filter_for_none,
    # ):
    #     return DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #         collate_fn=collate_fn,
    #     )
