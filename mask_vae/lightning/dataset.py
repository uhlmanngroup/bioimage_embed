import pytorch_lightning as pl
from .. import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class DatamoduleGlob(pl.LightningDataModule):
    def __init__(self, glob_str, batch_size=32):
        super().__init__()
        self.glob_str = glob_str
        self.batch_size = batch_size
        self.dataset = datasets.DatasetGlob()

    def splitting(self, dataset, split=0.8):

        train, test, predict, val = random_split(
            dataset,
            [
                len(dataset) * split * split, # train
                len(dataset) * split * 1-split, # test
                len(dataset) * split * 1-split, # predict
                len(dataset) * 1-split * 1-split, # val
            ],
            generator=torch.Generator().manual_seed(42),
        )

        return test, train, predict, val

    def setup(self):
        test, train, predict, val = self.splitting(self.dataset)
        
        self.test = self.get_dataloader(test)
        self.predict = self.get_dataloader(predict)
        self.train = self.get_dataloader(train)
        self.val = self.get_dataloader(val)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
    #     # Used to clean-up when the run is finished
    #     ...

    def collate_filter_for_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def get_dataloader(
        self,
        dataset,
        batch_size=32,
        num_workers=2**4,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_filter_for_none,
    ):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
