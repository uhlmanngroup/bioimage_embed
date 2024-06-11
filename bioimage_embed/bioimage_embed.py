import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import logging
from .config import Config
from bioimage_embed.lightning import DataModule
from .lightning.torch import LitAutoEncoderTorch
from hydra.utils import instantiate
from torch.autograd import Variable
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf 
from . import utils

import torch.optim as optim
logging.basicConfig(level=logging.INFO)


class BioImageEmbed:
    cfg: Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.icfg = instantiate(cfg)
        self.ocfg = OmegaConf.structured(self.cfg)
        OmegaConf.resolve(self.ocfg)

        self.setup()

    def checkpoint_hash(self):
        recipe = self.icfg.recipe
        self.checkpoint_dir = utils.hashing_fn(recipe)

    def setup(self):
        np.random.seed(self.icfg.recipe.seed)
        seed_everything(self.icfg.recipe.seed)

        self.make_dirs()
        self.icfg.lit_model.model.eval()

    def model_check(self):
        dataloader = self.icfg.dataloader
        dataloader_0 = next(iter(dataloader.train_dataloader()))
        output = self.icfg.lit_model(dataloader_0)
        logging.info("Model Check Passed")

    def trainer_check(self):
        trainer = instantiate(self.ocfg.trainer, fast_dev_run=True)
        trainer.fit(self.icfg.lit_model, self.icfg.dataloader)
        logging.info("Trainer Check Passed")

    def make_dirs(self):
        for path in (self.icfg.paths).values():
            os.makedirs(path, exist_ok=True)

    def find_checkpoint(self):
        for callback in self.icfg.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.last_model_path

    def train(self, resume: bool = True):
        try:
            self.icfg.trainer.fit(
                self.icfg.lit_model,
                datamodule=self.icfg.dataloader,
                ckpt_path="last",
            )
        except:
            self.icfg.trainer.fit(self.icfg.lit_model, datamodule=self.icfg.dataloader)

    def validate(self):
        validation = self.icfg.trainer.validate(
            self.lit_model, datamodule=self.dataloader
        )

    def test(self):
        testing = self.icfg.trainer.test(
            self.icfg.lit_model, datamodule=self.icfg.dataloader
        )

    def __call__(self, x):
        return self.icfg.lit_model(x)

    def infer(self):
        dataloader = DataModule(
            self.icfg.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.icfg.receipe.num_workers,
        )
        predictions = self.icfg.trainer.predict(
            self.icfg.lit_model, datamodule=dataloader
        )
        return predictions

    def export(self):
        example_input = Variable(torch.rand(1, *self.cfg.recipe.input_dim))
        self.icfg.lit_model.to_onnx(
            f"{self.icfg.uuid}.onnx",
            example_input,
            export_params=True,
            verbose=True,
        )
    
    def check(self):
        self.model_check()
        self.trainer_check()
