import os
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
from .config import Config
from .lightning import DataModule
from hydra.utils import instantiate
from torch.autograd import Variable
from pytorch_lightning import seed_everything
from . import utils, config

logging.basicConfig(level=logging.INFO)


class BioImageEmbed:
    cfg: Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.icfg = instantiate(cfg)
        self.ocfg = self.resolve()
        self.setup()

    def resolve(self):
        """
        Resolves the config using omegaconf,
        without the flag this will crash with mixed types
        """
        self.ocfg = config.resolve_config(self.cfg)
        return self.ocfg

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

        # dataloader_0 = next(iter(dataloader.predict_dataloader()))
        # TODO smarter way to do this
        data = dataloader.dataset[0][0].unsqueeze(0)
        assert self.icfg.lit_model(data)
        logging.info("Model Check Passed")
        return self

    def trainer_check(self):
        trainer = instantiate(self.ocfg.trainer, fast_dev_run=True, accelerator="cpu")
        trainer.test(self.icfg.lit_model, self.icfg.dataloader)
        logging.info("Trainer Check Passed")
        return self

    def trainer_check_fit(self):
        trainer = instantiate(self.ocfg.trainer, fast_dev_run=True)
        trainer.fit(self.icfg.lit_model, self.icfg.dataloader)
        logging.info("Trainer Check Passed")
        return self

    # TODO fix resume
    def make_dirs(self):
        for path in (self.icfg.paths).values():
            os.makedirs(path, exist_ok=True)

    def find_checkpoint(self):
        for callback in self.icfg.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.last_model_path

    def train(self, resume: bool = True):
        # FYI
        # last_checkpoint = chkpt_callbacks.last_model_path
        # best_checkpoint_path = chkpt_callbacks.best_model_path

        # TODO add tests for checkpointing (properply)
        return self._train()

    def train_resume(self):
        return self.train("last")

    def _train(self, ckpt_path=None):
        self.icfg.trainer.fit(
            self.icfg.lit_model,
            datamodule=self.icfg.dataloader,
            ckpt_path=ckpt_path,
        )
        return self

    def validate(self):
        assert self.icfg.trainer.validate(
            self.icfg.lit_model, datamodule=self.icfg.dataloader
        )
        return self

    def test(self):
        assert self.icfg.trainer.test(
            self.icfg.lit_model, datamodule=self.icfg.dataloader
        )
        return self

    def __call__(self, x, ckpt_path="best"):
        dataloader = DataModule(
            x,
            batch_size=1,
            shuffle=False,
            num_workers=self.icfg.receipe.num_workers,
            # TODO: Add transform here if batch_size > 1 assuming averaging?
            # Transform is commented here to avoid augmentations in real data
            # HOWEVER, applying a the transform multiple times and averaging the results might produce better latent embeddings
            # transform=self.cfg.transform,
        )
        # dataloader.setup()
        return self.icfg.trainer.predict(
            self.icfg.lit_model,
            datamodule=dataloader,
            ckpt_path=ckpt_path,
        )

        # return self.icfg.lit_model(x)

    def forward(self, x):
        self.icfg.lit_model(x)

    def infer(self, ckpt_path="best"):
        return self(self.icfg.dataset, ckpt_path)

        # dataloader = DataModule(

        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=self.icfg.receipe.num_workers,
        #     # TODO: Add transform here if batch_size > 1 assuming averaging?
        #     # Transform is commented here to avoid augmentations in real data
        #     # HOWEVER, applying a the transform multiple times and averaging the results might produce better latent embeddings
        #     # transform=self.cfg.transform,
        # )
        # # dataloader.setup()
        # return self.icfg.trainer.predict(
        #     self.icfg.lit_model,
        #     datamodule=dataloader,
        #     ckpt_path=ckpt_path,
        # )

    def export(self):
        # TODO export best model to onnx
        data = torch.rand(1, *self.cfg.recipe.input_dim)
        assert self(data)
        example_input = Variable()
        self.icfg.lit_model.to_onnx(
            f"{self.icfg.uuid}.onnx",
            example_input,
            export_params=True,
            # opset_version=11,
            verbose=True,
        )

    def check(self):
        self.model_check()
        self.trainer_check()
        self.trainer_check_fit()
