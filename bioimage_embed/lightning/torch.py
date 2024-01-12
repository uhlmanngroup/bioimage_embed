import torchvision
import torch
import pytorch_lightning as pl
import pythae
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
import timm
from pythae.models.base.base_utils import ModelOutput
import torch.nn.functional as F
import wandb


class LitAutoEncoderTorch(pl.LightningModule):
    args = argparse.Namespace(
        opt="adamw",
        weight_decay=0.001,
        momentum=0.9,
        sched="cosine",
        epochs=50,
        lr=1e-4,
        min_lr=1e-6,
        t_initial=10,
        t_mul=2,
        lr_min=None,
        decay_rate=0.1,
        warmup_lr=1e-6,
        warmup_lr_init=1e-6,
        warmup_epochs=5,
        cycle_limit=None,
        t_in_epochs=False,
        noisy=False,
        noise_std=0.1,
        noise_pct=0.67,
        noise_seed=None,
        cooldown_epochs=5,
        warmup_t=0,
    )

    def __init__(self, model, args=SimpleNamespace(), wandb=None):
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        # Flatten hparams
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        if args:
            self.args = SimpleNamespace(**{**vars(args), **vars(self.args)})

        self.save_hyperparameters(vars(self.args))
        self.wandb = wandb
        self.wandb.watch(model)
        self.train_epoch_loss = []
        self.val_epoch_loss = []


    def forward(self, batch):
        x = self.batch_to_tensor(batch)
        return ModelOutput(x=x, out=self.model(x))

    def get_results(self, batch):
        x = self.batch_to_tensor(batch)
        return self.model.forward(x)

    def batch_to_tensor(self, batch):
        return ModelOutput(data=batch)

    def embedding_from_output(self, model_output):
        return model_output.z.view(model_output.z.shape[0], -1)

    def get_model_output(self, x, batch_idx):
        model_output = self.model(x, epoch=batch_idx)
        loss = self.loss_function(model_output)

        return model_output, loss
        

    def training_step(self, batch, batch_idx):
        self.model.train()
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(
            x,
            batch_idx,
        )
   
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(x["data"]), batch_idx
        )

        # if self.PYTHAE_FLAG:
        self.logger.experiment.add_image(
            "output",
            torchvision.utils.make_grid(model_output.recon_x),
            batch_idx,
        )

        #log_dict = {"train_loss_step": loss}

        mse_loss = F.mse_loss(model_output["recon_x"], x["data"])

        self.log("mse_train_loss_step", mse_loss)

        log_dict = {"mse_train_loss_step": mse_loss}

        #self.wandb.log({"train_loss_step": loss})
        if batch_idx == 0 and len(self.train_epoch_loss) > 0:
            log_dict["train_epoch_loss"] = sum(self.train_epoch_loss) / len(self.train_epoch_loss)
            self.train_epoch_loss = []
        
        self.train_epoch_loss.append(loss)
        
        self.wandb.log(log_dict)

        return loss

    def loss_function(self, model_output, *args, **kwargs):
        return model_output.loss

    def validation_step(self, batch, batch_idx):
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(x, batch_idx)
        z = self.embedding_from_output(model_output)
        self.logger.experiment.add_embedding(
            z,
            label_img=x["data"],
            global_step=self.current_epoch,
            tag="z",
        )

        self.logger.experiment.add_scalar("Loss/val", loss, batch_idx)
        self.logger.experiment.add_image(
            "val",
            torchvision.utils.make_grid(model_output["recon_x"]),
            batch_idx,
        )

        # Log test metrics
        self.log("validation_loss", loss)

        #log_dict = {"val_loss_epoch": loss}

        #self.wandb.log(log_dict)
        mse_loss = F.mse_loss(model_output["recon_x"], x["data"])

        self.log("mse_validation_loss_step", mse_loss)

        log_dict = {"mse_validation_loss_step": mse_loss}

        self.wandb.log(log_dict)

        return loss


    def timm_optimizers(self, model):
        optimizer = optim.create_optimizer(self.args, model.parameters())
        lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        return optimizer, lr_scheduler

    def timm_to_lightning(self, optimizer, lr_scheduler):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # or 'epoch' for step vs epoch training, respectively
            },
        }

    def configure_optimizers(self):
        # optimizer = optim.create_optimizer(self.args, self.model.parameters())
        # lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        optimizer, lr_scheduler = self.timm_optimizers(self.model)
        return self.timm_to_lightning(optimizer, lr_scheduler)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch, metric=metric)


    def test_step(self, batch, batch_idx):
        x = self.batch_to_tensor(batch)
        model_output = self.model(x)  # Forward pass with the test batch

        # Optionally compute a loss or metric if relevant
        loss = self.loss_function(model_output)

        # Log test metrics
        self.log("test_loss", loss)

        mse_loss = F.mse_loss(model_output["recon_x"], x["data"])

        self.log("mse_test_loss_step", mse_loss)

        log_dict = {"mse_test_loss_step": mse_loss}

        #log_dict = {"test_loss_step": loss}

        self.wandb.log(log_dict)

        # Optionally you can add more logging, for example, visualizations:
        self.logger.experiment.add_image(
            "test_input",
            torchvision.utils.make_grid(x["data"]),
            batch_idx,
        )
        self.logger.experiment.add_image(
            "test_output",
            torchvision.utils.make_grid(model_output.recon_x),
            batch_idx,
        )

        # Return whatever data you need, for example, the loss
        return loss
