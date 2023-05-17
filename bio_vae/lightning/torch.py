import torchvision
import torch
import pytorch_lightning as pl
import pythae
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
import timm

class LitAutoEncoderTorch(pl.LightningModule):
    args = argparse.Namespace(
        opt="adamw",
        weight_decay=0.001,
        momentum=0.9,
        sched="cosine",
        epochs=50,
        lr=0.1,
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
    )
    def __init__(self, model,args):
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        # self.model.train()
        
    def forward(self, x):
        return self.model({"data":x}
                          )
    def get_results(self, batch):
        # if self.PYTHAE_FLAG:
        return self.model.forward({"data": batch})
        # return self.model.forward(batch)
        
    def training_step(self, batch, batch_idx):
        results = self.get_results(batch)
        self.model.train()
        x = {"data":batch}
        model_output = self.model(x,epoch=batch_idx)
        # loss = self.model.training_step(x)
        self.loss = model_output.loss

        # self.log("train_loss", self.loss)
        # self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Loss/train", self.loss, batch_idx)

        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(batch), batch_idx
        )
        # if self.PYTHAE_FLAG:
        self.logger.experiment.add_image(
            "output",
            torchvision.utils.make_grid(results.recon_x),
            batch_idx,
        )
        return self.loss

    def lr_scheduler_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # Implement your own logic for updating the lr scheduler
        # This method will be called at each training step
        # Update the lr scheduler based on the provided arguments
        # You can access the lr scheduler using `self.lr_schedulers()`

        # Example:
        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.step()
            
    def configure_optimizers(self):
        self.optimizer = optim.create_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = scheduler.create_scheduler(
            self.args, self.optimizer
        )[0]
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }
    # def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
    
# class LitAutoEncoderTorch(pl.LightningModule):
#     # lr_scheduler = None
#     # lr_scheduler_params = {
#     #     "scheduler":"cosine",
#     #     "epochs":50,
#     #     "lr":0.1,
#     #     "min_lr":1e-6,
#     #     "t_initial":10,
#     #     "t_mul":2,
#     #     "lr_min":None,
#     #     "decay_rate":0.1,
#     #     "warmup_lr_init":1e-6,
#     #     "warmup_epochs":5,
#     #     "cycle_limit":None,
#     #     "t_in_epochs":False,
#     #     "noisy":False,
#     #     "noise_std":0.1,
#     #     "noise_pct":0.67,
#     #     "noise_seed":None
#     # }
#     # optimizer = None
#     # optimizer_params = {
#     #     "opt": "admaw",
#     #     "lr": 0.1,
#     #     "weight_decay": 0.001,
#     #     "momentum": 0.9,
#     # }

#     args = argparse.Namespace(
#         opt="admaw",
#         weight_decay=0.001,
#         momentum=0.9,
#         scheduler="cosine",
#         epochs=50,
#         lr=0.1,
#         min_lr=1e-6,
#         t_initial=10,
#         t_mul=2,
#         lr_min=None,
#         decay_rate=0.1,
#         warmup_lr_init=1e-6,
#         warmup_epochs=5,
#         cycle_limit=None,
#         t_in_epochs=False,
#         noisy=False,
#         noise_std=0.1,
#         noise_pct=0.67,
#         noise_seed=None,
#     )

#     def __init__(
#         self,
#         model,
#         args=None,
#         # batch_size=1,
#     ):
#         super().__init__()

#         # self.batch_size = batch_size

#         # self.loss_fn = torch.nn.MSELoss()

#         self.model = model

#         # self.PYTHAE_FLAG = issubclass(self.model.__class__, pythae.models.BaseAE)

#         # if self.PYTHAE_FLAG:
#         # self.pythae_init()
#         self.model = self.model.to(self.device)
#         # TODO: pythonic way to do this
#         if args is not None:
#             self.args = args

#     # def pythae_init(self):
#         # self.model = self.model.to(self.device)
#         # self.model.train()

#     def forward(self, x):
#         # if self.PYTHAE_FLAG:
#         return self.model.forward({"data": x})["recon_x"]
#         # return self.model.forward(x)

#     def recon(self, x):
#         # if self.PYTHAE_FLAG:
#         return self.forward({"data": x})
#         # return self.model.recon(x)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#         # return [optimizer], [scheduler]
#         return optimizer
    # def configure_optimizers(self):
    #     self.optimizer = optim.create_optimizer(self.args, self.model.parameters())
    #     self.lr_scheduler = scheduler.create_scheduler(
    #         self.args, self.optimizer
    #     )[0]
    #     return {
    #         'optimizer': self.optimizer,
    #         'lr_scheduler': self.lr_scheduler
    #     }
        
#     # def lr_scheduler_step(self, optimizer, epoch, optimizer_idx):
#     #     # Check if the current scheduler is the custom scheduler
#     #     if isinstance(self.lr_scheduler, timm.scheduler.CosineLRScheduler):
#     #         # Update the learning rate using the custom scheduler
#     #         self.lr_scheduler.step(epoch)
#     #     else:
#     #         # Update the learning rate using the default PyTorch Lightning logic
#     #         super().lr_scheduler_step(optimizer, epoch, optimizer_idx)
            
#     # def predict_step(self, batch, batch_idx, dataloader_idx=0):
#     # return self.recon(batch)

#     def get_loss(self, batch,batch_idx):
#         return self.model({"data": batch},epoch=batch_idx).loss

#     def get_results(self, batch):
#         # if self.PYTHAE_FLAG:
#         return self.model.forward({"data": batch})
#         # return self.model.forward(batch)

#     def test_step(self, batch, batch_idx):
#         test_loss = self.get_loss(batch,batch_idx)
#         self.log("test_loss", test_loss, on_epoch=True)
#         return test_loss

#     def validation_step(self, batch, batch_idx):
#         val_loss = self.get_loss(batch,batch_idx)
#         self.log("val_loss", val_loss, on_epoch=True)
#         return val_loss

#     def training_step(self, batch, batch_idx):
#         self.model.train()
#         loss = self.get_loss(batch,batch_idx)
#         # loss = self.model({"data": batch}).loss
#         # self.model.update()
#         results = self.get_results(batch)

#         self.log("train_loss", loss)
#         # self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

#         self.logger.experiment.add_image(
#             "input", torchvision.utils.make_grid(batch), batch_idx
#         )
#         # if self.PYTHAE_FLAG:
#         self.logger.experiment.add_image(
#             "output",
#             torchvision.utils.make_grid(results["recon_x"]),
#             batch_idx,
#         )
#         # else:
#         #     self.logger.experiment.add_image(
#         #         "output",
#         #         torchvision.utils.make_grid(self.model.output_from_results(*results)),
#         #         batch_idx,
#         #     )
#         return loss

#     def get_model(self):
#         return self.model

#     @property
#     def num_training_steps(self) -> int:
#         return 100
