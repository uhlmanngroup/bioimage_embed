# %%
from hydra.utils import instantiate

import albumentations as A
import hydra
from omegaconf import DictConfig, OmegaConf


from pytorch_lightning import seed_everything


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # OmegaConf.resolve(cfg)
    # cli = LightningCLI(Bio_VAE)
    seed_everything(cfg.seed)
    # args = cfg.timm
    transform = A.from_dict(OmegaConf.to_container(cfg.albumentations, resolve=True))
    # dataset = instantiate(cfg.dataset)
    # pythae = instantiate(cfg.pythae)
    # model = instantiate(cfg.model)
    dataloader = instantiate(cfg.dataloader, transform=transform)
    dataloader.setup()
    # model = instantiate(cfg.model)
    # pythae = instantiate(cfg.pythae)
    lightning = instantiate(cfg.lightning)
    # logger = instantiate(cfg.logger)
    # checkpoint_callback = instantiate(cfg.checkpoints)
    trainer = instantiate(cfg.trainer)

    try:
        trainer.fit(
            lightning,
            datamodule=dataloader,
            ckpt_path=f"{cfg.checkpoints.dirpath}/last.ckpt",
        )
    except:
        trainer.fit(lightning, datamodule=dataloader)


if __name__ == "__main__":
    main()
