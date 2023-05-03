# %%
from hydra.utils import instantiate

import albumentations as A
import hydra
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch.transforms import ToTensorV2


from pytorch_lightning import seed_everything

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # OmegaConf.resolve(cfg)
    # cli = LightningCLI(Bio_VAE)
    seed_everything(cfg.seed)
    # args = cfg.timm
    transform = A.from_dict(OmegaConf.to_container(cfg.albumentations, resolve=True))
    transform = A.Compose(
    [
        A.RandomCrop(width=64, height=64, p=1.0),
        A.ToFloat(max_value=1, p=1.0),
        ToTensorV2(),
    ]
)
    # dataset = instantiate(cfg.dataset)
    pythae = instantiate(cfg.pythae)
    from pythae.models import VQVAE, VQVAEConfig, VAE, VAEConfig
    model_config = VAEConfig(
            latent_dim=64,
            input_dim=(3, 64, 64),
        )
    from bio_vae.models.bolts import ResNet18VAEEncoder,ResNet18VAEDecoder
    encoder = ResNet18VAEEncoder(model_config)
    decoder = ResNet18VAEDecoder(model_config)

    model = VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder,
    )
    # model = instantiate(cfg.model)
    dataloader = instantiate(cfg.dataloader, transform=transform)
    dataloader.setup()
    test_data = dataloader.get_dataset()[0].unsqueeze(dim=0)
    test = pythae({"data":test_data[:,:,:64,:64] })
    # model = instantiate(cfg.model)
    # pythae = instantiate(cfg.pythae)
    lightning = instantiate(cfg.lightning,model=model)
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
