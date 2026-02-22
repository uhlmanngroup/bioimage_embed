"""
Example training script using BioimageTrainer.

This demonstrates how to use the HuggingFace Trainer wrapper
for training bioimage embedding models.
"""

import torch
from torch.utils.data import Dataset
from bioimage_embed.train import BioimageTrainer, create_training_args


class DummyBioimageDataset(Dataset):
    """Example dataset for demonstration."""
    
    def __init__(self, num_samples: int = 100, image_size: int = 64):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return dummy data - replace with actual data loading
        image = torch.randn(3, self.image_size, self.image_size)
        return {"pixel_values": image, "labels": idx % 10}


class DummyModel(torch.nn.Module):
    """Example model for demonstration."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, latent_dim)
        )
        self.decoder = torch.nn.Linear(latent_dim, input_channels * 64 * 64)
    
    def forward(self, pixel_values, **kwargs):
        z = self.encoder(pixel_values)
        recon = self.decoder(z).view(-1, 3, 64, 64)
        
        # Compute reconstruction loss
        loss = torch.nn.functional.mse_loss(recon, pixel_values)
        
        return {"loss": loss, "latent": z, "reconstruction": recon}


def main():
    """Example training workflow."""
    
    # Create datasets
    train_dataset = DummyBioimageDataset(num_samples=1000)
    eval_dataset = DummyBioimageDataset(num_samples=200)
    
    # Create model
    model = DummyModel()
    
    # Create training arguments
    training_args = create_training_args(
        output_dir="./outputs",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.001,
        warmup_epochs=2,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Enable mixed precision training
        fp16=torch.cuda.is_available(),
    )
    
    # Create trainer
    trainer = BioimageTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")
    
    # Save model
    trainer.save_model("./outputs/final_model")


if __name__ == "__main__":
    main()
