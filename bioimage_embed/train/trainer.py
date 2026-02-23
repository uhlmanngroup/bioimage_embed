"""
HuggingFace Trainer wrapper for bioimage_embed.

This module provides a thin wrapper around transformers.Trainer
for training bioimage embedding models.
"""

from typing import Optional, Dict, Any, Callable
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback


class BioimageTrainer(Trainer):
    """
    Thin wrapper around HuggingFace Trainer for bioimage embedding models.
    
    Reuses HF TrainingArguments and callbacks infrastructure.
    Minimal custom logic - delegates to parent Trainer wherever possible.
    
    Args:
        model: PyTorch model to train
        args: HuggingFace TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        compute_metrics: Function to compute metrics (optional)
        callbacks: List of TrainerCallback instances (optional)
        **kwargs: Additional arguments passed to Trainer
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )
    
    def compute_loss(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ):
        """
        Compute loss for bioimage embedding models.
        
        Override this method in subclasses for custom loss computation.
        Default behavior: assume model returns a dict with 'loss' key.
        
        Args:
            model: The model
            inputs: Input dict
            return_outputs: Whether to return outputs along with loss
            
        Returns:
            loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        outputs = model(**inputs)
        
        # Handle different output formats
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            raise ValueError(
                "Model output must have a 'loss' attribute or be a dict with 'loss' key"
            )
        
        return (loss, outputs) if return_outputs else loss


def create_training_args(
    output_dir: str = "./outputs",
    num_train_epochs: int = 50,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.001,
    warmup_epochs: int = 5,
    logging_steps: int = 100,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    **kwargs
) -> TrainingArguments:
    """
    Helper function to create TrainingArguments with sensible defaults.
    
    Args:
        output_dir: Output directory for checkpoints and logs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs
        logging_steps: Log every N steps
        eval_strategy: Evaluation strategy
        save_strategy: Save strategy
        **kwargs: Additional arguments for TrainingArguments
        
    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_epochs,  # Note: HF uses steps not epochs
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        **kwargs
    )
