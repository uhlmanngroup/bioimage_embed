# bioimage_embed.train - HuggingFace Trainer Integration

This module provides training utilities using HuggingFace's `transformers.Trainer`.

## Migration from PyTorch Lightning

This is a **thin wrapper** around HuggingFace Trainer, designed to replace PyTorch Lightning
with minimal changes to the codebase.

### Why HuggingFace Trainer?

- **Industry standard**: Used by most modern ML projects
- **Better maintained**: Active development and extensive community
- **Rich ecosystem**: Seamless integration with accelerate, datasets, hub
- **Simpler**: Less boilerplate than Lightning
- **Flexible**: Easy to extend without fighting the framework

### Quick Start

```python
from bioimage_embed.train import BioimageTrainer, create_training_args

# Create training arguments
training_args = create_training_args(
    output_dir="./outputs",
    num_train_epochs=50,
    learning_rate=1e-4,
)

# Create trainer
trainer = BioimageTrainer(
    model=your_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()
```

See `example_train.py` for a complete example.

### Design Principles

1. **Thin wrapper**: Reuse HF TrainingArguments, don't reinvent
2. **Minimal logic**: Delegate to parent Trainer wherever possible
3. **Easy migration**: Drop-in replacement for Lightning trainer

### Extending BioimageTrainer

For custom loss computation:

```python
class CustomBioimageTrainer(BioimageTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # Your custom loss logic here
        loss = custom_loss_fn(outputs, inputs)
        return (loss, outputs) if return_outputs else loss
```

### HuggingFace Resources

- [Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [Accelerate](https://huggingface.co/docs/accelerate)

### TODO

- [ ] Migrate existing Lightning models to HF-compatible format
- [ ] Add custom callbacks for bioimage-specific logging
- [ ] Integration with Weights & Biases via HF callbacks
- [ ] Multi-GPU training examples with accelerate
- [ ] Model export utilities (ONNX, TorchScript)
