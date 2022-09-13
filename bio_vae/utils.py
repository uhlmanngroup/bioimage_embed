import torch

def collate_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_test_image(dataset):
    test_img = next(item for item in dataset if item is not None)
    return test_img.unsqueeze(0)