# %%
# import torchvision
import torch
import math
from idr import connection
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor
import numpy as np

from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
from PIL import Image


# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py
# https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html

index_num = 4684936
conn = connection("idr.openmicroscopy.org")
class IDRDataSet(VisionDataset):
    def __init__(self):
        super(IDRDataSet).__init__()
        # assert end > start, "this example code only works with end >= start"
        # self.start = start
        # self.end = end
        # self.conn = connection("idr.openmicroscopy.org")
        # self.loader = default_loader

    def __getitem__(self, index):
        image = get_idr_image(index)
        if image == None:
            return None
            return np.full([2048,2048],np.nan)
        self.index = index
        image_np = np.array(image)
        print(f"Found image at {index}")
        return (image_np, image_np)

    def __len__(self):
        return 10000000

    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:  # single-process data loading, return the full iterator
    #         iter_start = self.start
    #         iter_end = self.end
    #     else:  # in a worker process
    #         # split workload
    #         per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = self.start + worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, self.end)
    #     return iter(range(iter_start, iter_end))

RETURN_NONE = True
# np.full([2048, 2048], np.nan)
def get_idr_image(imageid=171499):
    # conn = connection("idr.openmicroscopy.org")
    image_wrapped = conn.getObject("Image", imageid)
    if image_wrapped == None:
        return None
    # image_wrapped.getPrimaryPixels()
    try:
        image_plane = image_wrapped.getPrimaryPixels().getPlane(0)
        norm_image_plane = ((image_plane/image_plane.max())*255).astype(np.int8)
        pil_image = Image.fromarray(norm_image_plane,"L")
        image = make_size(pil_image, size=[2048, 2048])
        # if image_plane == None:
        return image
    except:
        return None


def make_size(im, size=[2048, 2048]):
    if list(im.size) < list(size):
        image = pad_to(im, size)
    else:
        image = crop_to(im, size)
    return image


def pad_to(im, size=[2048, 2048]):

    left = int(size[0] / 2 - im.size[0] / 2)
    top = int(size[1] / 2 - im.size[1] / 2)

    image = Image.new(im.mode, size, 0)
    image.paste(im, (left, top))
    return image


def crop_to(im, size=[2048, 2048]):
    left = int(size[0] / 2 - im.size[0] / 2)
    upper = int(size[1] / 2 - im.size[1] / 2)
    right = left + size[0]
    lower = upper + size[1]
    image = im.crop((left, upper, right, lower))
    return image

    # Define a `worker_init_fn` that configures each dataset copy differently
    # def worker_init_fn(worker_id):
    #     worker_info = torch.utils.data.get_worker_info()
    #     dataset = worker_info.dataset  # the dataset copy in this worker process
    #     overall_start = dataset.start
    #     overall_end = dataset.end
    #     # configure the dataset to only process the split workload
    #     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    #     worker_id = worker_info.id
    #     dataset.start = overall_start + worker_id * per_worker
    #     dataset.end = min(dataset.start + per_worker, overall_end)

# %%
transforms = ToTensor()  # we need this to convert PIL images to Tensor
shuffle = True
bs = 5

dataset = IDRDataSet()
import nonechucks as nc

# dataset = nc.SafeDataset(dataset)
# dataloader = nc.SafeDataLoader(dataset, batch_size=bs, shuffle=shuffle)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,collate_fn=collate_fn)
# dataloader = nc.SafeDataset(dataloader)
# data = iter(dataloader)
# a, b = data
# conn = connection(host="wss://idr.openmicroscopy.org/", user="public", password="public", port=443, verbose=0)

# conn = connection('idr.openmicroscopy.org')

# %%

for i, (rgb, gt) in enumerate(dataloader):
    print(f"batch {i+1}:")
    # some plots
    for i in range(bs):
        plt.figure(figsize=(10, 5))
        plt.subplot(221)
        plt.imshow(torch.squeeze(rgb[0]))
        plt.title(f"RGB img{i+1}")
        plt.subplot(222)
        plt.imshow(torch.squeeze(gt[0]))
        plt.title(f"GT img{i+1}")
        # plt.show()

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# def get_idr_image(imageid = 171499):
#     conn = connection('idr.openmicroscopy.org')
#     image_wrapped = conn.getObject("Image", imageid).getPrimaryPixels()
#     image_plane = image_wrapped.getPlane(0)
#     conn.close()
#     return image_plane

# def get_idr_image(imageid = 171499):
#     conn = connection('idr.openmicroscopy.org')
#     image_wrapped = conn.getObject("Image", imageid).getPrimaryPixels()
#     image_plane = image_wrapped.getPlane(0)
#     conn.close()
#     yield image_plane
plt.imshow(get_idr_image(4684936))

# %%
