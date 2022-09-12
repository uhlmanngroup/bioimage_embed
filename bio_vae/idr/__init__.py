import torch
import math
# from idr import connection
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
from PIL import Image
from torchvision import transforms



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class IDRDataSet(Dataset):
    def __init__(self, file_list, transform=None):
        super(IDRDataSet).__init__()
        self.transform = transform
        self.mode = None
        # if file_list == None:
            # self.mode = "http"
        # assert end > start, "this example code only works with end >= start"
        # self.start = start
        # self.end = end
        # self.conn = connection("idr.openmicroscopy.org")
        # self.loader = default_loader
        # self.conn = connection("idr.openmicroscopy.org")

    def __getitem__(self, index):
        # conn = self.conn
        # if (self.mode=="http"):
            # image = self.get_idr_image(index)
        # else:
        image = self.get_nfs_idr_image(index)

        if image == None:
            return None
            return np.full([2048, 2048], np.nan)
        image = make_size(image, size=[2048, 2048])
        self.index = index
        image_np = np.array(image)
        # print(f"Found image at {index}")
        if self.transform:
            trans = self.transform(image_np)
            return (trans, trans)
        return (image_np, image_np)

    def __len__(self):
        # return 100
        if (self.mode=="http"):
            return 100000
        else:
            return len(self.file_list)

    # Deprecated: 
    # def get_idr_image(self, imageid=171499):
    #     try:
    #         conn = connection("idr.openmicroscopy.org", verbose=0)
    #         # print(f"Get image {str(imageid)}")
    #         image_wrapped = conn.getObject("Image", imageid)
    #         if image_wrapped == None:
    #             return None
    #     except:
    #         return None
    #     # image_wrapped.getPrimaryPixels()
    #     try:
    #         image_plane = image_wrapped.getPrimaryPixels().getPlane(0)
    #         norm_image_plane = ((image_plane / image_plane.max()) * 255).astype(np.int8)
    #         pil_image = Image.fromarray(norm_image_plane, "L")
    #         # if image_plane == None:
    #         return pil_image
    #     except:
    #         return None
        
    def get_nfs_idr_image(self,file_list,index=0):
        file_path = file_list[index] 
        Image.open(file_path)
    
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


# RETURN_NONE = True
# np.full([2048, 2048], np.nan)
# def get_idr_image(imageid=171499):
#     # conn = connection("idr.openmicroscopy.org")
#     image_wrapped = conn.getObject("Image", imageid)
#     if image_wrapped == None:
#         return None
#     # image_wrapped.getPrimaryPixels()
#     try:
#         image_plane = image_wrapped.getPrimaryPixels().getPlane(0)
#         norm_image_plane = ((image_plane/image_plane.max())*255).astype(np.int8)
#         pil_image = Image.fromarray(norm_image_plane,"L")
#         image = make_size(pil_image, size=[2048, 2048])
#         # if image_plane == None:
#         return image
#     except:
#         return None


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
