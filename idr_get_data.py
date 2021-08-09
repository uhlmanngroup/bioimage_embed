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

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

import numpy as np
import matplotlib.pyplot as plt
# import NPC8nodes
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import convolve
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py
# https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html

index_num = 4684936
conn = connection("idr.openmicroscopy.org")
class IDRDataSet(VisionDataset):
    def __init__(self,transform=None):
        super(IDRDataSet).__init__()
        self.transform = transform
        # assert end > start, "this example code only works with end >= start"
        # self.start = start
        # self.end = end
        # self.conn = connection("idr.openmicroscopy.org")
        # self.loader = default_loader
        self.conn = connection("idr.openmicroscopy.org")

    def __getitem__(self, index):
        conn = self.conn
        image = self.get_idr_image(index,conn)
        if image == None:
            return None
            return np.full([2048,2048],np.nan)
        self.index = index
        image_np = np.array(image)
        print(f"Found image at {index}")
        if self.transform:
            trans = self.transform(image_np)
            return (trans,trans)
        return (image_np, image_np)

    def __len__(self):
        return 10000000

    def get_idr_image(self,imageid=171499,conn=None):
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

# %%

from torchvision import transforms
# transforms = ToTensor()  # we need this to convert PIL images to Tensor
shuffle = True
bs = 5
transform = torch.nn.Sequential()

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.RandomCrop((512,512))
 ])

dataset = IDRDataSet(transform=transform)
import nonechucks as nc

# dataset = nc.SafeDataset(dataset)
# dataloader = nc.SafeDataLoader(dataset, batch_size=bs, shuffle=shuffle)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

                                           
dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,collate_fn=collate_fn,num_workers=0)

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
        break
    break
        # plt.show()

dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,collate_fn=collate_fn,num_workers=0)
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
plt.imshow(dataset.get_idr_image(4684936))


#  %%
import torchvision
from torch.utils.data import Dataset, DataLoader


#  Unet structure outright stolen from the internet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 1, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 1, out_channels, 3, 1)

    # Call is essentially the same as running "forward"
    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(upconv3)
        upconv1 = self.upconv1(upconv2)

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand

class Net(nn.Module):
    def __init__(self, batch_size=16, n_class=1):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # self.conv1 = nn.Conv2d(1, 32, 3, 3)
        # self.conv2 = nn.Conv2d(32, 64, 3, 3)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        # self.conv1 = nn.Conv2d(1, 1, 1, 1)
        # self.dp = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(1, 1, 1, 1)
        self.dp = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        self.conv3 = nn.Conv2d(1, 1, 5, 5)
        self.fc2 = nn.AdaptiveAvgPool2d((100, 100))
        self.fc1 = nn.AdaptiveAvgPool2d((16, 16))
        self.conv4 = nn.Conv2d(1, 1, 1, 1)
        self.unet = UNet(1, 1)

    def forward(self, x):
        # x = x.double()
        x = self.unet(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.dp(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        # # # x = x.view((1,-1))
        # x = self.fc1(x)

        # x = F.relu(x)
        # x = self.fc1(x)

        # x = F.relu(x)
        # x = self.unet(x)

        # x = F.relu(x)
        # # x = x.view((1,-1))
        # x = self.fc1(x)
        # x = self.unet(x)
        # x = F.relu(x)
        # x = x.view((self.batch_size,1,14,14))
        # x = self.conv4(x)
        # x = torch.sigmoid(x)
        # x = F.relu(x)
        output = x
        return output


#  Test Unet is working correctly by feeding in random data with the correct dimensionality
#  One channel in and out
#  You can also use channels to encode time which is useful
unet = UNet(1, 1)
x = torch.randn(16, 1, 128, 128)
unet(x).shape

#  Testing full model
model = Net(1, 1)
x = torch.randn(16, 1, 128, 128)
model(x).shape


# %% Define model and push to gpu (if availiable)
model = Net(1, 1).to(device)
#  Test
x = torch.randn(16, 1, 128, 128).to(device)

output = model(x)
output.shape


samples = 10000 # Number of samples per epoch, any number will do
# dataset = NPCDataSet(samples) # Generate callable datatset (it's an iterator)
batch_size = 32 # Publications claim that smaller is better for batch
# model = Net(batch_size).to(device)#
# %% Define model and push to gpu (if availiable)
from torch.utils.tensorboard import SummaryWriter

model = Net(batch_size, 1).to(device)

loader = dataloader
i = 0 
# Make a run output folder for convenience
writer = SummaryWriter()
# %%
# Always set a manual seed else your runs will be incomparable
torch.manual_seed(42)

batch_size = 16
lr = 1e-5 # Low learning rates converge better
log_interval = 10 # Logging interval
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam does better than SGD
optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Adam does better than SGD

model.train() #Set model to training mode (probably deprecated?)
# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.MSELoss() # MSE is fine for this

# loss_fn = nn.BCELoss()
# loss_fn = nn.BCEWithLogitsLoss()


for epoch in range(0, epochs):
    for batch_idx, (inputs, outputs) in enumerate(loader):
        # data, target = data.to(device), target.to(device)
        # final_position = get_npc()
        # npc_image = npc_image_from_position(final_position)
        # distogram = euclidean_distances(final_position)

        # inputs = npc_image
        # outputs = distogram
        # inputs.shape[-2:0]
        # data = torch.tensor(inputs).view([-1,1,*(inputs.shape)]).float().to(device)
        # target = torch.tensor(outputs).view([-1,1,*(outputs.shape)]).float().to(device)
        # data = inputs.view([-1, 1, 128, 128]).float().to(device)
        # target = outputs.view([-1, 1, 16, 16]).float().to(device)

        data = inputs
        target = outputs

        optimizer.zero_grad()

        for dwell in range(0, 100):
            output = model(data)
            loss = loss_fn(output, target)
            # loss = torch.nn.functional.nll_loss( output, target)
            loss.backward()
            optimizer.step()
        i += 1
        writer.add_scalar("Loss/train", loss, i)
        writer.add_image("output", output[0])
        writer.add_image("target", target[0])
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {str(epoch)} {str(batch_idx)} \t Loss: {str(loss.item())}"
            )
# %% 
torch.save(model.state_dict(), "model")
#  %%
plt.imshow(output.cpu().detach()[0, 0, :, :])
#  %%
plt.imshow(target.cpu().detach()[0, 0, :, :])

#  %%

plt.imshow(inputs.cpu().detach()[0, :, :])


# 
# # %%
# import torch
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST
# import os

# # https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )

#     def forward(self, x):
# # %%
# import torch
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST
# import os

# # https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3


# model = autoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)
# total_loss = 0
# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss}')
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, f'./dc_img/image_{str(epoch)}.png')

# torch.save(model.state_dict(), './conv_autoencoder.pth')
#         x = self.decoder(x)
#         return x

# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3


# model = autoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)
# total_loss = 0
# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss}')
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, f'./dc_img/image_{str(epoch)}.png')

# torch.save(model.state_dict(), './conv_autoencoder.pth')