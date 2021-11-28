#  %%
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import glob
# Note - you must have torchvision installed for this example
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import torchvision
# class VAEDataset(Dataset):
#     def __init__(self,
#                 data_dir,
#                 batch_size= 32):
#         super().__init__()    
    
#     def __getitem__(self, index):

path = os.path.join(os.path.expanduser("~"),
                    "data-science-bowl-2018/stage1_train/*/masks/*.png");path

#  %%

window_size = 128-32
class cropCentroid(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return self.crop_centroid(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"
    
    def crop_centroid(self,image,size):
        np_image = np.array(image)
        im_height,im_width = np_image.shape

        properties = regionprops(np_image.astype(int),
                                np_image.astype(int))
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        top = int(center_of_mass[0]-size/2)
        left = int(center_of_mass[1]-size/2)
        height,width = size,size
        # TODO find bad croppings
        # if ((top <= 0)  or (top+height >= im_height)  or (left <= 0) or (left+width >= 0) ):
            # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        return crop(image,top,left,height,width)

transform = transforms.Compose(
    [
        cropCentroid(window_size),
        transforms.ToTensor(),
        # transforms.Normalize(0, 1),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.RandomCrop((512, 512)),
        # transforms.ConvertImageDtype(torch.bool)

    ]
)

class DSB2018(Dataset):
    def __init__(self, path_glob, transform=None):
        self.image_paths = glob.glob(path_glob,recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        # if self.transform is not None:
        x = self.transform(x)

        return x

    def __len__(self):
        return len(self.image_paths)

train_dataset_glob=os.path.join(os.path.expanduser("~"),
                    "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
                    # "data-science-bowl-2018/stage1_test/*/masks/*.png")
train_dataset=DSB2018(train_dataset_glob,transform=transform)
train_dataset[0]
#  %%


batch_size=32

dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

#  %%

# fig,ax = plt.subplots(10,10)
# for i,ax in enumerate(ax.flat):
#     ax.imshow(transform(train_dataset[i]).reshape(window_size,window_size))

#  %%
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
    
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 1, 3, 1)

        self.fc1 = nn.AdaptiveMaxPool2d((25))
        self.fc2 = nn.AdaptiveMaxPool2d((12,12))


        # self.conv4 = self.contract_block(128, 256, 3, 1)

        # self.upconv4 = self.contract_block(256, 128, 3, 1)
        self.upconv3 = self.expand_block(1, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 1, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 1, out_channels, 3, 1)

        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3,self.fc1)
        self.decoder = nn.Sequential(self.fc2,self.upconv3, self.upconv2, self.upconv1)

    # Call is essentially the same as running "forward"
    def __call__(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)
        return self.forward(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

model = AutoEncoder(1,1)
img = train_dataset[0].unsqueeze(0)
y = model(img)
z = model.encoder(img)
print(f"img_dims:{img.shape} y:_dims:{y.shape} z:_dims:{z.shape}")
#  %%
# TODO better loss is needed, outshapes are currently not always full
# loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCEWithLogitsLoss()

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size=1, learning_rate=1e-3):
        super().__init__()
        self.encoder = AutoEncoder(batch_size,1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = torch.nn.BCELoss()
        

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.encoder(inputs)
        loss = self.loss_fn(output,inputs)
        self.log("train_loss", loss)
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("Loss/train", loss, batch_idx)
        # torchvision.utils.make_grid(output)
        tensorboard.add_image("input", torchvision.utils.make_grid(inputs), batch_idx)
        tensorboard.add_image("output", torchvision.utils.make_grid(torch.sigmoid(output)), batch_idx)

        # tensorboard.add_image("input", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        # tensorboard.add_image("output", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        return loss

#  %%
tb_logger = pl_loggers.TensorBoardLogger("runs/")
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(every_n_train_steps=100)

trainer = pl.Trainer(
    logger=tb_logger,
    gpus=1,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
)#.from_argparse_args(args)
#  %%
# 
# if __name__ = main:
# 
#  %%
import sys
model = LitAutoEncoder()
trainer.fit(model, dataloader)
# loss_function = torch.nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters())
# epochs = 20
# outputs = []
# losses = []
# for epoch in range(epochs):
#     for image in train_dataloader:
        
#         # Reshaping the image to (-1, 784)
#     #   image = image.reshape(-1, 28*28)
        
#         # Output of Autoencoder
#         reconstructed = model(image)
        
#         # Calculating the loss function
#         loss = loss_function(reconstructed, image)
        
#         # The gradients are set to zero,
#         # the the gradient is computed and stored.
#         # .step() performs parameter update
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Storing the losses in a list for plotting
#         losses.append(loss)
#     outputs.append((epochs, image, reconstructed))
  
# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
  
# # Plotting the last 100 values
# plt.plot(losses[-100:])
# from torchviz import make_dot
# make_dot(y,params=dict(model.named_parameters()))
#  %%



# class MaskAE(pl.LightningModule):
#         def __init__(self):
#         super(MaskAE, self).__init__()
#         self.batch_size = 4
#         self.learning_rate = 1e-3
# #         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
# #         self.net = UNet(num_classes = 19, bilinear = False)
# #         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             cropCentroid()
#         ])
#         self.trainset = None
#         self.testset = None
        
#     def forward(self, x):
#         return self.net(x)
    
#     def training_step(self, batch, batch_nb) :
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self.forward(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index = 250)
# #         print(loss.shape)
#         return {'loss' : loss_val}
    
#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
#         return [opt], [sch]
    
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)
    
#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size = 1, shuffle = True)


#  %%


# #  %%
# class SegModel(pl.LightningModule):
#     def __init__(self):
#         super(SegModel, self).__init__()
#         self.batch_size = 4
#         self.learning_rate = 1e-3
# #         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
# #         self.net = UNet(num_classes = 19, bilinear = False)
# #         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             cropCentroid()
#         ])
#         self.trainset = None
#         self.testset = None
        
#     def forward(self, x):
#         return self.net(x)
    
#     def training_step(self, batch, batch_nb) :
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self.forward(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index = 250)
# #         print(loss.shape)
#         return {'loss' : loss_val}
    
#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
#         return [opt], [sch]
    
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)
    
#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size = 1, shuffle = True)


#  %%
# test_dataloader_dir="data/stage1_test"

# val_dataloader_dir=
# test_dataloader_dir=
# predict_dataloader_dir=
