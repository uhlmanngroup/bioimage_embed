#  %%
import sys
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro
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
from sklearn.manifold import MDS  
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import convolve,sobel 
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

path = os.path.join(os.path.expanduser("~"),
                    "data-science-bowl-2018/stage1_train/*/masks/*.png")
path

#  %%

window_size = 128-32
batch_size = 32
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

class DSB2018(Dataset):
    def __init__(self, path_glob, transform=None):
        self.image_paths = glob.glob(path_glob, recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        # if self.transform is not None:
        x = self.transform(x)

        return x

    def __len__(self):
        return len(self.image_paths)

class cropCentroid(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return self.crop_centroid(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def crop_centroid(self, image, size):
        np_image = np.array(image)
        im_height, im_width = np_image.shape

        properties = regionprops(np_image.astype(int),
                                 np_image.astype(int))
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        top = int(center_of_mass[0]-size/2)
        left = int(center_of_mass[1]-size/2)
        height, width = size, size
        # TODO find bad croppings
        # if ((top <= 0)  or (top+height >= im_height)  or (left <= 0) or (left+width >= 0) ):
        # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        return crop(image, top, left, height, width)

class cropCentroid(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return self.crop_centroid(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def crop_centroid(self, image, size):
        np_image = np.array(image)
        im_height, im_width = np_image.shape

        properties = regionprops(np_image.astype(int),
                                 np_image.astype(int))
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        top = int(center_of_mass[0]-size/2)
        left = int(center_of_mass[1]-size/2)
        height, width = size, size
        # TODO find bad croppings
        # if ((top <= 0)  or (top+height >= im_height)  or (left <= 0) or (left+width >= 0) ):
        # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        return crop(image, top, left, height, width)



class DistogramtoImage(torch.nn.Module):
    def __init__(self,size=256+128):
        super().__init__()
        self.size = size

    def forward(self, image):
        return(self.get_points_from_dist_C(image))

    def __repr__(self):
        return self.__class__.__name__
    
    def get_points_from_dist(self,image):
        return MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=0).fit_transform(image)

    def get_points_from_dist_C(self,tensor):
        dist_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0,:,:]
            dist_list.append(self.get_points_from_dist(image))
        return torch.tensor(np.array(dist_list))

class ImagetoDistogram(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # return self.get_distogram(img, self.size)
        return self.get_distogram_C(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def get_distogram_C(self,tensor,size):
        dist_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0,:,:]
            dist_list.append(self.get_distogram(image,size))
        return torch.tensor(np.array(dist_list))

    def get_distogram(self, image, size):
        distograms = []
        np_image = np.array(image)
        scaling = np.linalg.norm(np_image.shape)
        # for i in range(np_image_full.shape[0]):
        # np_image = np_image_full[i]
        # im_height, im_width = np_image.shape

        contour = find_contours(np_image)
        contour_x,contour_y = contour[0][:,0],contour[0][:,1]
        # plt.scatter(contour_x,contour_y)
        # plt.show()
        #  %%
        rho,phi = self.cart2pol(contour_x,contour_y)

        rho_interp = interp1d(np.linspace(0,1, len(rho)),rho, kind='cubic')(np.linspace(0,1, size))
        phi_interp = interp1d(np.linspace(0,1, len(phi)),phi, kind='cubic')(np.linspace(0,1, size))

        xii,yii = np.divide(self.pol2cart(rho_interp,phi_interp),scaling)
        # distograms.append(euclidean_distances(np.array([xii,yii]).T))
        return euclidean_distances(np.array([xii,yii]).T)

    def cart2pol(self,x, y):
        return(np.sqrt(x**2 + y**2), np.arctan2(y, x))

    def pol2cart(self,rho, phi):
        return(rho * np.cos(phi), rho * np.sin(phi))


transformer_crop = transforms.Compose(
    [
        # transforms.ToPILImage(),
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

transformer_dist = transforms.Compose(
    [
        transformer_crop,
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        ImagetoDistogram(window_size),
        # transforms.ToPILImage(),
        # transforms.RandomCrop((512, 512)),
        transforms.ConvertImageDtype(torch.float32)
    ]
)

transformer_image = transforms.Compose(
    [
        DistogramtoImage(window_size)
    ]
)

transformer = transformer_dist


train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")
train_dataset_dist = DSB2018(train_dataset_glob, transform=transformer_dist)
train_dataset_crop = DSB2018(train_dataset_glob, transform=transformer_crop)
plt.imshow(train_dataset_crop[0][0])
plt.show()
plt.imshow(train_dataset_dist[0][0])
plt.show()

train_dataset = train_dataset_dist
#  %%


transform_disttoimage = transforms.Compose([
    DistogramtoImage(window_size)
])


# dist = transformer_dist(train_dataset[0][0])
coords = transform_disttoimage(train_dataset_dist[0])
plt.scatter(coords[0][:,0],coords[0][:,1])

# print(out.shape)
# plt.imshow(transforms.ToPILImage()(transformer(train_dataset[0])))
# cell_image = transforms.ToPILImage()(train_dataset[0])
# distogram = ImagetoDistogram(512)(cell_image).astype(np.float32)
# plt.imshow(transforms.ToPILImage()(distogram))
# x = np.array(cell_image)
# out = convolve(x,fil, mode='constant')



# plt.imshow(cell_image)
# plt.show()
#  %%



# plt.plot(rho)
# plt.plot(rho_interp)
# plt.show()
# plt.plot(phi)
# plt.plot(phi_interp)
# plt.show()
# #  %%

# xii,yii = pol2cart(rho_interp,phi_interp)
# plt.plot(xii,yii)
# plt.show()
# #  %%
# xxii,yyii = np.meshgrid(xii,yii)

# dist_euclid = euclidean_distances(np.array([xii,yii]).T)
# plt.imshow(dist_euclid)
# plt.show()

# plt.plot(X_transform[:,0],X_transform[:,1])
#  %%
# batch_size = 32

dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

#  %%

# fig,ax = plt.subplots(10,10)
# for i,ax in enumerate(ax.flat):
#     ax.imshow(transform(train_dataset[i]).reshape(window_size,window_size))

#  %%


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        z_dim = (5, 5)
        decoder_input = (12, 12)
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 1, 3, 1)

        # self.softmax = nn.AdaptiveSoftmax()
        self.fc1 = nn.AdaptiveMaxPool2d(z_dim)
        self.fc2 = nn.AdaptiveMaxPool2d(decoder_input)

        # self.conv4 = self.contract_block(128, 256, 3, 1)

        # self.upconv4 = self.contract_block(256, 128, 3, 1)
        self.upconv3 = self.expand_block(1, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 1, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 1, out_channels, 3, 1)

        self.encoder = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.fc1)
        self.decoder = nn.Sequential(
            self.fc2, self.upconv3, self.upconv2, self.upconv1)

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


#  %% VAE

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, h_dim=(1, 5, 5), z_dim=(1, 5, 5), use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.autoencoder = AutoEncoder(1, 1)
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder

        self.z_dim = torch.tensor(z_dim)
        self.x_dim = (1, window_size, window_size)
        self.h_dim = torch.tensor(h_dim)

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))
        self.fc21 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))
        self.fc22 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))

        pyro.module("decoder", self.autoencoder)

        # self.fc3 = nn.Linear(torch.prod(self.z_dim), torch.prod(self.h_dim))
        self.softplus = nn.Softplus()
    
    def encode(self, x):
        h = self.encoder(x)
        # h = self.softplus(h)
        # h = self.flatten(h)
        # z = self.sigmoid(h)
        
        # No clue if this is actually mu
        z = self.fc21(self.flatten(h))
        mu = torch.exp(self.fc22(self.flatten(h)))
        # z, mu = self.bottleneck(h)
        return z.reshape(h.shape), mu.reshape(h.shape)

    def decode(self, z):
        # z = self.fc3(z).reshape(-1,*tuple(self.h_dim))
        z = z.reshape(-1, *tuple(self.h_dim))
        return self.decoder(z)

    def forward(self, x):
        z, mu = self.encode(x)
        return self.decode(z)
        

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], *tuple(self.h_dim))))
            z_scale = x.new_ones(torch.Size((x.shape[0], *tuple(self.h_dim))))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))
            # decode the latent code z
            img = self.decode(z)
            loc_img = torch.sigmoid(img)
            scale = torch.ones_like(loc_img)
            # score against actual images
            pyro.sample("obs", dist.ContinuousBernoulli(logits=img).to_event(3), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encode(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))

    def construct_from_z(self,z):
        return torch.sigmoid(self.decode(z))

    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encode(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        # loc_img = torch.sigmoid(self.decode(z))
        return self.construct_from_z(z)


#  %% VQ-VAE


# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,in_channels):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,out_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)

class VQ_VAE(nn.Module):
    def __init__(self,
                    num_hiddens = 64, 
                    num_residual_hiddens = 32, 
                    num_residual_layers = 2, 
                    embedding_dim = 64,
                    num_embeddings = 512, 
                    commitment_cost = 0.25,
                    decay = 0.99, channels=1):
        super(VQ_VAE, self).__init__()
        
        self._encoder = Encoder(num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens,
                                in_channels=channels)
        self.encoder = self._encoder
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                out_channels=channels)
        self.decoder = self._decoder

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
    
    def model(self,x):
        return self.forward(x)
# define a helper function for reconstructing images

#  %%

# vae = VAE()
# model_check = vae.model(img)


# vae = AutoEncoder(1, 1)
vae = VQ_VAE(channels=1)
img = train_dataset[0].unsqueeze(0)
loss, x_recon, perplexity = vae(img)
z = vae.encoder(img)
y_prime = vae.decoder(z)

# print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
print(f"img_dims:{img.shape} x_recon:_dims:{x_recon.shape} z:_dims:{z.shape}")

# z_loc, z_scale = vae.encode(img)
# out = vae.decode(nn.Flatten()(z))
#  %%

# vae = VAE()
# model_check = vae.model(img)

# z_loc, z_scale = vae.encode(img)
# out = vae.decode(nn.Flatten()(z))

# encode
# guide_check = vae_model.guide(img)
# #  %%

# optimizer = Adam({"lr": 1.0e-3})
# svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# for x in dataloader:
#     epoch_loss = svi.step(x)
#     print(epoch_loss)
#     break
#  %%
# TODO better loss is needed, outshapes are currently not always full
# loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCEWithLogitsLoss()

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size=1, learning_rate=1e-3):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.autoencoder = VQ_VAE(channels=1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        # self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        return self.autoencoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch
        inputs
        vq_loss, x_recon, perplexity = self.forward(inputs)
        output = x_recon
        # loss = self.loss_fn(output, inputs)
        
        # vq_loss, data_recon, perplexity = model(inputs)
        # recon_error = F.mse_loss(output, inputs)
        recon_error = self.loss_fn(output, inputs)
        loss = recon_error + vq_loss # Missing variance bit
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        # self.logger.experiment.add_embedding(
        #     "input_image", torchvision.utils.make_grid(transformer_image(inputs)), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(output), batch_idx)
        # self.logger.experiment.add_embedding(
        #     "output_image", torchvision.utils.make_grid(transformer_image(output)), batch_idx)


        # tensorboard.add_image("input", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        # tensorboard.add_image("output", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        return loss

# %%
class LitVariationalAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size=1, learning_rate=1e-3):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vae = VAE()
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    def forward(self, x):
        return self.vae.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def torch_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.forward(inputs)
        loss = self.loss_fn(output, inputs)
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(torch.sigmoid(output)), batch_idx)

    def pyro_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.vae.reconstruct_img(inputs)
        loss = self.loss_fn(self.vae.model, self.vae.guide, inputs)
        self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(torch.sigmoid(output)), batch_idx)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.torch_training_step(train_batch, batch_idx)

    def training_step(self, train_batch, batch_idx):
        return self.pyro_training_step(train_batch, batch_idx)

# %%

tb_logger = pl_loggers.TensorBoardLogger("runs/")

# from pathlib import Path
# Path("checkpoints/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            )

trainer = pl.Trainer(
    logger=tb_logger,
    enable_checkpointing=True,
    gpus=1,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=75,
)  # .from_argparse_args(args)

model = LitAutoEncoder()
# model = LitVariationalAutoEncoder()
trainer.fit(model, dataloader)

# tb_logger = pl_loggers.TensorBoardLogger("runs/")

# checkpoint_callback = ModelCheckpoint(
#             dirpath="checkpoints/",
#             save_last=True,
#             every_n_train_steps=100)

# last_checkpoint_path = "checkpoints/last.ckpt"

# if not(os.path.isfile(last_checkpoint_path)):
#     last_checkpoint_path = ""

# trainer = pl.Trainer(
#     resume_from_checkpoint=last_checkpoint_path,
#     logger=tb_logger,
#     enable_checkpointing=True,
#     gpus=1,
#     accumulate_grad_batches=1,
#     callbacks=[checkpoint_callback],
#     min_epochs=50,
#     max_epochs=75,
# )  # .from_argparse_args(args)

# #
# # if __name__ = main:
# #

# model = LitAutoEncoder(batch_size=batch_size)
# # model = LitVariationalAutoEncoder()
# trainer.fit(model, dataloader)

#  %%
# model
for i in range(10):
    z_random = torch.normal(torch.zeros_like(z),torch.ones_like(z)).cuda()
    generated_image = model.autoencoder.decoder(z_random)
    plt.imshow(transforms.ToPILImage()(generated_image[0]))
    plt.show()

# loss_function = torch.nn.MSELoss()
#  %%
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

# %%

# %%

# %%
