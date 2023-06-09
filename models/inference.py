import torch
import umap
import numpy as np
import torch.nn as nn
from torchvision import transforms
import os
import requests
import json
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import umap.plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#import os
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset
from idr import connection
model_path = "models/model.pt"

# Load the TorchScript model
model = torch.jit.load(model_path)

# Use the loaded model for inference or other tasks
# output = model(torch.randn(1, 1, 512, 512))
encoder = model.model._encoder
# output = encoder(torch.randn(1, 1, 512, 512))


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=512),
        # transforms.RandomCrop(size=(512,512)),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485), (0.229)),
    ]
)


# Create a connection
conn = connection("idr.openmicroscopy.org", "public", "public")

# Define the folder
folder = "/home/ctr26/gdrive/+projects/ai_data/data/idr/inference"

# Get the list of .tiff files in the folder and subfolders
glob_strs = ["*", "0_3_0", "0_2_0", "0_1_0", "0_0_0"]
image_name = glob_strs[-1]
# for glob_str in glob_strs:
glob_str = f"/**/{image_name}.tiff"
# seed = 42
filenames = glob.glob(folder + glob_str, recursive=True)
# filenames = np.random.RandomState(seed).permutation(filenames)
# Initialize lists for images and labels
images = []
labels = []
from io import BytesIO

# Loop over the filenames
for filename in filenames[0:500]:
    # Open the image and convert to PyTorch tensor
    with open(image_name, 'rb') as f:
        image_data = f.read()
    image = Image.open(BytesIO(image_data))
    # image = Image.open(filename)
    # image = ToTensor()(image)
    images.append(image)

    # Extract the image ID from the filename
    image_id = os.path.basename(os.path.dirname(filename))

    # Get the image data
    image_data = conn.getObject("Image", image_id)
    well_id = image_data.getParent().getWell().getId()
    well = conn.getObject("Well", well_id)
    metadata = well.getAnnotation().getValue()
    try:
        gene_symbol = dict(metadata)["Gene Symbol"]
    except:
        gene_symbol = "Wildtype"

    # well.getAnnotation().getValue()
    # gene_id = image_data.getPrimaryAnnotatedTerms()[0]['id']
    labels.append(gene_symbol)


# Now you have your images and labels, you can create a PyTorch dataset:
class GeneDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


dataset = GeneDataset(images, labels)

z = [encoder((transform(data[0]).unsqueeze(0).to(torch.float))) for data in dataset]
labels = np.array([data[1] for data in dataset])
X = torch.stack(z).detach().numpy().reshape(500, -1)
# Create a UMAP object and fit-transform the data
reducer = umap.UMAP()

# Convert the target variable to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# projection = reducer.fit_transform(X, y=y)
mapper = reducer.fit(X, y=y)

umap.plot.points(mapper, labels=labels)
umap_plot.savefig(f"{image_name}.png")
plt.show()

conn.close()


# Create a pipeline with PCA and Random Forest classifier
pipeline = Pipeline(
    [
        ("pca", PCA(n_components=1)),  # Set the number of desired components for PCA
        ("classifier", RandomForestClassifier()),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# Predict labels for the validation set
y_pred = classifier.predict(X_test)

# Evaluate the classifier using classification report
classification_metrics = classification_report(y_test, y_pred)
print(classification_metrics)
