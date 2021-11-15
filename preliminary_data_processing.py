
from numpy import asarray
from PIL import Image
import pandas as pd
import shutil
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


dir = "/711Ass3/"

data = pd.read_csv( dir + 'Train.csv')
data.isnull().values.any() #check for any missing values
data.rename(columns={"width": "xmax", "height": "ymax", "class": "label"}, inplace=True)
data['width'] = 512
data['height'] = 512

indices = data.index.values
for a in indices:
  data.loc[a, "xmax"] = data.loc[a, "xmin"] + data.loc[a, "xmax"]
  data.loc[a, "ymax"] = data.loc[a, "ymin"] + data.loc[a, "ymax"]

data['file_name'] = data['Image_ID'].map(lambda s: s+ ".jpg")

from sklearn.model_selection import train_test_split

data , test = train_test_split(data, test_size=0.2, stratify = data['label'])

data.to_csv( dir + 'train_set.csv', index= False)
test.to_csv( dir + 'test_set.csv', index= False)

os.chdir( dir + 'Train_Images/')
dst_dir =  dir + "Training/"
for f in data['file_name'].values:
  shutil.copy(f, dst_dir)

os.chdir( dir + 'Train_Images/')
dst_dir =  dir + "Testing/"
for f in test['file_name'].values:
  shutil.copy(f, dst_dir)


transform_img = transforms.Compose([
    transforms.ToTensor()
])

image_data = torchvision.datasets.ImageFolder(
  root= dir + 'Training/', transform=transform_img
)
image_data_loader = DataLoader(
  image_data, 
  batch_size=64, 
  shuffle=False, 
  num_workers=0
)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

print(get_mean_and_std(image_data_loader))