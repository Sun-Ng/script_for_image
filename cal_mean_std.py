'''
use pytorch to calculate the mean and std of image dataset
'''

import os
import torch
from torchvision import datasets, transforms


train_path = your_dataset_root_dir
traindir = os.path.join(train_path, 'train')
# valdir = os.path.join(train_path, 'val')

dataset = datasets.ImageFolder(
    traindir,
    transform=transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=100,
    num_workers=10,
    shuffle=False)

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*300*300))

print(">>> mean:", mean)
print(">>> std:", std)
