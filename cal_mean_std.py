#-*- coding : utf-8 -*-
# coding: utf-8

'''
    Author: Xin Wu
    E-mail: wuxin@icarbonx.com
    Spe, 2019
    use pytorch to calculate the mean and std of image dataset
'''
import os
import torch
from torchvision import datasets, transforms


train_path = "your_dataset_root_dir"
traindir = os.path.join(train_path, 'train')

image_resize = 300

dataset = datasets.ImageFolder(
    traindir,
    transform=transforms.Compose([
        transforms.Resize([image_resize, image_resize]),
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
std = torch.sqrt(var / (len(loader.dataset) * image_resize * image_resize))

print(">>> mean:", mean)
print(">>> std:", std)

