#!/usr/bin/env python

import torch
import numpy as np
from PIL import Image
import os
import glob
import random

from torch.utils.data import Dataset
from torchvision import transforms

class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        #Estimate classes
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        for _dir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, _dir)):
                self.classes.append(_dir)

        self.files = glob.glob(os.path.join(root_dir, "*", "*"))
        random.shuffle(self.files)
        self.labels = [self.classes.index(x.split("/")[-2]) for x in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx])
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]

        return im, label

if __name__ == "__main__":

    pv_transforms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = PlantVillageDataset("data/mini-plantvillage", transform=pv_transforms)
    img, label = dataset[0]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    print(len(data_loader))
