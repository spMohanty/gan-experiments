#!/usr/bin/env python

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

DATA_FOLDER = './torch_data/VGAN/MNIST'

def mnist_data():
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5 ), (.5, .5, .5))
        ]
    )
    out_dir = "{}/dataset".format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = mnist_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)
print(data)

class Discriminator:
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_features = 784

        self.hidden_layer_sizes = [1024, 512, 256]

        self.n_out = 1

        self.layers = []
        for _idx, layer_size in enumerate(self.hidden_layer_sizes):
            if _idx == 0:
                input_size = self.n_features
                output_size = layer_size
            else:
                input_size = self.hidden_layer_sizes[_idx-1]
                output_size = layer_size

            layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )

            self.layers.append(layer)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_layer_sizes[-1], self.n_out),
            nn.Sigmoid()
        )
        self.layers.append(self.out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x