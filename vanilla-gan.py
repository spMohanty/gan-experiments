#!/usr/bin/env python

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import math

DATA_FOLDER = './torch_data/VGAN/MNIST'
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

DISCRIMINATOR_INPUT_SIZE = 784
NOISE_SIZE = 100
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_features = DISCRIMINATOR_INPUT_SIZE

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

def images_to_vectors(images):
    return images.view(images.size(0), DISCRIMINATOR_INPUT_SIZE)
def vectors_to_images(vector):
    """
    Assumes gray scale image of aspect ratio 1:1
    In this case, its MNIST, so will be batch_size, 1, 28, 28
    """
    return vector.view(vectors.size(0), 1, int(math.sqrt(DISCRIMINATOR_INPUT_SIZE)), int(math.sqrt(DISCRIMINATOR_INPUT_SIZE)))



class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.noise_size = NOISE_SIZE
        self.output_size = DISCRIMINATOR_INPUT_SIZE #This is the input to the Discriminator network

        self.hidden_layer_sizes = [256, 512, 1024]
        self.layers = []

        for _idx, layer_size in enumerate(self.hidden_layer_sizes):
            if _idx == 0:
                input_size = self.noise_size
                output_size = layer_size
            else:
                input_size = self.hidden_layer_sizes[_idx-1]
                output_size = layer_size

            layer = nn.Sequential(
                nn.Linear(input_size, output_size)
            )
            self.layers.append(layer)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_layer_sizes[-1], self.output_size),
            nn.Tanh()
        )
        self.layers.append(self.out)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


# Instantiate networks
discriminator = Discriminator()
generator = Generator()

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    # discriminator = nn.DataParallel(discriminator)
    # generator = nn.DataParallel(generator)


# Instantiate optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()
d_steps = 1 #Number of steps to train the discriminator in every batch
num_epochs = 200


def real_data_target(size):
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # reset gradients
    optimizer.zero_grad()

    # Train on real data
    prediction_real = discriminator(real_data)
    # Calculate error and back propagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()


    # Train on fake data
    prediction_fake = discriminator(fake_data)
    # Calculate error and back propagate
    error_fake = loss(prediction_fake, fake_data_target(fake_data.size(0)))
    error_fake.backward()


    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()

    prediction = discriminator(fake_data)
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()

    optimizer.step()
    return error


num_test_samples = 16
test_noise =    noise(num_test_samples)


for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):

        real_data = Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()

        fake_data = generator(noise(real_data.size(0))).detach()

        # train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # Train Generator
        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)

        print(d_error, g_error, epoch, n_batch, num_batches)
