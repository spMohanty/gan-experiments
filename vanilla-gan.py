#!/usr/bin/env python

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import math
import torchvision.utils as vutils

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
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
def images_to_vectors(images):
    return images.view(images.size(0), DISCRIMINATOR_INPUT_SIZE)
def vectors_to_images(vector):
    """
    Assumes gray scale image of aspect ratio 1:1
    In this case, its MNIST, so will be batch_size, 1, 28, 28
    """
    return vector.view(vector.size(0), 1, int(math.sqrt(DISCRIMINATOR_INPUT_SIZE)), int(math.sqrt(DISCRIMINATOR_INPUT_SIZE)))



class Generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
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
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

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


"""
Setup Plotters
"""

from plotter import *

loss_plot = TortillaLinePlotter(
    experiment_name="gan-experiments",
    fields=["discriminator_error", "generator_error"],
    title="Loss",
    opts=dict(
        xtickmin=0,
        xtickmax=num_epochs,
        xlabel="Epochs",
        ylabel="Loss"
    )
)

images_plot = TortillaImagesPlotter(
    experiment_name="gan-experiments",
    title="Generated Images",
    opts=dict(
        padding=5,
        nrows=8
    )
)

def save_models(generator, discriminator, epoch):
    print("Saving checkpoint....")
    torch.save(generator.state_dict(),
               '{}/G_epoch_{}'.format("checkpoints", epoch))
    torch.save(discriminator.state_dict(), '{}/D_epoch_{}'.format("checkpoints", epoch))

# TRAIN
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

        if n_batch % 100 == 0:
            time = epoch + n_batch*1.0/len(data_loader)
            loss_plot.append_plot(np.array([d_error.data[0], g_error.data[0]]), time)
            num_samples = 64
            generated_sample = vectors_to_images(fake_data[:num_samples]).data.cpu()

            # generated_sample = vutils.make_grid(generated_sample, normalize=True, scale_each=True)

            # images_plot.update_images(real_batch[:64].cpu())
            images_plot.update_images(generated_sample)

            print(d_error.data[0], g_error.data[0], epoch, n_batch, num_batches)

    save_models(generator, discriminator, epoch)
