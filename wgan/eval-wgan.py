# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:24:04 2019

@author: beniz
"""
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import Generator, Discriminator

filepath = '/home/beni/gen_MNIST_model.pt'
img_size = (32, 32, 1)
batch_size = 64
nz = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(filepath, generator):
    checkpoint = torch.load(filepath)
    generator.load_state_dict(checkpoint)
  
def sample_generator(generator, num_samples, use_cuda):
    latent_samples = Variable(generator.sample_latent(num_samples))
    if use_cuda:
        latent_samples = latent_samples.cuda()
    generated_data = generator(latent_samples)
    return generated_data

def sample(generator, num_samples):
    generated_data = sample_generator(generator,num_samples, True)
    # Remove color channel
    return generated_data.data.cpu()

generator = Generator(img_size=img_size, latent_dim=100, dim=16).to(device)

load_model(filepath, generator)

fixed_noise = torch.randn(batch_size , nz, 1, 1, device=device)
fake = sample(generator, 64)


# Plot the fake images from the last epoch
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=3, normalize=True),(1,2,0)))
plt.savefig('/cloudstorage/result-MNIST-wgan.png')

  
    
