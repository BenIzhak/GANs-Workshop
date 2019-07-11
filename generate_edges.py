# -*- coding: utf-8 -*-

import torch
import torch.nn.parallel
import numpy as np
from DCGAN_models import Generator
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Gpath", help="Path to the generator weights")
parser.add_argument("Rpath", help="The generated edges will be save in that directory")
parser.add_argument("numOfImages", help="How many images we want to generate", nargs='?', default=2048, type=int)
parser.add_argument("manualSeed", help="Set random seed for reproducibility", nargs='?', default=999, type=int)
args = parser.parse_args()


# Set random seem for reproducibility
manualSeed = args.manualSeed
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# The path to the weights 
Gpath = args.Gpath

# How many images to generate
batch_size = args.numOfImages

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of channels in the training images
nc = 1

# Size of feature maps in generator
ngf = 64

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def load_model(filepath, nz, ngf, nc):
    checkpoint = torch.load(filepath)
    model = Generator(ngpu, nz, ngf, nc).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def normalize(img):
    img = img + 1
    min_value = img.min()
    max_value = img.max()
    img = img - min_value
    img = img / max_value
    return img

netG = load_model(Gpath, nz, ngf, nc).to(device)

fixed_noise = torch.randn(batch_size , nz, 1, 1, device=device)

fake = netG(fixed_noise).detach().cpu()

for i in range(batch_size):
    img = fake.numpy()[i]
    img = normalize(img)
    img = np.transpose(img, (1,2,0))
    plt.axis("off")
    plt.imsave(args.Rpath + "/g-" + str(i) + ".png", img.reshape(64,64), cmap='gray')
