# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import argparse
from DCGAN_models import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument("dataroot", help="Path to the dataroot")
parser.add_argument("manualSeed", help="Set random seed for reproducibility", nargs='?', default=999, type=int)
parser.add_argument("epochs", help="Number of epochs", nargs='?', default=101, type=int)
parser.add_argument("batchSize", help="Batch size", nargs='?', default=16, type=int)
args = parser.parse_args()

# Set random seed for reproducibility
manualSeed = args.manualSeed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = args.dataroot

# If you already have pre-trained model put here path to the generator weights
Gpath = None  
# If you already have pre-trained model put here path to the discriminator weights
Dpath = None #"/home/beni/checkpointD60.pth"

# Number of workers for dataloader
workers = 0

# Number of training epochs
num_epochs = args.epochs

# Batch size during training
batch_size = args.batchSize

# Spatial size of training images
# for diffrent size use transforms
image_size = 64

# Save weights every "save_weights" epochs
save_weights = 10

# Save generated samples every 10 epoch after epoch 50 
save_samples = True

# Number of channels in the training images
nc = 1

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


if(image_size == 64):
    transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),
                            ])
else:
    transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),
                           ])


# Create the dataset
dataset = dset.ImageFolder(root=dataroot, transform=transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# load model 
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_optim(filepath, optim):
    checkpoint = torch.load(filepath)
    optim.load_state_dict(checkpoint['optimizer'])
    return optim

# save fake images
def saveImages(epoch, img_list):
    # Plot the fake images from the last epoch
    plt.axis("off")
    plt.title("Fake Images [" + str(epoch) + "]")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('/cloudstorage/benResult/result_' + str(epoch) + '.png')
      
# Create the generator
if(Gpath != None):
    netG = load_model(Gpath).to(device)
else:
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)

# Create the Discriminator
if(Dpath != None):
    netD = load_model(Dpath).to(device)
else:
    netD = Discriminator(ngpu, ndf, nc).to(device)
    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netD.apply(weights_init)
  
# Print the model
print(netG)
# Print the model
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

if(Gpath != None):
    load_optim(Gpath, optimizerG)
if(Dpath != None):
    load_optim(Dpath, optimizerD)

# Setup scheduler for optimizers
schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.5)
schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.5)


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
errG = 0


# Training Loop
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ##############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################################################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        if(i % 3 == 0):
            if(epoch > 60):
                if(i % 4 == 0):
                    errD_real.backward()
            else:
                errD_real.backward()
                
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D 
        if(i % 3 == 0):
            if(epoch > 60):
                if(i % 4 == 0):
                    errD_fake.backward()
                    optimizerD.step()
            else:
                errD_fake.backward()
                optimizerD.step()
            
        #############################################
        # (2) Update G network: maximize log(D(G(z)))
        #############################################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%03d/%03d][%03d/%03d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
        
    if(epoch % 8 == 0):
       schedulerD.step()
       schedulerG.step()

    #save model and fake images
    if (epoch % save_weights == 0): 
       checkpoint = {
             'state_dict': netG.state_dict(),
             'optimizer' : optimizerG.state_dict()}

       torch.save(checkpoint, 'checkpointG' + str(epoch) + '.pth')
    
       checkpoint = {
             'state_dict': netD.state_dict(),
             'optimizer' : optimizerD.state_dict()}

       torch.save(checkpoint, 'checkpointD' + str(epoch) + '.pth')
       
    if (save_samples and epoch >= 0 and epoch % 10 == 0):
       saveImages(epoch, img_list)

    
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('result_' + str(num_epochs) + '.png')

# Plot 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_plot_' + str(num_epochs) + '.png')

