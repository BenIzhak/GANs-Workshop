import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Generator, Discriminator
from training import Trainer

# Define dataloader
def get_edges_dataloaders(dataroot = '/home/beni/dataset1', batch_size=128):
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.ImageFolder(root=dataroot, transform=all_transforms)
    test_data = datasets.ImageFolder(root=dataroot, transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('/home/beni/dataset_MNIST', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('/home/beni/dataset_MNIST', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

#data_loader, _ = get_edges_dataloaders(batch_size=64)
data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 191
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'MNIST_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
