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

data_loader, _ = get_edges_dataloaders(batch_size=64)
img_size = (64, 64, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.5, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 80
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'colored_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
