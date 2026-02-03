import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader

# handle device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # construct generator based on the hint
        # we use nn.Sequential for clean layering
        self.model = nn.Sequential(
            # Linear args.latent_dim -> 128
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 128 -> 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 1024 -> 784 (MNIST is 28x28=784)
            # Note: The template said 768, but MNIST requires 784.
            nn.Linear(1024, 784),
            # Output non-linearity: Tanh to scale between -1 and 1
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        # reshape to (Batch, Channel, Height, Width)
        img = img_flat.view(img_flat.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Linear 784 -> 512
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Linear 256 -> 1
            nn.Linear(256, 1),
            # Output non-linearity: Sigmoid for probability (0 to 1)
            nn.Sigmoid()
        )

    def forward(self, img):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # move to device
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    total_batches = len(dataloader) * args.n_epochs
    halfway_point = total_batches // 2

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(batch_size, args.latent_dim).to(device)

            # generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            # we want D(G(z)) to be 1 (valid)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Logging & Saving
            # -----------------
            batches_done = epoch * len(dataloader) + i
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            # save Images Logic for Task 2
            # 1. Start of training
            if batches_done == 0:
                save_image(gen_imgs[:25], 'images/task2_start.png', nrow=5, normalize=True)
            # 2. Halfway through training
            elif batches_done == halfway_point:
                save_image(gen_imgs[:25], 'images/task2_halfway.png', nrow=5, normalize=True)
            # 3. Regular interval saving
            elif batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25], f'images/{batches_done}.png', nrow=5, normalize=True)
    
    # End of training save
    save_image(gen_imgs[:25], 'images/task2_final.png', nrow=5, normalize=True)

def interpolate_task3(generator, latent_dim):
    """
    Task 3: Sample 2 images, interpolate between them (7 steps, 9 total).
    """
    print("Running Task 3 Interpolation...")
    generator.eval() # Set to evaluation mode
    
    
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    # Interpolation: z' = (1-alpha)*z1 + alpha*z2
    alphas = np.linspace(0, 1, 9)
    z_interp_list = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        z_interp_list.append(z_interp)
        
    # stack into a single batch tensor
    z_batch = torch.cat(z_interp_list, dim=0) # Shape: (9, 100)
    
    with torch.no_grad():
        interp_imgs = generator(z_batch)
        
    save_image(interp_imgs, 'images/task3_interpolation.png', nrow=9, normalize=True)
    print("Task 3 image saved to images/task3_interpolation.png")


def main():
    # create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    
    # adam usually works better with beta1=0.5 for GANs
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args)

    # save model
    torch.save(generator.state_dict(), "mnist_generator.pt")

    # perform Task 3
    interpolate_task3(generator, args.latent_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs') # Reduced default to 50 for speed, standard MNIST converges fast
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500, help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()