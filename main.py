import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import zipfile
import multiprocessing
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 64
image_size = 32
batch_size = 8
epochs = 100
n_samples = 3000

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Custom CelebA dataset
class CelebADataset(Dataset):
    def __init__(self, root, transform=None, n_samples=None):
        self.root = root
        self.transform = transform

        if not os.path.exists(os.path.join(root, 'img_align_celeba')):
            print("Extracting images...")
            with zipfile.ZipFile(os.path.join(root, 'img_align_celeba.zip'), 'r') as zip_ref:
                zip_ref.extractall(root)

        all_image_paths = [os.path.join(root, 'img_align_celeba', img) for img in
                           os.listdir(os.path.join(root, 'img_align_celeba')) if img.endswith('.jpg')]

        if n_samples is not None and n_samples < len(all_image_paths):
            self.image_paths = random.sample(all_image_paths, n_samples)
        else:
            self.image_paths = all_image_paths

        # Filter out unreadable images
        self.image_paths = [path for path in self.image_paths if self.is_valid_image(path)]
        print(f"Using {len(self.image_paths)} valid images for training.")

    def is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except:
            print(f"Skipping corrupted image: {path}")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random valid image instead
            return self[random.randint(0, len(self) - 1)]


# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)


# Function to generate and save images
def save_generated_images(epoch, generator):
    with torch.no_grad():
        gen_imgs = generator(torch.randn(16, latent_dim, 1, 1).to(device)).cpu()
        grid = make_grid(gen_imgs, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f"celeba_generated_epoch_{epoch}.png")
        plt.close()


def train():
    dataset = CelebADataset(root='./data/celeba', transform=transform, n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()

    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            real = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            real_imgs = imgs.to(device)

            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), real)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"Epoch [{epoch + 1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        save_generated_images(epoch + 1, generator)

    print("Training complete. Check the generated images in your current directory.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
