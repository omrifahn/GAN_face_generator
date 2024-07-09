import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image
import os
import zipfile
import random
from typing import Tuple
from datetime import datetime

# Configuration
CONFIG = {
    'seed': 42,
    'latent_dim': 100,
    'image_size': 256,
    'batch_size': 64,
    'epochs': 100,
    'n_samples': 3000,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'data_root': './data/celeba',
    'zip_file': 'img_align_celeba.zip',
    'image_folder': 'img_align_celeba'
}

# Set random seed for reproducibility
torch.manual_seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_e{CONFIG['epochs']}_s{CONFIG['image_size']}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(CONFIG['image_size']),
    transforms.CenterCrop(CONFIG['image_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CelebADataset(Dataset):
    def __init__(self, root: str, transform=None, n_samples: int = None):
        self.root = root
        self.transform = transform
        self._extract_images_if_needed()
        self.image_paths = self._get_valid_image_paths(n_samples)
        print(f"Using {len(self.image_paths)} valid images for training.")

    def _extract_images_if_needed(self):
        if not os.path.exists(os.path.join(self.root, CONFIG['image_folder'])):
            print("Extracting images...")
            with zipfile.ZipFile(os.path.join(self.root, CONFIG['zip_file']), 'r') as zip_ref:
                zip_ref.extractall(self.root)

    def _get_valid_image_paths(self, n_samples: int) -> list:
        all_image_paths = [os.path.join(self.root, CONFIG['image_folder'], img) for img in
                           os.listdir(os.path.join(self.root, CONFIG['image_folder'])) if img.endswith('.jpg')]

        if n_samples and n_samples < len(all_image_paths):
            all_image_paths = random.sample(all_image_paths, n_samples)

        return [path for path in all_image_paths if self._is_valid_image(path)]

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except:
            print(f"Skipping corrupted image: {path}")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image) if self.transform else image
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return self[random.randint(0, len(self) - 1)]


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 64 x 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is 3 x 256 x 256
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 128 x 128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 64 x 64
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 32 x 32
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 16 x 16
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1024 x 8 x 8
            nn.Conv2d(1024, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)


def save_generated_images(epoch: int, generator: nn.Module):
    with torch.no_grad():
        gen_imgs = generator(torch.randn(16, CONFIG['latent_dim'], 1, 1, device=device)).cpu()
        grid = make_grid(gen_imgs, nrow=4, normalize=True)
        save_image(grid, os.path.join(output_dir, f"generated_epoch_{epoch}.png"))


def train_gan():
    dataset = CelebADataset(root=CONFIG['data_root'], transform=transform, n_samples=CONFIG['n_samples'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    generator = Generator(CONFIG['latent_dim']).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], CONFIG['beta2']))

    adversarial_loss = nn.BCELoss()

    for epoch in range(CONFIG['epochs']):
        for i, imgs in enumerate(dataloader):
            batch_size = imgs.size(0)
            real = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, CONFIG['latent_dim'], 1, 1, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), real)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs.to(device)), real)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
        save_generated_images(epoch + 1, generator)

    print(f"Training complete. Generated images saved in: {output_dir}")


if __name__ == "__main__":
    train_gan()
