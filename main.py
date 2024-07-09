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
from datetime import datetime

# Configuration
CONFIG = {
    'random_seed': 42,
    'latent_dim': 64,
    'image_size': 32,
    'batch_size': 8,
    'epochs': 100,
    'n_samples': 3000,
    'learning_rate': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'data_root': './data/celeba',
    'output_dir': './output'
}


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        CONFIG['output_dir'],
        f"output_b{CONFIG['batch_size']}_e{CONFIG['epochs']}_i{CONFIG['image_size']}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_transforms():
    return transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.CenterCrop(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_all_image_paths(root):
    image_dir = os.path.join(root, 'img_align_celeba')
    if not os.path.exists(image_dir):
        print("Extracting images...")
        with zipfile.ZipFile(os.path.join(root, 'img_align_celeba.zip'), 'r') as zip_ref:
            zip_ref.extractall(root)

    all_files = os.listdir(image_dir)
    print(f"Total files in directory: {len(all_files)}")

    image_paths = [os.path.join(image_dir, img) for img in all_files
                   if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total image files found: {len(image_paths)}")
    return image_paths


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            width, height = img.size
            if width != 256 or height != 256:
                print(f"Image with incorrect dimensions: {path} ({width}x{height})")
                return False
        return True
    except Exception as e:
        print(f"Invalid or corrupted image: {path}. Error: {str(e)}")
        return False


def filter_valid_images(image_paths):
    valid_paths = []
    for path in image_paths:
        if is_valid_image(path):
            valid_paths.append(path)
        if len(valid_paths) % 100 == 0:
            print(f"Processed {len(valid_paths)} valid images so far...")
    return valid_paths


def print_image_stats(image_paths):
    dimensions = {}
    for path in image_paths:
        with Image.open(path) as img:
            size = img.size
            if size in dimensions:
                dimensions[size] += 1
            else:
                dimensions[size] = 1
    print("Image dimension statistics:")
    for size, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
        print(f"{size}: {count} images")


class CelebADataset(Dataset):
    def __init__(self, root, transform=None, n_samples=None):
        self.root = root
        self.transform = transform

        all_image_paths = get_all_image_paths(root)

        if n_samples is not None and n_samples < len(all_image_paths):
            self.image_paths = random.sample(all_image_paths, n_samples)
        else:
            self.image_paths = all_image_paths

        print(f"Images before validation: {len(self.image_paths)}")

        self.image_paths = filter_valid_images(self.image_paths)
        print(f"Using {len(self.image_paths)} valid images for training.")

        print_image_stats(self.image_paths)

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
            return self[random.randint(0, len(self) - 1)]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(CONFIG['latent_dim'], 256, 4, 1, 0, bias=False),
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


def save_generated_images(epoch, generator, device, output_dir):
    with torch.no_grad():
        gen_imgs = generator(torch.randn(16, CONFIG['latent_dim'], 1, 1).to(device)).cpu()
        grid = make_grid(gen_imgs, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"celeba_generated_epoch_{epoch}.png"))
        plt.close()


def train_batch(real_imgs, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, device):
    batch_size = real_imgs.size(0)
    real = torch.ones(batch_size, 1).to(device)
    fake = torch.zeros(batch_size, 1).to(device)

    # Train Generator
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, CONFIG['latent_dim'], 1, 1).to(device)
    gen_imgs = generator(z)
    g_loss = adversarial_loss(discriminator(gen_imgs), real)
    g_loss.backward()
    optimizer_G.step()

    # Train Discriminator
    optimizer_D.zero_grad()
    real_loss = adversarial_loss(discriminator(real_imgs), real)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    return d_loss.item(), g_loss.item()


def train():
    set_random_seed(CONFIG['random_seed'])
    device = get_device()
    print(f"Using device: {device}")

    output_dir = create_output_directory()
    print(f"Outputs will be saved to: {output_dir}")

    transform = get_transforms()
    dataset = CelebADataset(root=CONFIG['data_root'], transform=transform, n_samples=CONFIG['n_samples'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['learning_rate'],
                             betas=(CONFIG['beta1'], CONFIG['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate'],
                             betas=(CONFIG['beta1'], CONFIG['beta2']))

    adversarial_loss = nn.BCELoss()

    for epoch in range(CONFIG['epochs']):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for i, imgs in enumerate(dataloader):
            print(f"Batch {i} size: {imgs.size()}")  # Print batch size
            real_imgs = imgs.to(device)
            d_loss, g_loss = train_batch(real_imgs, generator, discriminator, optimizer_G, optimizer_D,
                                         adversarial_loss, device)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss

            if i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{CONFIG['epochs']}] Batch [{i}/{len(dataloader)}] D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")

        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] Avg D loss: {avg_d_loss:.4f}, Avg G loss: {avg_g_loss:.4f}")

        save_generated_images(epoch + 1, generator, device, output_dir)

    print(f"Training complete. Generated images saved in {output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
