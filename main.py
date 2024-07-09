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
    'latent_dim': 128,
    'image_size': 128,
    'batch_size': 64,
    'epochs': 300,
    'n_samples': 3000,
    'learning_rate': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'data_root': './data/celeba',
    'output_dir': './output',
    'generator_features': [1024, 512, 256, 128, 64],  # Adjusted for 128x128 images
    'discriminator_features': [64, 128, 256, 512, 1024],  # Adjusted for 128x128 images
    'dropout': 0.3,
    'use_spectral_norm': True
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


def gen_block(in_feat, out_feat, normalize=True, dropout=False):
    layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat))
    layers.append(nn.ReLU(True))
    if dropout:
        layers.append(nn.Dropout(CONFIG['dropout']))
    return nn.Sequential(*layers)


def disc_block(in_feat, out_feat, normalize=True, dropout=False):
    layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
    if CONFIG['use_spectral_norm']:
        layers[0] = nn.utils.spectral_norm(layers[0])
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if dropout:
        layers.append(nn.Dropout(CONFIG['dropout']))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = CONFIG['image_size'] // (2 ** len(CONFIG['generator_features']))
        self.l1 = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'],
                      CONFIG['generator_features'][0] * self.init_size ** 2)
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(len(CONFIG['generator_features']) - 1):
            self.conv_blocks.append(
                gen_block(CONFIG['generator_features'][i],
                          CONFIG['generator_features'][i + 1],
                          dropout=(i < len(CONFIG['generator_features']) // 2))
            )

        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(CONFIG['generator_features'][-1], 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], CONFIG['generator_features'][0], self.init_size, self.init_size)
        for conv_block in self.conv_blocks:
            out = conv_block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.ModuleList()
        in_features = 3
        for i, out_features in enumerate(CONFIG['discriminator_features']):
            self.model.append(
                disc_block(in_features, out_features,
                           normalize=(i != 0),
                           dropout=(i < len(CONFIG['discriminator_features']) // 2))
            )
            in_features = out_features

        self.model.append(nn.Conv2d(in_features, 1, 4, 1, 0, bias=False))
        self.model.append(nn.Sigmoid())

    def forward(self, img):
        for layer in self.model:
            img = layer(img)
        return img.view(-1, 1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def save_generated_images(epoch, generator, device, output_dir):
    with torch.no_grad():
        gen_imgs = generator(torch.randn(16, CONFIG['latent_dim']).to(device)).cpu()
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
    z = torch.randn(batch_size, CONFIG['latent_dim']).to(device)
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

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['learning_rate'],
                             betas=(CONFIG['beta1'], CONFIG['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate'],
                             betas=(CONFIG['beta1'], CONFIG['beta2']))

    adversarial_loss = nn.BCELoss()

    save_generated_images(0, generator, device, output_dir)
    for epoch in range(CONFIG['epochs']):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for i, imgs in enumerate(dataloader):
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
