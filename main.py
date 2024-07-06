import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
img_size = 28
batch_size = 64
epochs = 500


# Generate simple "8"-like figures
def generate_eight(size=28, noise=0.2):
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 4
    t = np.linspace(0, 2 * np.pi, 100)
    x = center + radius * np.sin(t)
    y = center + radius / 2 * np.sin(2 * t)
    x = x.astype(int)
    y = y.astype(int)
    img[y, x] = 1
    img += np.random.randn(size, size) * noise
    img = np.clip(img, 0, 1)
    return img


# Create a dataset of "8"s
num_samples = 1000
dataset = torch.tensor([generate_eight() for _ in range(num_samples)]).float().unsqueeze(1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, img_size, img_size)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training
start_time = time.time()
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        # Adversarial ground truths
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        # Generate and save images
        with torch.no_grad():
            gen_imgs = generator(torch.randn(25, latent_dim)).cpu()
            fig, axs = plt.subplots(5, 5, figsize=(10, 10))
            for ax, img in zip(axs.flat, gen_imgs):
                ax.imshow(img.squeeze(), cmap='gray')
                ax.axis('off')
            plt.savefig(f"generated_8s_epoch_{epoch + 1}.png")
            plt.close()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Generate final results
with torch.no_grad():
    gen_imgs = generator(torch.randn(25, latent_dim)).cpu()
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for ax, img in zip(axs.flat, gen_imgs):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig("final_generated_8s.png")
    plt.close()

print("Training complete. Check the generated images in your current directory.")
