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
epochs = 200  # Increased epochs for more complex task
num_classes = 10  # 0-9 digits


# Generate simple digit-like figures
def generate_digit(digit, size=28, noise=0.2):
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 4

    if digit in [0, 6, 8, 9]:  # Digits with loops
        t = np.linspace(0, 2 * np.pi, 100)
        x = center + radius * np.sin(t)
        y = center + radius * np.cos(t)
        if digit == 6:
            y = np.clip(y, center, size)
        elif digit == 9:
            y = np.clip(y, 0, center)
        x, y = x.astype(int), y.astype(int)
        img[y, x] = 1

    if digit in [1, 4, 7]:  # Digits with vertical lines
        img[:, center] = 1

    if digit in [2, 3, 4, 5, 7]:  # Digits with horizontal lines
        img[center, :] = 1

    if digit in [2, 3, 5]:  # Digits with additional horizontal lines
        img[center // 2, :] = 1
        img[center + center // 2, :] = 1

    img += np.random.randn(size, size) * noise
    return np.clip(img, 0, 1)


# Create a dataset of digits
num_samples = 10000  # Increased sample size
digits = np.random.randint(0, 10, num_samples)
dataset = np.array([generate_digit(d) for d in digits])
dataset = torch.tensor(dataset).float().unsqueeze(1)
labels = torch.tensor(digits).long()
dataset = torch.utils.data.TensorDataset(dataset, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(z)
        img = img.view(img.size(0), 1, img_size, img_size)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, self.label_emb(labels)), -1)
        validity = self.model(x)
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
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)

        # Adversarial ground truths
        real = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs, labels), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        # Generate and save images
        with torch.no_grad():
            n_rows = 10
            n_cols = 10
            z = torch.randn(n_rows * n_cols, latent_dim)
            labels = torch.tensor([[i] * n_cols for i in range(n_rows)]).flatten()
            gen_imgs = generator(z, labels).cpu()

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))
            for ax, img, label in zip(axs.flatten(), gen_imgs, labels):
                ax.imshow(img.squeeze(), cmap='gray')
                ax.axis('off')
                ax.set_title(f"Digit: {label.item()}")
            plt.savefig(f"generated_digits_epoch_{epoch + 1}.png")
            plt.close()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

print("Training complete. Check the generated images in your current directory.")
