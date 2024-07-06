import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate real data: points from a circle
def generate_real_data(n):
    r = torch.sqrt(torch.rand(n))
    theta = torch.rand(n) * 2 * torch.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize networks and optimizers
generator = Generator()
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=0.01)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.01)

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
num_epochs = 1000
batch_size = 128

for epoch in range(num_epochs):
    # Generate real and fake data
    real_data = generate_real_data(batch_size)
    z = torch.randn(batch_size, 2)
    fake_data = generator(z)

    # Train Discriminator
    optimizer_D.zero_grad()
    real_loss = adversarial_loss(discriminator(real_data), torch.ones(batch_size, 1))
    fake_loss = adversarial_loss(discriminator(fake_data.detach()), torch.zeros(batch_size, 1))
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    g_loss = adversarial_loss(discriminator(fake_data), torch.ones(batch_size, 1))
    g_loss.backward()
    optimizer_G.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

# Visualize results
with torch.no_grad():
    real_data = generate_real_data(1000)
    z = torch.randn(1000, 2)
    fake_data = generator(z)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.5, label='Real')
    plt.title('Real Data')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c='red', alpha=0.5, label='Fake')
    plt.title('Generated Data')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gan_results.png')
    plt.close()

print("Training finished! Results saved as 'gan_results.png'")