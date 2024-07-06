import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate real data: mixture of Gaussians
def generate_real_data(n):
    return torch.from_numpy(np.random.multivariate_normal(
        mean=[0, 0], cov=[[0.1, 0], [0, 0.1]], size=n
    ).astype(np.float32))

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
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

# Training parameters
n_epochs = 1000
batch_size = 128
sample_size = 1000

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
scatter_real = ax.scatter([], [], c='blue', alpha=0.5, label='Real Data')
scatter_fake = ax.scatter([], [], c='red', alpha=0.5, label='Generated Data')
contour = ax.contourf(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100),
                      np.zeros((100, 100)), levels=[0, 0.5, 1],
                      alpha=0.3, colors=['red', 'green'])
ax.legend()
plt.close()  # Prevent the empty figure from displaying

# Training loop with animation
def train(frame):
    # Generate real and fake data
    real_data = generate_real_data(batch_size)
    z = torch.randn(batch_size, 2)
    fake_data = generator(z).detach()

    # Train Discriminator
    optimizer_D.zero_grad()
    real_loss = adversarial_loss(discriminator(real_data), torch.ones(batch_size, 1))
    fake_loss = adversarial_loss(discriminator(fake_data), torch.zeros(batch_size, 1))
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, 2)
    fake_data = generator(z)
    g_loss = adversarial_loss(discriminator(fake_data), torch.ones(batch_size, 1))
    g_loss.backward()
    optimizer_G.step()

    # Visualize results
    if frame % 10 == 0:
        with torch.no_grad():
            # Sample points for visualization
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x, y)
            Z = discriminator(torch.Tensor(np.column_stack([X.ravel(), Y.ravel()]))).reshape(X.shape).numpy()

            # Update plot
            ax.clear()
            ax.contourf(X, Y, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'green'])
            ax.scatter(real_data[:sample_size, 0], real_data[:sample_size, 1], c='blue', alpha=0.5, label='Real Data')
            fake_sample = generator(torch.randn(sample_size, 2)).detach().numpy()
            ax.scatter(fake_sample[:, 0], fake_sample[:, 1], c='red', alpha=0.5, label='Generated Data')
            ax.legend()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_title(f'Epoch {frame+1}')

    return ax

# Create the animation
anim = FuncAnimation(fig, train, frames=n_epochs, interval=20, repeat=False)
anim.save('gan_training.gif', writer='pillow', fps=30)

print("Training and animation complete. Check 'gan_training.gif' in your current directory.")
