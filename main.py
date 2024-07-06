import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate real data: smooth color gradients
def generate_real_data(n):
    t = torch.linspace(0, 1, n).unsqueeze(1)
    return torch.cat([t, 1-t, torch.zeros_like(t)], dim=1)

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize networks and optimizers
generator = Generator()
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
num_epochs = 10000
batch_size = 256
save_interval = 500

# Lists to store generated samples for animation
generated_samples = []

for epoch in range(num_epochs):
    # Generate real and fake data
    real_data = generate_real_data(batch_size)
    z = torch.rand(batch_size, 1)
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

    if (epoch + 1) % save_interval == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        generated_samples.append(fake_data.detach().numpy())

# Create animation
fig, ax = plt.subplots(figsize=(10, 4))

def animate(i):
    ax.clear()
    ax.imshow(generated_samples[i].reshape(1, -1, 3), aspect='auto')
    ax.set_title(f'Epoch {(i+1)*save_interval}')
    ax.axis('off')

anim = FuncAnimation(fig, animate, frames=len(generated_samples), interval=200, repeat_delay=1000)
anim.save('color_gradient_gan.gif', writer='pillow', fps=5)

print("Training finished! Animation saved as 'color_gradient_gan.gif'")

# Display final results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

real_data = generate_real_data(batch_size)
ax1.imshow(real_data.numpy().reshape(1, -1, 3), aspect='auto')
ax1.set_title('Real Color Gradients')
ax1.axis('off')

z = torch.rand(batch_size, 1)
fake_data = generator(z)
ax2.imshow(fake_data.detach().numpy().reshape(1, -1, 3), aspect='auto')
ax2.set_title('Generated Color Gradients')
ax2.axis('off')

plt.tight_layout()
plt.savefig('color_gradient_comparison.png')
plt.close()

print("Comparison image saved as 'color_gradient_comparison.png'")