import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# === 1. Data Loading (MNIST) ===
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=256, shuffle=True)

# === 2. Autoencoder Definition ===
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),  # Latent space
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

latent_dim = 16
model = Autoencoder(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# === 3. Training ===
for epoch in range(5):
    for batch, _ in loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# === 4. Extract Decoder and Visualize Latent Neuron Effects ===
decoder = model.decoder

def decode_latent(latent_vec):
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_vec, dtype=torch.float32).unsqueeze(0)  # shape (1, latent_dim)
        img = decoder(latent_tensor).squeeze().numpy()
    return img

n_steps = 7  # Number of values to sweep through for each neuron
vals = np.linspace(-3, 3, n_steps)

plt.figure(figsize=(n_steps, latent_dim+1))

for i in range(latent_dim):
    for j, v in enumerate(vals):
        z = np.zeros(latent_dim)
        z[i] = v
        img = decode_latent(z)
        plt.subplot(latent_dim, n_steps, i * n_steps + j + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(f"{v:.1f}")

plt.suptitle("Effect of Varying Each Latent Neuron (rows: neurons, cols: value sweep)")
plt.tight_layout()
plt.show()

# === Generate a digit "2" using its real latent code ===

# Find a real "2" in MNIST
for img_batch, label_batch in loader:
    idx = (label_batch == 2).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        img2 = img_batch[idx[0]].unsqueeze(0)  # shape (1, 1, 28, 28)
        break


# Encode to latent space
with torch.no_grad():
    latent2 = model.encoder(img2)

# Decode back
img_recon = decode_latent(latent2.squeeze().numpy())

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img2.squeeze().numpy().reshape(28, 28), cmap='gray')
plt.title("Original '2'")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_recon.reshape(28, 28), cmap='gray')
plt.title("Reconstructed '2'")
plt.axis('off')

plt.tight_layout()
plt.show()
