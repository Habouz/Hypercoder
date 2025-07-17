import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from VGG import VGGLoss

# === 1. Data Loading (CIFAR) ===
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=True)

# === 2. Autoencoder Definition ===
class CIFAR10_Autoencoder(nn.Module):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: reduce spatial dimensions, increase channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 4, 4]
            nn.ReLU(),
        )
        # Linear autoencoder
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128*4*4),
            nn.Sigmoid(),
            nn.Unflatten(1,(128, 4, 4))
        )
        # Decoder: mirror the encoder using ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),  # [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),   # [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1, padding=1),    # [B, 3, 32, 32]
            nn.Sigmoid()  # scale output to [0, 1] for images
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x

latent_dim = 1024
model = CIFAR10_Autoencoder(latent_dim = latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = VGGLoss()

# === 3. Training ===
for epoch in range(20):
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

# === Generate a digit "2" using its real latent code ===

# CIFAR-10: "cat" is class index 3
cat_label = 3

for img_batch, label_batch in loader:
    idx = (label_batch == cat_label).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        img_cat = img_batch[idx[0]].unsqueeze(0)  # shape: (1, 3, 32, 32)
        break


# Encode to latent space
with torch.no_grad():
    latent2 = model.encoder(img_cat)

# Decode back
img_recon = decode_latent(latent2.squeeze().numpy())

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
img_orig = img_cat.squeeze().numpy().reshape(3, 32, 32).transpose(1, 2, 0)  # (3,32,32) → (32,32,3)
plt.imshow(img_orig)
plt.title("Original 'cat'")
plt.axis('off')

plt.subplot(1, 2, 2)
img_recon_disp = img_recon.reshape(3, 32, 32).transpose(1, 2, 0)  # (3,32,32) → (32,32,3)
plt.imshow(img_recon_disp)
plt.title("Reconstructed 'cat'")
plt.axis('off')

plt.tight_layout()
plt.savefig("cat_reconstruction.png")
plt.show()
