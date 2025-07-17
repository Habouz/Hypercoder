import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np


latent_dim = 256

encoder = torch.load('encoder_full_cifar10.pth')
decoder = torch.load('decoder_full_cifar10.pth')
print("Encoder architecture:")
print(encoder)
print("\nDecoder architecture:")
print(decoder)
encoder.eval()
decoder.eval()

# === 1. Data Loading (MNIST) ===
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=True)

for img_batch, label_batch in loader:
    idx = (label_batch == 4).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        img = img_batch[idx[0]].unsqueeze(0)
        break

# === 2. Encode and Decode an Image ===
with torch.no_grad():
    latent = encoder(img)
    print("Latent representation shape:", latent.shape)
    print("Latent representation:", latent)
    recon = decoder(latent)

# === 3. Visualize the Original and Reconstructed Images ===
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img.squeeze().numpy().transpose(1, 2, 0))  # Convert from CHW to HWC format
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(recon.squeeze().numpy().transpose(1, 2, 0))  # Convert from CHW to HWC format
plt.axis('off')
plt.show()
