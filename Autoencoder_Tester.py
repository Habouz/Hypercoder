import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np


latent_dim = 32
n_steps = 10



encoder = torch.load('encoder_full.pth')
decoder = torch.load('decoder_full.pth')
print("Encoder architecture:")
print(encoder)
print("\nDecoder architecture:")
print(decoder)
encoder.eval()
decoder.eval()

# === 1. Data Loading (MNIST) ===
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=256, shuffle=True)

for img_batch, label_batch in loader:
    idx = (label_batch == 5).nonzero(as_tuple=True)[0]
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
plt.imshow(img.squeeze().numpy(), cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(recon.squeeze().numpy(), cmap='gray')
plt.axis('off')
plt.show()

vals = np.linspace(0, 20, n_steps)

# for i in range(latent_dim):
#     for j, v in enumerate(vals):
#         z = torch.zeros(latent_dim)
#         z[i] = v
#         # Add batch dimension
#         img = decoder(z.unsqueeze(0))         # Shape: (1, 1, 28, 28)
#         img = img.squeeze().detach().numpy()  # Shape: (28, 28)
#         plt.subplot(latent_dim, n_steps, i * n_steps + j + 1)
#         plt.imshow(img, cmap='gray')
#         plt.axis('off')
#         if i == 0:
#             plt.title(f"{v:.1f}")
# plt.show()

while True:
    mask = torch.tensor([11, 13, 15, 17, 18, 19, 20, 21, 24, 25, 27, 28, 29, 30, 31], dtype=torch.long)
    my_input = torch.zeros(latent_dim, dtype=torch.float)
    my_input[mask] = ((torch.rand(len(mask))) * 10).float()  # Ensure source is torch.long
    my_input[8] = 30
    # my_input[2] = 0
    # my_input[6] = 0
    # my_input[torch.tensor([0,1,2,3,4,5,6,7,8,9])] += torch.rand(10)

    print("Custom input latent vector:", my_input)

    my_input = my_input.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        my_recon = decoder(my_input)
    plt.figure(figsize=(4, 4))
    plt.imshow(my_recon.squeeze().numpy(), cmap='gray')
    plt.title("Custom Input Reconstruction")
    plt.axis('off')
    plt.show()