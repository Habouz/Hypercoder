import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from layer_finder import find_layer_with_least_effective
from entropic_losses import entropic_loss, cumm_entropic_loss

np.set_printoptions(linewidth=200)

# === 1. Data Loading (CIFAR10) ===
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=True)

# === 2. Autoencoder Definition ===
class Autoencoder(nn.Module):
    def __init__(self, extended_latent_dim=20):
        super().__init__()
        self.extended_latent_dim = extended_latent_dim
        self.autoencoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*32*32),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 32, 32))
        )
    def forward(self, x):
        recon = self.autoencoder(x)
        return recon
    def half_forward(self, x):
        return nn.Sequential(*list(self.autoencoder)[:7])(x)



if __name__ == "__main__":
    extended_latent_dim = 256
    model = Autoencoder(extended_latent_dim=extended_latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    paths = None
    label_number = 10
    criterion_term = 1.0
    entropy_term = 0.1
    cross_entropy_term = 0.0

    # === 3. Training ===
    for epoch in range(10):
        doprint = True
        for batch, labels in loader:
            optimizer.zero_grad()


            # Extract Layers
            linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
            weight_list = [layer.weight for layer in linear_layers]
            bias_list = [layer.bias for layer in linear_layers] if doprint else None

            
            # Compute Entropy

            W_l = linear_layers[2].weight
            W_r = linear_layers[3].weight
            B = linear_layers[2].bias
            eloss = entropic_loss(W_l, W_r, B, alpha=1.0, beta=1) # Achieved optimal results with alpha = 1.0, beta = 0.04

            # Cross Entropy
            # classification = model.half_forward(batch)[:,:label_number]
            classification_loss = torch.tensor(0) # cross_entropy(classification, labels) #TODO: Not making the latent space predictible yet. Needs more work.

            # Scarcity Loss
            l_1 = torch.tensor(0.0)
            for weight in weight_list:
                l_1 += torch.norm(weight, p=1)

            # Reconstruction Loss
            recon = model(batch)
            criterion_loss = criterion(recon, batch)
            loss = criterion_term * criterion_loss + entropy_term * eloss + cross_entropy_term * classification_loss # Entire Loss Function
            loss.backward()
            optimizer.step()
            doprint = False

        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}, Entropy Loss = {eloss.item():.6f}, Criterion Loss = {criterion_loss.item():.6f}, Cross Entropy Loss = {classification_loss.item():.6f}")

        print([torch.max(weight).item() for weight in weight_list]) # Print max weight of each layer (debugging)


    # === 4. Visualize Latent Neuron Effects ===
    layers = [
        (layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
        for layer in model.modules() if isinstance(layer, nn.Linear)
    ]
    smallest = find_layer_with_least_effective(layers, threshold=1e-3)
    print('Smallest layer:', smallest)

    # Print weights of 2nd and 3rd linear layers (index 2 and 3 in linear_layers)
    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    w2 = linear_layers[2].weight.detach().cpu().numpy()
    b2 = linear_layers[2].bias.detach().cpu().numpy()
    w3 = linear_layers[3].weight.detach().cpu().numpy()
    b3 = linear_layers[3].bias.detach().cpu().numpy()

    print("Weights of 2nd Linear Layer:\n", np.array2string(w2, precision=3, suppress_small=True))
    print("Average weight of 2nd Linear Layer:\n", np.array2string(np.mean(np.abs(w2), axis=0), precision=3, suppress_small=True))
    print("Bias of 2nd Linear Layer:\n", np.array2string(b2, precision=3, suppress_small=True))
    print("Weights of 3rd Linear Layer:\n", np.array2string(w3, precision=3, suppress_small=True))
    print("Average weight of 3rd Linear Layer:\n", np.array2string(np.mean(np.abs(w3), axis=1), precision=3, suppress_small=True))
    print("Bias of 3rd Linear Layer:\n", np.array2string(b3, precision=3, suppress_small=True))


    #=== 5. Save Encoder and Decoder ===

    split_idx = 7           # Specified here, but could be dynamically determined
    encoder = nn.Sequential(*list(model.autoencoder)[:split_idx])
    decoder = nn.Sequential(*list(model.autoencoder)[split_idx:])


    #=== 6. Visualize Encoder and Decoder ===

    sample, _ = next(iter(loader))
    sample = sample[:1]  # Just one image

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(sample.squeeze().numpy().transpose(1, 2, 0))  # CIFAR-10 images are RGB
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    with torch.no_grad():
        latent_code = encoder(sample)
        recon_img = decoder(latent_code)
        plt.imshow(recon_img.numpy().reshape(3,32,32).transpose(1, 2, 0), cmap='gray')
        plt.axis('off')
        plt.show()

    # === Save ===
    torch.save(encoder, "encoder_full_cifar10.pth")
    torch.save(decoder, "decoder_full_cifar10.pth")