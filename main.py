import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from Distance_Distribution import path_entropy_parallel
from Layer_Finder import find_layer_with_least_effective
from Entropic_Loss import entropic_loss, cumm_entropic_loss

np.set_printoptions(linewidth=200)
classifier = torch.load('mnist_classifier_entire.pth')


# === 1. Data Loading (MNIST) ===
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=256, shuffle=True)

# === 2. Autoencoder Definition ===
class Autoencoder(nn.Module):
    def __init__(self, extended_latent_dim=20):
        super().__init__()
        self.extended_latent_dim = extended_latent_dim
        self.autoencoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, extended_latent_dim),
            nn.ReLU(),
            nn.Linear(extended_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        recon = self.autoencoder(x)
        return recon
    def half_forward(self, x):
        return nn.Sequential(*list(self.autoencoder)[:7])(x)


def classifier_loss(x, y):
    
    logits_x = classifier(x)
    logits_x = nn.functional.softmax(logits_x, dim=1)

    logits_y = classifier(y)
    logits_y = nn.functional.softmax(logits_y, dim=1)
    
    return nn.functional.kl_div(logits_y.log(), logits_x, reduction='batchmean')


if __name__ == "__main__":
    extended_latent_dim = 32
    model = Autoencoder(extended_latent_dim=extended_latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    paths = None
    criterion_term = 1.0
    entropy_term = 0.5
    cross_entropy_term = 0.0
    turnon = 0.0

    # === 3. Training ===
    for epoch in range(15):
        doprint = True
        if epoch == 8: 
            turnon = 1.0
            print(f"Turned on classifier loss at epoch {epoch+1}")
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
            eloss = entropic_loss(W_l, W_r, B, alpha=1.0, beta = 0.04) # Achieved optimal results with alpha = 1.0, beta = 0.04

            # Cross Entropy
            classification = model.half_forward(batch)[:,:10]
            classification_loss = cross_entropy(classification, labels) #TODO: Not making the latent space predictible yet. Needs more work.

            # Reconstruction Loss
            recon = model(batch)
            criterion_loss = turnon*classifier_loss(batch, recon) + criterion(recon, batch) # Reconstruction loss + classifier loss
            loss = criterion_term * criterion_loss + entropy_term * eloss + cross_entropy_term * classification_loss # Entire Loss Function
            loss.backward()
            optimizer.step()
            doprint = False

        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}, Entropy Loss = {eloss.item():.6f}, Criterion Loss = {criterion_loss.item():.6f}, Cross Entropy Loss = {classification_loss.item():.6f}")

        print([torch.max(weight).item() for weight in weight_list]) # Print max weight of each layer (debugging)


    # === 4. Extract Decoder and Visualize Latent Neuron Effects ===
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



    split_idx = 7           # Specified here, but could be dynamically determined
    encoder = nn.Sequential(*list(model.autoencoder)[:split_idx])
    decoder = nn.Sequential(*list(model.autoencoder)[split_idx:])


    # Usage example
    sample, _ = next(iter(loader))
    sample = sample[:1]  # Just one image

    with torch.no_grad():
        latent_code = encoder(sample)
        recon_img = decoder(latent_code)
        plt.imshow(recon_img.squeeze().numpy().reshape(28,28), cmap='gray')
        plt.axis('off')
        plt.show()

    # === Save ===
    torch.save(encoder, "encoder_full.pth")
    torch.save(decoder, "decoder_full.pth")