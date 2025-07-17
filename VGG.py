import torch
import torch.nn as nn
import torchvision.models as models

# --- VGG Feature Extractor for Perceptual Loss ---
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer='relu2_2'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Select cutoff layer (by convention)
        cutoff = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
        }[layer]
        self.slice = nn.Sequential(*[vgg[i] for i in range(cutoff)])
        # VGG expects normalized images (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.slice(x)

# --- VGG Perceptual Loss ---
class VGGLoss(nn.Module):
    def __init__(self, layer='relu2_2'):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer)
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        f_out = self.vgg(output)
        f_tgt = self.vgg(target)
        return self.criterion(f_out, f_tgt)

# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dummy data: batch of 4 CIFAR-like images
    img1 = torch.rand(4, 3, 32, 32, device=device)
    img2 = torch.rand(4, 3, 32, 32, device=device)

    # Instantiate loss
    perceptual_loss = VGGLoss(layer='relu2_2').to(device)
    # Compute perceptual loss
    loss = perceptual_loss(img1, img2)
    print(f"VGG perceptual loss: {loss.item():.6f}")
