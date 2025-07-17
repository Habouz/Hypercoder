import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data loaders
transform = transforms.ToTensor()
trainset = datasets.CIFAR10('./data_cifar10', train=True, download=True, transform=transform)
testset  = datasets.CIFAR10('./data_cifar10', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 3. Model
class CIFAR10_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [batch, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # [batch, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # [batch, 128, 8, 8]
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                 # [batch, 128*8*8]
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
model = CIFAR10_Classifier().to(device)

# 4. Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 5. Training loop
epochs = 1
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Train loss: {total_loss / len(trainloader):.4f}")

# 6. Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"Test accuracy: {correct / total * 100:.2f}%")

# 7. Save the entire model
torch.save(model, 'cifar10_classifier_entire.pth')
print("Model saved as 'cifar10_classifier_entire.pth'")











