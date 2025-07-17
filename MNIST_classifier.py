import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data loaders
transform = transforms.ToTensor()
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
testset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 3. Model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

model = MNISTClassifier().to(device)

# 4. Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 5. Training loop
epochs = 30
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
torch.save(model.net, 'mnist_classifier_entire.pth')
print("Model saved as 'mnist_classifier_entire.pth'")
