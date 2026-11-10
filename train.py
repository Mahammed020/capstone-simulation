import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = ImageFolder(
    "dataset",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))

# Data Loader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# Model (No Internet Download)
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    7
)

device = torch.device("cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "models/currency_quantum.pth"
)

print("Model Saved Successfully")
