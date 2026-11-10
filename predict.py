import torch
import torch.nn.functional as F

from torchvision import models
from torchvision.transforms import transforms
from PIL import Image

classes = [
'10',
'20',
'50',
'100',
'200',
'500',
'2000'
]

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    7
)

model.load_state_dict(
    torch.load(
        "models/currency_quantum.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("test.jpg").convert("RGB")

image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    probs = F.softmax(output,dim=1)

    confidence,pred = torch.max(probs,1)

print(f"Currency: ₹{classes[pred.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")
