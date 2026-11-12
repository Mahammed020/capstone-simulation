import streamlit as st
import torch
import torch.nn.functional as F

from torchvision import models
from torchvision.transforms import transforms
from PIL import Image

classes = [
    '10',
    '100',
    '20',
    '200',
    '2000',
    '50',
    '500'
]

@st.cache_resource
def load_model():

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

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

st.title("Quantum Enhanced Currency Recognition")

uploaded = st.file_uploader(
    "Upload Currency Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(
        uploaded
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Currency Note",
        width=500
    )

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = F.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    denomination = classes[
        pred.item()
    ]

    st.success(
        f"Detected Currency: ₹{denomination}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )
