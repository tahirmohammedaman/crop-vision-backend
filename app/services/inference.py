import os
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same class order as training
class_names: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def ConvBlock(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    # EXACTLY as in the notebook: use MaxPool2d(4) when pooling
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # 256 -> 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # 64 -> 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 16 -> 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4 -> 1
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


_model: Optional[nn.Module] = None
_loaded_path: Optional[str] = None


def _build_model() -> nn.Module:
    model = ResNet9(3, len(class_names))
    model.to(device).eval()
    return model


def load_model(model_path: str):
    global _model, _loaded_path
    state = torch.load(model_path, map_location=device)  # weights file (.pth)
    model = _build_model()
    model.load_state_dict(state, strict=True)
    _model = model
    _loaded_path = model_path


def ensure_loaded(model_path: str):
    global _model, _loaded_path
    if _model is None or _loaded_path != model_path:
        load_model(model_path)


# Same preprocessing as used in the notebook/test script
_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])


def predict_pil(image: Image.Image) -> Tuple[str, List[float], float, int]:
    if _model is None:
        raise RuntimeError("Model not loaded")
    xb = _transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(xb)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, idx = torch.max(probs, 0)
    return (
        class_names[idx.item()],
        probs.tolist(),
        float(top_prob.item()),
        int(idx.item()),
    )
