import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import sys
import os

# --- 1. Model Architecture ---
# This is the exact same ResNet9 model architecture from your notebook.
# It's required here so PyTorch can unpickle the saved model object correctly.


def conv_block(in_channels, out_channels, pool=False):
    """Convolutional block with batch normalization and ReLU activation."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """ResNet9 architecture for image classification."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        """Forward pass through the network."""
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# --- 2. Configuration and Class Names ---
# List of all possible diseases the model can predict.
# This must be in the same order as the training data.
class_names = [
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

# Set the device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Path to your trained model
MODEL_PATH = (
    "./models/20250928-165707-resnet9-plant-disease-classifier-model-complete.pth"
)


# --- 3. Prediction Function ---
def predict_image(image_path, model, class_names):
    """
    Loads an image, preprocesses it, and returns
    (predicted_class, [(class_name, probability_float), ...] sorted desc).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.Resize((256, 256)),  # training images were 256x256
                T.ToTensor(),  # no normalization used during training
            ]
        )
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]
            top_prob, predicted_idx = torch.max(probs, 0)

        predicted_class = class_names[predicted_idx.item()]
        probs_list = [
            (class_names[i], probs[i].item()) for i in range(len(class_names))
        ]
        probs_list.sort(key=lambda x: x[1], reverse=True)

        return predicted_class, probs_list

    except FileNotFoundError:
        return f"Error: The file at {image_path} was not found.", None
    except Exception as e:
        return f"An error occurred: {e}", None


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    test_directory = "./test"

    # Check if the test directory exists
    if not os.path.isdir(test_directory):
        print(f"Error: Test directory not found at '{test_directory}'")
        print("Please create a directory named 'test' and put your images inside it.")
        sys.exit(1)

    # Load the entire model object
    try:
        # Since the model file contains the entire object, we pass `weights_only=False`
        # to allow PyTorch to unpickle the ResNet9 class structure from the file.
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print(
            f"Please make sure 'resnet9-plant-disease-classifier-model-complete.pth' is at {MODEL_PATH}."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    # Move model to the correct device
    model.to(device)
    # Set the model to evaluation mode
    model.eval()

    # Find all image files in the test directory
    image_files = [
        f
        for f in os.listdir(test_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No images found in the '{test_directory}' directory.")
        sys.exit(0)

    print(f"Found {len(image_files)} images to test in '{test_directory}'.")
    print("-" * 30)

    # Loop through all found images and make predictions
    for image_file in image_files:
        image_path_to_test = os.path.join(test_directory, image_file)

        # Get the prediction and probabilities
        prediction, probs_list = predict_image(image_path_to_test, model, class_names)

        if probs_list is None:
            print(f"Image: {image_file}")
            print(prediction)
            print("-" * 30)
            continue

        print(f"Image: {image_file}")
        print(f"Top-1 prediction: {prediction}  | confidence: {probs_list[0][1]:.2%}")
        print("Per-class confidence (>= 1%):")
        header = f"{'#':>3}  {'confidence':>12}  class"
        print(header)
        print("-" * len(header))
        shown = False
        for rank, (cls, p) in enumerate(probs_list, start=1):
            if p < 0.01:  # hide confidences under 1%
                continue
            print(f"{rank:>3}  {p:>10.2%}  {cls}")
            shown = True
        if not shown:
            print("  (none >= 1%)")
        print("-" * 30)
