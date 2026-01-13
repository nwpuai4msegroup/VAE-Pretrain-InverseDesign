import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# Gradient Reversal Layer for DANN
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


# DANN Model
class DANN(nn.Module):
    def __init__(self, feature_dim=128):
        super(DANN, self).__init__()
        # Feature Extractor (CNN for images)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )
        # Domain Classifier (binary: synthetic=0, real=1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=0.0):
        features = self.feature_extractor(x)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        return features, domain_pred

    def extract_features(self, x):
        """Extract features without domain prediction."""
        with torch.no_grad():
            features, _ = self(x, alpha=0.0)
            return features.cpu().numpy().flatten()


# Global transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_input, checkpoint_path=None, device=None):
    """
    Extract features from an image using the loaded DANN model.

    Args:
        image_input (str or PIL.Image): Path to the input image or PIL Image object.
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to 'dann_results/dann_checkpoint_epoch_95.pth'.
        device (torch.device, optional): Device to use. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        np.ndarray: Flattened feature vector (128-dim).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        checkpoint_path = os.path.join('dann_checkpoint_epoch_95.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model = DANN(feature_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and preprocess image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        # Assume it's a PIL Image
        img = image_input.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract features
    features = model.extract_features(img_tensor)

    return features
