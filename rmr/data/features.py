"""Feature extraction helpers for text and visual modalities."""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def extract_text_features_dummy(
    texts: list[str], dim: int = 384
) -> np.ndarray:
    """Return random normalized vectors as placeholder text features.

    Args:
        texts: List of raw text strings.
        dim: Dimensionality of the random embeddings.

    Returns:
        An array of shape (len(texts), dim) with L2-normalized rows.
    """
    rng = np.random.default_rng(42)
    feats = rng.normal(size=(len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


def extract_visual_features_dummy(
    image_paths: list[str], dim: int = 4096
) -> np.ndarray:
    """Return random normalized vectors as placeholder visual features.

    Args:
        image_paths: List of image file paths.
        dim: Dimensionality of the random embeddings.

    Returns:
        An array of shape (len(image_paths), dim) with L2-normalized rows.
    """
    rng = np.random.default_rng(43)
    feats = rng.normal(size=(len(image_paths), dim)).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


def extract_text_features_st(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Extract text features using sentence-transformers.

    Args:
        texts: List of raw text strings.
        model_name: Hugging Face model identifier for the encoder.

    Returns:
        An array of shape (len(texts), encoder_dim) with dtype float32.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def extract_visual_features_resnet(
    image_paths: list[str],
) -> np.ndarray:
    """Extract visual features using a pretrained ResNet-50.

    The final fully-connected layer is replaced with an identity mapping so
    that the model outputs the 2048-dim penultimate activations.

    Args:
        image_paths: List of image file paths.

    Returns:
        An array of shape (len(image_paths), 2048) with dtype float32.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    feats = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            f = model(x).cpu().numpy().squeeze()
            feats.append(f)
    return np.stack(feats, axis=0).astype(np.float32)


def save_features(features: np.ndarray, path: str) -> None:
    """Persist a NumPy array to disk.

    Args:
        features: The array to save.
        path: Target file path (``.npy`` extension recommended).

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, features)


def load_features(path: str) -> np.ndarray:
    """Load a NumPy array from disk.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        The loaded array.
    """
    return np.load(path)
