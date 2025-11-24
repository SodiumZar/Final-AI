import cv2
import numpy as np
import torch
import pickle
import os

from model import build_unet


def _classical_vessel_segmentation(image_path: str) -> np.ndarray:
    """
    Classical fallback segmentation using image processing when model weights fail to load.
    Returns a binary mask (0/1) with vessel-like structures.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Use green channel which best contrasts vessels
    green = img[:, :, 1]

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Smooth and threshold (vessels are darker -> invert threshold)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = (closed > 0).astype(np.uint8)
    return mask


def predict(image_path, checkpoint_path="files/checkpoint_compatible.pth", threshold=0.1):
    """
    Predict vessel mask for the given image.
    Args:
        image_path: Path to the retina image
        checkpoint_path: Path to model checkpoint
        threshold: Prediction threshold (default 0.1, lower = more vessels detected)
    Returns either a binary mask (np.uint8 0/1) or a dict { 'mask': np.ndarray, 'warning': str } when fallback is used.
    """
    # Try model-based prediction first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = build_unet().to(device)
        # Try loading checkpoint with weights_only compatibility
        state = None
        try:
            # Try weights_only=True first (PyTorch 2.6+ default)
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError) as e:
            # If weights_only=True fails, try weights_only=False
            # This allows loading older checkpoints with custom objects
            try:
                state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                # Older PyTorch versions don't support weights_only parameter
                state = torch.load(checkpoint_path, map_location=device)

        if state is None:
            raise RuntimeError("Failed to load checkpoint")

        if isinstance(state, dict):
            # Load as state dict (may be under 'state_dict')
            state_dict = state.get('state_dict', state)
            model.load_state_dict(state_dict, strict=False)
        else:
            # Unexpected format
            raise RuntimeError("Checkpoint format not recognized as state dict")

        model.eval()

        # Reading image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # (H, W, 3)
        x = np.transpose(image, (2, 0, 1))  # (3, H, W)
        x = x / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)  # (1, 3, H, W)
        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            pred_y = model(x_t)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()  # (1, H, W)
            pred_y = np.squeeze(pred_y, axis=0)  # (H, W)
            # Use configurable threshold (default 0.3 for better vessel detection)
            pred_y = (pred_y > threshold).astype(np.uint8)

        return pred_y

    except Exception as e:
        # Fallback path using classical image processing
        warning = (
            "Model checkpoint couldn't be loaded (" + str(e) + "). "
            "Using classical image-processing fallback for segmentation. Results may be less accurate."
        )
        try:
            mask = _classical_vessel_segmentation(image_path)
            return {"mask": mask, "warning": warning}
        except Exception as e2:
            raise RuntimeError(f"Segmentation failed: {e2}") from e2
