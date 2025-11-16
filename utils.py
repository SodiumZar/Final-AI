import os
import random
import numpy as np
import torch
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

""" Function to seed the randomness """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Function to create a new directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Function to calculate the taken time """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


""" Function to create overlay visualization """
def create_overlay(original_image, mask):
    """
    Create an overlay of the segmented vessels on the original image
    Args:
        original_image: Original retina image (numpy array)
        mask: Binary segmentation mask (numpy array)
    Returns:
        overlay: RGB image with vessels highlighted
    """
    # Ensure original image is in RGB format
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create colored mask (red vessels)
    overlay = original_image.copy().astype(np.float32)
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = mask * 255  # Red channel
    
    # Blend images
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1-alpha, red_mask, alpha, 0)
    overlay = overlay.astype(np.uint8)
    
    return overlay


""" Function to calculate vessel density """
def calculate_vessel_density(mask):
    """
    Calculate the percentage of vessel pixels in the image
    Args:
        mask: Binary segmentation mask (numpy array)
    Returns:
        density: Vessel density as percentage
    """
    total_pixels = mask.size
    vessel_pixels = np.sum(mask > 0)
    density = (vessel_pixels / total_pixels) * 100
    return round(density, 2)


""" Function to calculate tortuosity score """
def calculate_tortuosity(mask):
    """
    Calculate tortuosity score based on vessel curvature
    Args:
        mask: Binary segmentation mask (numpy array)
    Returns:
        tortuosity_score: Average tortuosity measure
    """
    # Skeletonize the mask to get vessel centerlines
    skeleton = skeletonize(mask > 0)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    
    tortuosity_scores = []
    
    for label in range(1, num_labels):
        # Get points for this vessel
        points = np.column_stack(np.where(labels == label))
        
        if len(points) < 10:  # Skip very short segments
            continue
        
        # Calculate path length (actual vessel length)
        path_length = 0
        for i in range(len(points) - 1):
            path_length += np.linalg.norm(points[i] - points[i+1])
        
        # Calculate Euclidean distance (straight line)
        euclidean_dist = np.linalg.norm(points[0] - points[-1])
        
        # Avoid division by zero
        if euclidean_dist > 0:
            tortuosity = path_length / euclidean_dist
            tortuosity_scores.append(tortuosity)
    
    if tortuosity_scores:
        avg_tortuosity = np.mean(tortuosity_scores)
        return round(avg_tortuosity, 3)
    else:
        return 1.0  # Default value if no vessels detected


""" Function to analyze vessel characteristics """
def analyze_vessels(mask):
    """
    Comprehensive vessel analysis
    Args:
        mask: Binary segmentation mask (numpy array)
    Returns:
        analysis: Dictionary with various vessel metrics
    """
    analysis = {
        'vessel_density': calculate_vessel_density(mask),
        'tortuosity_score': calculate_tortuosity(mask),
        'total_vessel_pixels': int(np.sum(mask > 0)),
        'image_size': f"{mask.shape[0]}x{mask.shape[1]}"
    }
    
    return analysis
