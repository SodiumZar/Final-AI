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
    Uses the ratio of skeleton pixel count to endpoint distance
    Args:
        mask: Binary segmentation mask (numpy array)
    Returns:
        tortuosity_score: Normalized tortuosity (1.0 = straight, >1.3 = tortuous)
    """
    # Clean up the mask first to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Skeletonize the cleaned mask to get vessel centerlines
    skeleton = skeletonize(mask_cleaned > 0)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    
    tortuosity_scores = []
    
    for label in range(1, num_labels):
        # Get points for this vessel segment
        points = np.column_stack(np.where(labels == label))
        
        if len(points) < 50:  # Only process substantial vessels
            continue
        
        # Find the two farthest points (true endpoints)
        max_dist = 0
        
        # Sample points to find true endpoints efficiently
        sample_size = min(30, len(points))
        sample_indices = np.linspace(0, len(points)-1, sample_size, dtype=int)
        
        for i in sample_indices:
            for j in sample_indices:
                if i < j:
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist > max_dist:
                        max_dist = dist
        
        euclidean_dist = max_dist
        
        # Only process longer vessels (at least 80 pixels for reliability)
        if euclidean_dist < 80:
            continue
        
        # Calculate raw ratio: pixel_count / euclidean_dist
        # For straight vessels: ~1.0
        # For curved vessels: >1.0 (higher = more curved)
        pixel_count = len(points)
        raw_ratio = pixel_count / euclidean_dist
        
        # Only include reasonable values (filter out artifacts)
        # Tortuosity cannot be < 1.0 (path cannot be shorter than straight line)
        if 1.0 <= raw_ratio < 2.5:  # Valid range
            tortuosity_scores.append(raw_ratio)
    
    if len(tortuosity_scores) >= 3:  # Need at least 3 vessels for reliable measurement
        # Sort and exclude top 20% outliers
        tortuosity_scores.sort()
        cutoff_index = int(len(tortuosity_scores) * 0.8)
        filtered_scores = tortuosity_scores[:cutoff_index]
        
        # Use median of filtered scores
        avg_tortuosity = np.median(filtered_scores)
        # Ensure result is at least 1.0 (cannot be shorter than straight line)
        return round(max(1.0, avg_tortuosity), 3)
    elif tortuosity_scores:
        # If we have 1-2 vessels, just use median
        avg_tortuosity = np.median(tortuosity_scores)
        return round(max(1.0, avg_tortuosity), 3)
    else:
        return 1.0  # Default for no vessels detected


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
