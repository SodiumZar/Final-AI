"""
Image validation module to detect if uploaded image is a retina/fundus image
"""
import cv2
import numpy as np


def is_retina_image(image_path):
    """
    Validate if the uploaded image is likely a retina/fundus image.
    Uses heuristics like circular shape detection, color characteristics, etc.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (is_valid, confidence_score, reason)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, "Could not read image"
        
        h, w = img.shape[:2]
        
        # Check 1: Aspect ratio (fundus images are usually square-ish or circular)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.0:
            return False, 0.2, "Image aspect ratio too extreme (not circular/square)"
        
        # Check 2: Color distribution (fundus images have red/orange tones)
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Fundus images typically have:
        # - Hue: Red/Orange (0-30 or 160-180)
        # - Saturation: Medium to high
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Count red/orange pixels
        red_orange_mask = ((h_channel < 30) | (h_channel > 160)) & (s_channel > 30)
        red_orange_ratio = np.sum(red_orange_mask) / (h * w)
        
        if red_orange_ratio < 0.1:
            return False, 0.3, "Image lacks typical fundus red/orange coloring"
        
        # Check 3: Circular region detection (fundus images have circular field of view)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find the bright circular region
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                # Circularity = 4π * area / perimeter²
                # Perfect circle = 1.0
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity < 0.4:
                    return False, 0.4, "Image lacks circular field of view typical of fundus images"
        
        # Check 4: Central brightness (optic disc area)
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        mean_brightness_center = np.mean(center_region)
        mean_brightness_edges = np.mean(gray[0:h//10, :]) + np.mean(gray[-h//10:, :])
        mean_brightness_edges /= 2
        
        # Fundus images typically have brighter center (optic disc)
        if mean_brightness_center < mean_brightness_edges:
            return False, 0.5, "Image brightness pattern doesn't match fundus characteristics"
        
        # Check 5: Dark borders/vignetting (common in fundus images)
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        mean_border_brightness = np.mean(border_pixels)
        
        if mean_border_brightness > 100:  # Borders should be relatively dark
            return False, 0.6, "Missing typical fundus vignetting/dark borders"
        
        # All checks passed
        confidence = 0.85 + (red_orange_ratio * 0.15)  # Boost confidence based on color
        return True, min(confidence, 1.0), "Image appears to be a retina/fundus image"
    
    except Exception as e:
        return False, 0.0, f"Validation error: {str(e)}"


def validate_image_for_analysis(image_path, min_confidence=0.7):
    """
    Wrapper function to validate image with a confidence threshold.
    
    Args:
        image_path: Path to image
        min_confidence: Minimum confidence score to accept (default 0.7)
        
    Returns:
        tuple: (is_valid, message)
    """
    is_valid, confidence, reason = is_retina_image(image_path)
    
    if not is_valid:
        return False, f"⚠️ This doesn't appear to be a retina/fundus image. {reason}"
    
    if confidence < min_confidence:
        return False, f"⚠️ Low confidence ({confidence:.0%}) that this is a fundus image. {reason}"
    
    return True, f"✓ Valid fundus image (confidence: {confidence:.0%})"
