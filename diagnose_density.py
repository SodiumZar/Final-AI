"""
Diagnostic script to check vessel density and segmentation quality
"""
import cv2
import numpy as np
from predict import predict
from utils import calculate_vessel_density
import sys

if len(sys.argv) < 2:
    print("Usage: python diagnose_density.py <image_path>")
    print("Example: python diagnose_density.py static/uploads/fundus.jpg")
    sys.exit(1)

image_path = sys.argv[1]

print(f"Analyzing: {image_path}")
print("=" * 60)

# Get prediction
result = predict(image_path)
if isinstance(result, dict):
    mask = result['mask']
    print("⚠ Using fallback segmentation")
else:
    mask = result
    print("✓ Using U-Net model")

# Check mask statistics
print(f"\nMask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Unique values in mask: {np.unique(mask)}")
print(f"Min value: {mask.min()}, Max value: {mask.max()}")

# Calculate density
total_pixels = mask.size
vessel_pixels_0 = np.sum(mask > 0)  # Original threshold
vessel_pixels_128 = np.sum(mask > 128)  # Higher threshold (for uint8 0-255)

density_0 = (vessel_pixels_0 / total_pixels) * 100
density_128 = (vessel_pixels_128 / total_pixels) * 100

print(f"\nTotal pixels: {total_pixels:,}")
print(f"Vessel pixels (>0): {vessel_pixels_0:,} → Density: {density_0:.2f}%")
print(f"Vessel pixels (>128): {vessel_pixels_128:,} → Density: {density_128:.2f}%")

# Calculate actual density
actual_density = calculate_vessel_density(mask)
print(f"\nCalculated density: {actual_density}%")

# Assessment
print("\n" + "=" * 60)
if actual_density < 8:
    print("⚠ VERY LOW - Possible under-segmentation")
    print("Recommendations:")
    print("  1. Lower the prediction threshold (currently 0.5)")
    print("  2. Check if image preprocessing is correct")
    print("  3. Verify the model is trained on similar images")
elif actual_density < 10:
    print("⚠ LOW - Borderline")
    print("May be normal for some fundus images, but could indicate under-segmentation")
elif actual_density <= 20:
    print("✓ NORMAL - Within expected range (10-20%)")
else:
    print("⚠ HIGH - Possible over-segmentation")

# Visual check
print("\nCreating diagnostic visualization...")
original = cv2.imread(image_path)
if original is None:
    print("✗ Could not load original image")
else:
    # Create side-by-side visualization
    h, w = mask.shape
    
    # Normalize mask to 0-255 for visualization
    if mask.max() <= 1:
        mask_vis = (mask * 255).astype(np.uint8)
    else:
        mask_vis = mask.astype(np.uint8)
    
    # Resize original to match mask
    original_resized = cv2.resize(original, (w, h))
    
    # Create colored mask
    mask_colored = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 0] = 0  # Remove blue
    mask_colored[:, :, 1] = 0  # Remove green
    # Red channel stays
    
    # Create overlay
    overlay = cv2.addWeighted(original_resized, 0.7, mask_colored, 0.3, 0)
    
    # Combine all three
    combined = np.hstack([original_resized, mask_colored, overlay])
    
    # Save
    output_path = "diagnostic_output.jpg"
    cv2.imwrite(output_path, combined)
    print(f"✓ Saved visualization to: {output_path}")
    print("  (Left: Original | Middle: Mask | Right: Overlay)")
