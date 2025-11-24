"""
Test script to debug tortuosity calculation
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize

# Create a test image with a straight horizontal line
test_mask = np.zeros((200, 200), dtype=np.uint8)
test_mask[100, 20:180] = 255  # Straight horizontal line, 160 pixels long

print("Test 1: Straight horizontal line (160 pixels)")
print("Expected Euclidean distance: 160")
print("Expected pixel count after skeletonization: ~160")
print("Expected tortuosity: ~1.0")
print("-" * 50)

# Skeletonize
skeleton = skeletonize(test_mask > 0)
num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))

for label in range(1, num_labels):
    points = np.column_stack(np.where(labels == label))
    pixel_count = len(points)
    
    # Find endpoints
    max_dist = 0
    for i in range(min(20, len(points))):
        for j in range(len(points)-min(20, len(points)), len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
    
    euclidean_dist = max_dist
    raw_ratio = pixel_count / euclidean_dist if euclidean_dist > 0 else 0
    
    print(f"Pixel count: {pixel_count}")
    print(f"Euclidean distance: {euclidean_dist:.2f}")
    print(f"Raw ratio (pixel_count/euclidean_dist): {raw_ratio:.3f}")
    print()

# Test 2: Slightly curved line
test_mask2 = np.zeros((200, 200), dtype=np.uint8)
for x in range(20, 180):
    y = int(100 + 20 * np.sin((x-20) / 25))  # Sine wave
    test_mask2[y, x] = 255

print("\nTest 2: Slightly curved line (sine wave)")
print("Expected tortuosity: ~1.1-1.3")
print("-" * 50)

skeleton2 = skeletonize(test_mask2 > 0)
num_labels2, labels2 = cv2.connectedComponents(skeleton2.astype(np.uint8))

for label in range(1, num_labels2):
    points = np.column_stack(np.where(labels2 == label))
    pixel_count = len(points)
    
    # Find endpoints
    max_dist = 0
    for i in range(min(20, len(points))):
        for j in range(len(points)-min(20, len(points)), len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
    
    euclidean_dist = max_dist
    raw_ratio = pixel_count / euclidean_dist if euclidean_dist > 0 else 0
    
    print(f"Pixel count: {pixel_count}")
    print(f"Euclidean distance: {euclidean_dist:.2f}")
    print(f"Raw ratio (pixel_count/euclidean_dist): {raw_ratio:.3f}")
