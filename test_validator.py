"""
Test the image validator with different types of images
"""
from image_validator import validate_image_for_analysis
import sys

if len(sys.argv) < 2:
    print("Usage: python test_validator.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

print(f"Testing: {image_path}")
print("=" * 60)

is_valid, message = validate_image_for_analysis(image_path, min_confidence=0.6)

print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
print(f"Message: {message}")
print("=" * 60)

if is_valid:
    print("\n✓ This image will be processed for vessel segmentation")
else:
    print("\n✗ This image will be rejected - user will see error message")
