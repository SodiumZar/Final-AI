"""
Test script to verify all components are working correctly
"""
import sys
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from skimage.morphology import skeletonize
        print("✓ scikit-image imported successfully")
    except ImportError as e:
        print(f"✗ scikit-image import failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False
    
    try:
        from google import genai
        print("✓ google-genai imported successfully")
    except ImportError as e:
        print(f"✗ google-genai import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    return True

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import calculate_vessel_density, calculate_tortuosity, create_overlay
        
        # Create a simple test mask
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[40:60, 40:60] = 1  # 20x20 square
        
        # Test vessel density
        density = calculate_vessel_density(test_mask)
        expected_density = 4.0  # 400 pixels out of 10000
        print(f"✓ Vessel density calculated: {density}% (expected: {expected_density}%)")
        
        # Test tortuosity
        tortuosity = calculate_tortuosity(test_mask)
        print(f"✓ Tortuosity calculated: {tortuosity}")
        
        # Test overlay
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        overlay = create_overlay(test_image, test_mask)
        print(f"✓ Overlay created with shape: {overlay.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retina_analyzer():
    """Test retina analyzer initialization"""
    print("\nTesting retina analyzer...")
    
    try:
        from retina_analyzer import RetinaAnalyzer
        analyzer = RetinaAnalyzer()
        print(f"✓ RetinaAnalyzer initialized with model: {analyzer.model}")
        return True
    except Exception as e:
        print(f"✗ RetinaAnalyzer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("Retina Analysis System - Component Test")
    print("="*50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Utility Functions", test_utils()))
    results.append(("Retina Analyzer", test_retina_analyzer()))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nTo start the application, run:")
        print("  python app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
