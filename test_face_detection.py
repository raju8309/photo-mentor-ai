#!/usr/bin/env python3
"""
Test script to verify improved face detection capabilities.
Run this to test face detection with various scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np
from app.services.analysis_service import analyze_frame, detect_faces_robust

def test_face_detection():
    """Test face detection with various scenarios."""
    
    print("🔍 Testing Improved Face Detection")
    print("=" * 50)
    
    # Test 1: Empty frame
    print("\n1. Testing empty frame...")
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyze_frame(empty_frame)
    print(f"   Faces detected: {result['num_faces']}")
    print(f"   Face hint: {result['face_hint']}")
    
    # Test 2: Frame with face-like pattern
    print("\n2. Testing frame with face-like pattern...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 100, 100)  # Gray background
    
    # Add face-like region with better features
    x, y, w, h = 250, 180, 140, 160
    cv2.rectangle(test_frame, (x, y), (x+w, y+h), (180, 180, 180), -1)  # Face
    
    # Add eyes
    cv2.circle(test_frame, (x+45, y+60), 20, (80, 80, 80), -1)  # Left eye
    cv2.circle(test_frame, (x+95, y+60), 20, (80, 80, 80), -1)  # Right eye
    
    # Add nose
    cv2.ellipse(test_frame, (x+70, y+90), (15, 25), 0, 0, 180, (120, 120, 120), -1)
    
    # Add mouth
    cv2.ellipse(test_frame, (x+70, y+120), (35, 20), 0, 0, 180, (60, 60, 60), -1)
    
    result = analyze_frame(test_frame)
    print(f"   Faces detected: {result['num_faces']}")
    print(f"   Face hint: {result['face_hint']}")
    print(f"   Expression: {result['expression_label']}")
    
    # Test 3: Multiple face-like patterns
    print("\n3. Testing multiple face-like patterns...")
    multi_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    multi_frame[:] = (120, 120, 120)
    
    # First face
    cv2.rectangle(multi_frame, (150, 150, 100, 120), (180, 180, 180), -1)
    cv2.circle(multi_frame, (180, 190), 15, (80, 80, 80), -1)
    cv2.circle(multi_frame, (220, 190), 15, (80, 80, 80), -1)
    
    # Second face
    cv2.rectangle(multi_frame, (400, 180, 110, 130), (180, 180, 180), -1)
    cv2.circle(multi_frame, (430, 220), 15, (80, 80, 80), -1)
    cv2.circle(multi_frame, (480, 220), 15, (80, 80, 80), -1)
    
    result = analyze_frame(multi_frame)
    print(f"   Faces detected: {result['num_faces']}")
    print(f"   Face hint: {result['face_hint']}")
    
    # Test 4: Low light conditions
    print("\n4. Testing low light conditions...")
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dark_frame[:] = (30, 30, 30)  # Very dark
    
    cv2.rectangle(dark_frame, (200, 150, 120, 150), (60, 60, 60), -1)
    cv2.circle(dark_frame, (230, 200), 15, (40, 40, 40), -1)
    cv2.circle(dark_frame, (290, 200), 15, (40, 40, 40), -1)
    
    result = analyze_frame(dark_frame)
    print(f"   Faces detected: {result['num_faces']}")
    print(f"   Exposure hint: {result['exposure_hint']}")
    
    # Test 5: Bright conditions
    print("\n5. Testing bright conditions...")
    bright_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bright_frame[:] = (220, 220, 220)  # Very bright
    
    cv2.rectangle(bright_frame, (200, 150, 120, 150), (200, 200, 200), -1)
    cv2.circle(bright_frame, (230, 200), 15, (180, 180, 180), -1)
    cv2.circle(bright_frame, (290, 200), 15, (180, 180, 180), -1)
    
    result = analyze_frame(bright_frame)
    print(f"   Faces detected: {result['num_faces']}")
    print(f"   Exposure hint: {result['exposure_hint']}")
    
    print("\n" + "=" * 50)
    print("✅ Face detection testing complete!")
    print("\nKey improvements made:")
    print("• Multiple detection strategies with different parameters")
    print("• Histogram equalization for better contrast")
    print("• Face region validation to reduce false positives")
    print("• Overlapping face merging to avoid duplicates")
    print("• Better aspect ratio and size filtering")
    print("• Debug logging for troubleshooting")

if __name__ == "__main__":
    test_face_detection()
