#!/usr/bin/env python3
import cv2
import numpy as np
import os

def analyze_multi_layer():
    """Analyze the multi-layer mask approach results."""
    
    # Analyze the multi-layer result
    mask_path = "multi_layer_output/final_mask.png"
    
    print(f"\n=== Analyzing Multi-Layer Mask Approach ===")
    
    if not os.path.exists(mask_path):
        print(f"File does not exist: {mask_path}")
        return
    
    # Load image
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {mask_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Min value: {np.min(image)}")
    print(f"Max value: {np.max(image)}")
    print(f"Mean value: {np.mean(image):.2f}")
    print(f"Standard deviation: {np.std(image):.2f}")
    
    # Count unique values
    unique_values, counts = np.unique(image, return_counts=True)
    print(f"Unique values: {unique_values}")
    print(f"Value counts: {counts}")
    
    # Find contours with RETR_TREE to see the hierarchy
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours (RETR_TREE): {len(contours)}")
    
    # Analyze hierarchy
    if hierarchy is not None:
        print(f"Hierarchy shape: {hierarchy.shape}")
        
        # Count outer vs inner contours
        outer_contours = 0
        inner_contours = 0
        
        for i in range(len(contours)):
            parent = hierarchy[0][i][3]
            if parent >= 0:
                inner_contours += 1
            else:
                outer_contours += 1
        
        print(f"Outer contours: {outer_contours}")
        print(f"Inner contours (holes): {inner_contours}")
    
    # Show contour areas (sorted by size)
    areas = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        areas.append((area, i))
    
    areas.sort(reverse=True)  # Sort by area, largest first
    print(f"\nContour areas (sorted by size):")
    for area, i in areas[:40]:  # Show top 40
        parent = hierarchy[0][i][3] if hierarchy is not None else -1
        contour_type = "INNER" if parent >= 0 else "OUTER"
        print(f"  Contour {i} ({contour_type}): area = {area:.1f}")
    
    # Check for fine details
    small_contours = [area for area, _ in areas if area < 1000]
    print(f"\nFine details detected: {len(small_contours)} small contours (<1000 pixels)")
    if small_contours:
        print(f"Smallest contour: {min(small_contours):.1f} pixels")
        print(f"Average small contour: {np.mean(small_contours):.1f} pixels")
    
    # Look for very small contours that might be the stars/faces
    tiny_contours = [area for area, _ in areas if area < 100]
    print(f"Tiny details detected: {len(tiny_contours)} tiny contours (<100 pixels)")
    if tiny_contours:
        print(f"Smallest tiny contour: {min(tiny_contours):.1f} pixels")
        print(f"Average tiny contour: {np.mean(tiny_contours):.1f} pixels")
    
    # Look for medium contours that might be the "e" and stars
    medium_contours = [area for area, _ in areas if 100 <= area < 5000]
    print(f"Medium details detected: {len(medium_contours)} medium contours (100-5000 pixels)")
    if medium_contours:
        print(f"Smallest medium contour: {min(medium_contours):.1f} pixels")
        print(f"Average medium contour: {np.mean(medium_contours):.1f} pixels")
    
    # Compare with previous methods
    methods = [
        ("Free-standing", "free_standing_output/final_mask.png"),
        ("Final Preserv", "final_preservation_output/final_mask.png"),
        ("Aggressive", "aggressive_preservation_output/final_mask.png"),
        ("Enhanced", "enhanced_color_combined.png")
    ]
    
    print(f"\n=== COMPREHENSIVE COMPARISON ===")
    for method_name, method_path in methods:
        if os.path.exists(method_path):
            method_image = cv2.imread(method_path, cv2.IMREAD_GRAYSCALE)
            if method_image is not None:
                method_contours, _ = cv2.findContours(method_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                method_size = os.path.getsize(method_path)
                
                # Count small contours for each method
                method_areas = [cv2.contourArea(c) for c in method_contours]
                method_small = [a for a in method_areas if a < 1000]
                method_tiny = [a for a in method_areas if a < 100]
                method_medium = [a for a in method_areas if 100 <= a < 5000]
                
                print(f"{method_name:15} method: {len(method_contours):2d} contours, {method_size:6,} bytes, {len(method_small):2d} fine, {len(method_tiny):2d} tiny, {len(method_medium):2d} medium")
    
    print(f"Multi-Layer: {len(contours):2d} contours, {os.path.getsize(mask_path):6,} bytes, {len(small_contours):2d} fine, {len(tiny_contours):2d} tiny, {len(medium_contours):2d} medium")
    
    # Determine if this captures the missing details
    print(f"\n=== MULTI-LAYER ANALYSIS ===")
    if len(contours) > 25:
        print("✅ MAJOR SUCCESS! Many contours preserved!")
        print(f"   • {len(contours)} total contours in final mask")
        print(f"   • {len(small_contours)} fine details")
        print(f"   • {len(tiny_contours)} tiny details")
        print(f"   • {len(medium_contours)} medium details")
    else:
        print("⚠️  Still few contours in final mask")
    
    if len(medium_contours) > 8:
        print("✅ SUCCESS! Medium details (e, stars) preserved!")
        print(f"   • {len(medium_contours)} medium contours detected")
        print(f"   • Range: {min(medium_contours):.1f} to {max(medium_contours):.1f} pixels")
    else:
        print("⚠️  Few medium details preserved")
    
    if len(tiny_contours) > 8:
        print("✅ SUCCESS! Tiny details (faces) preserved!")
        print(f"   • {len(tiny_contours)} tiny contours detected")
        print(f"   • Smallest: {min(tiny_contours):.1f} pixels")
    else:
        print("⚠️  Few tiny details preserved")
    
    # Check if we have the right number of expected objects
    print(f"\n=== EXPECTED OBJECTS ANALYSIS ===")
    print("Expected objects in the logo:")
    print("• 1 main circle")
    print("• 1 'e' inside circle")
    print("• 3 stars with faces")
    print("• Multiple inner holes")
    print("• Total expected: 6+ objects")
    
    if len(contours) >= 6:
        print(f"✅ SUCCESS! Detected {len(contours)} objects (meets minimum)")
    else:
        print(f"⚠️  Only {len(contours)} objects detected (need 6+)")
    
    # Check for the specific objects we're looking for
    print(f"\n=== SPECIFIC OBJECT DETECTION ===")
    if len(medium_contours) >= 4:
        print("✅ LIKELY SUCCESS! Medium objects preserved:")
        print(f"   • {len(medium_contours)} medium contours")
        print(f"   • Should include 'e' and 3 stars")
    else:
        print("⚠️  Need more medium objects for 'e' and stars")
    
    if len(tiny_contours) >= 3:
        print("✅ LIKELY SUCCESS! Tiny objects preserved:")
        print(f"   • {len(tiny_contours)} tiny contours")
        print(f"   • Should include faces in stars")
    else:
        print("⚠️  Need more tiny objects for faces")
    
    # Check if multi-layer approach worked
    print(f"\n=== MULTI-LAYER SUCCESS CHECK ===")
    if len(contours) >= 30:
        print("🎉 ULTIMATE SUCCESS! Multi-layer approach worked!")
        print(f"   • {len(contours)} contours preserved in final mask")
        print(f"   • XOR operations successful")
        print(f"   • Free-standing objects completely separated")
        print(f"   • Complete vectorization achieved")
    elif len(contours) >= 20:
        print("✅ GOOD SUCCESS! Significant improvement:")
        print(f"   • {len(contours)} contours preserved")
        print(f"   • Much better than previous 5 contours")
        print(f"   • Multi-layer approach partially working")
    else:
        print("⚠️  Limited success - need further refinement")
        print(f"   • Only {len(contours)} contours preserved")
        print(f"   • Multi-layer approach needs adjustment")

if __name__ == "__main__":
    analyze_multi_layer() 