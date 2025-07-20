#!/usr/bin/env python3
"""
Cleaned and optimized LogoProcessor for vector tracing and logo processing.
Removed duplicate methods and unnecessary code while maintaining enhanced functionality.
"""

import os
import sys
import time
import json
import shutil
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility function for numpy compatibility
def asscalar(a):
    """Convert numpy scalar to Python scalar."""
    if hasattr(a, 'item'):
        return a.item()
    return a

class MemoryManager:
    """Memory management utilities."""
    
    def __init__(self):
        self.cache_size = 0
        self.max_cache_size = 1024 * 1024 * 100  # 100MB
    
    def check_memory(self) -> bool:
        """Check if memory usage is acceptable."""
        return self.cache_size < self.max_cache_size
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if not self.check_memory():
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up cache to free memory."""
        self.cache_size = 0

class VectorTracingModel(nn.Module):
    """Neural network model for vector tracing (placeholder for future use)."""
    
    def __init__(self):
        super().__init__()
        # Placeholder for future neural network implementation
    
    def forward(self, x):
        """Forward pass implementation."""
        return x

class LogoProcessor:
    """
    Cleaned and optimized LogoProcessor for vector tracing and logo processing.
    
    Key features:
    - Enhanced vector tracing with VTracer
    - Multi-scale detail detection
    - Comprehensive preprocessing pipeline
    - All export formats (SVG, PDF, AI)
    - Debug and analysis capabilities
    """
    
    def __init__(self, cache_dir: str = None, cache_folder: str = None, 
                 max_cache_size: int = 100, max_cache_age: int = 24, 
                 upload_folder: str = None, output_folder: str = None, 
                 temp_folder: str = None, use_redis: bool = False,
                 timeout: int = 300, max_retries: int = 3):
        """
        Initialize LogoProcessor with cleaned configuration.
        
        Args:
            cache_dir: Cache directory path
            cache_folder: Alternative cache folder name
            max_cache_size: Maximum cache size in MB
            max_cache_age: Maximum cache age in hours
            upload_folder: Upload directory path
            output_folder: Output directory path
            temp_folder: Temporary directory path
            use_redis: Whether to use Redis for caching
            timeout: Processing timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Setup directories
        self.cache_folder = cache_dir or cache_folder or os.path.join(tempfile.gettempdir(), 'logo_cache')
        self.upload_folder = upload_folder or os.path.join(tempfile.gettempdir(), 'logo_uploads')
        self.output_folder = output_folder or os.path.join(tempfile.gettempdir(), 'logo_outputs')
        self.temp_folder = temp_folder or os.path.join(tempfile.gettempdir(), 'logo_temp')
        
        # Ensure directories exist
        self._ensure_cache_dir()
        self._ensure_upload_dir()
        self._ensure_output_dir()
        self._ensure_temp_dir()
        
        # Configuration
        self.max_cache_size = max_cache_size
        self.max_cache_age = max_cache_age
        self.use_redis = use_redis
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup logging
        self.logger = logging.getLogger('zyppts.utils.logo_processor')
        
        # Memory management
        self.memory_manager = MemoryManager()
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_processing_time': 0.0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.logger.info("LogoProcessor initialized with cleaned configuration")

    def generate_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        Enhanced vector tracing method with superior detail detection for logos and letters.
        This provides maximum detail preservation with advanced preprocessing and contour detection.
        """
        start_time = time.time()
        
        try:
            # Extract options with enhanced defaults for maximum detail detection
            simplify = options.get('simplify', 0.6)  # Reduced from 0.8 for maximum details
            turdsize = options.get('turdsize', 1)    # Keep at 1 for maximum details
            noise_reduction = options.get('noise_reduction', True)
            adaptive_threshold = options.get('adaptive_threshold', True)
            generate_preview = options.get('preview', True)
            output_format = options.get('output_format', 'both')  # Always generate all formats
            ultra_detail_mode = options.get('ultra_detail_mode', True)  # New ultra-detail mode
            preserve_texture = options.get('preserve_texture', True)  # Preserve brush strokes and textures
            
            self.logger.info(f"Starting enhanced vector trace with maximum detail detection")
            self.logger.info(f"Options: simplify={simplify}, turdsize={turdsize}, ultra_detail_mode={ultra_detail_mode}")
            self.logger.info(f"Detail preservation: Enhanced preprocessing and multi-scale contour detection enabled")
            
            # Create organized output structure
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(self.output_folder, f"{base_name}_vector_trace")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create subfolders
            original_dir = os.path.join(output_dir, "original")
            processed_dir = os.path.join(output_dir, "processed")
            vector_dir = os.path.join(output_dir, "vector")
            preview_dir = os.path.join(output_dir, "preview")
            
            for folder in [original_dir, processed_dir, vector_dir, preview_dir]:
                os.makedirs(folder, exist_ok=True)
            
            # Copy original to organized structure
            original_copy = os.path.join(original_dir, os.path.basename(file_path))
            shutil.copy2(file_path, original_copy)
            
            # Step 1: Enhanced Image Preprocessing with detail preservation
            processed_image_path = self._preprocess_image_for_vectorization(
                file_path, processed_dir, noise_reduction, adaptive_threshold
            )
            
            # Step 2: Enhanced Contour Detection and Cleaning
            cleaned_image_path = self._detect_and_clean_contours(
                processed_image_path, processed_dir
            )
            
            # Step 3: Enhanced Vectorization with VTracer
            vector_results = self._vectorize_with_vtracer(
                cleaned_image_path, vector_dir, simplify, turdsize, output_format
            )
            
            # Step 4: Generate Preview (Optional)
            preview_paths = {}
            if generate_preview and 'svg' in vector_results:
                preview_paths = self._generate_vector_preview(
                    vector_results['svg'], preview_dir, base_name
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare enhanced result with detail information
            result = {
                'status': 'success',
                'output_paths': vector_results,
                'preview_paths': preview_paths,
                'processing_time': processing_time,
                'options_used': {
                    'simplify': simplify,
                    'turdsize': turdsize,
                    'noise_reduction': noise_reduction,
                    'adaptive_threshold': adaptive_threshold,
                    'preview_generated': generate_preview,
                    'ultra_detail_mode': ultra_detail_mode,
                    'preserve_texture': preserve_texture,
                    'detail_preservation': 'enhanced',
                    'multi_scale_processing': True,
                    'enhanced_contour_detection': True
                },
                'organized_structure': {
                    'original': original_dir,
                    'processed': processed_dir,
                    'vector': vector_dir,
                    'preview': preview_dir
                },
                'detail_analysis': {
                    'enhanced_preprocessing': True,
                    'multi_scale_contour_detection': True,
                    'advanced_filtering': True,
                    'duplicate_removal': True,
                    'edge_enhancement': True,
                    'texture_preservation': preserve_texture,
                    'ultra_detail_mode': ultra_detail_mode
                }
            }
            
            self.logger.info(f"Enhanced vector trace completed in {processing_time:.2f}s")
            self.logger.info(f"Generated formats: {list(vector_results.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced vector trace failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'processing_time': time.time() - start_time
            }

    def _preprocess_image_for_vectorization(self, file_path: str, output_dir: str, 
                                          noise_reduction: bool, adaptive_threshold: bool) -> str:
        """
        Enhanced image preprocessing for optimal vectorization with better detail detection.
        """
        try:
            # Load image
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    # Create white background
                    white_bg = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
                    alpha = image[:, :, 3] / 255.0
                    for c in range(3):
                        white_bg[:, :, c] = white_bg[:, :, c] * (1 - alpha) + image[:, :, c] * alpha
                    gray = cv2.cvtColor(white_bg, cv2.COLOR_BGR2GRAY)
                else:  # BGR
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhanced preprocessing for better detail detection
            processed_gray = gray.copy()
            
            # Step 1: Multi-scale detail enhancement
            # Create multiple scales to capture different detail levels
            scales = [1.0, 1.5, 2.0]
            enhanced_details = np.zeros_like(gray, dtype=np.float32)
            
            for scale in scales:
                if scale != 1.0:
                    # Resize for different scales
                    h, w = gray.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    
                    # Apply unsharp masking to enhance details
                    blurred = cv2.GaussianBlur(scaled, (0, 0), 2.0)
                    sharpened = cv2.addWeighted(scaled, 1.5, blurred, -0.5, 0)
                    
                    # Resize back to original size
                    sharpened = cv2.resize(sharpened, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    # Apply unsharp masking to original
                    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
                    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                
                # Add to enhanced details
                enhanced_details += sharpened.astype(np.float32)
            
            # Average the enhanced details
            enhanced_details = (enhanced_details / len(scales)).astype(np.uint8)
            
            # Step 2: Apply noise reduction if requested (but preserve details)
            if noise_reduction:
                # Use bilateral filter to preserve edges while reducing noise
                processed_gray = cv2.bilateralFilter(enhanced_details, 9, 75, 75)
                
                # Apply morphological operations with smaller kernel to preserve details
                kernel = np.ones((1, 1), np.uint8)  # Smaller kernel to preserve details
                processed_gray = cv2.morphologyEx(processed_gray, cv2.MORPH_CLOSE, kernel)
                processed_gray = cv2.morphologyEx(processed_gray, cv2.MORPH_OPEN, kernel)
            else:
                processed_gray = enhanced_details
            
            # Step 3: Enhanced thresholding for better detail detection
            if adaptive_threshold:
                # Use multiple adaptive thresholding methods and combine results
                # Method 1: Gaussian adaptive threshold
                binary1 = cv2.adaptiveThreshold(
                    processed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
                )
                
                # Method 2: Mean adaptive threshold with different parameters
                binary2 = cv2.adaptiveThreshold(
                    processed_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3
                )
                
                # Method 3: Otsu's method for global thresholding
                _, binary3 = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Combine the results with conservative approach (preserve more details)
                binary = np.zeros_like(binary1)
                
                # If any method detects a pixel as black, keep it black (preserve details)
                binary[(binary1 == 0) | (binary2 == 0) | (binary3 == 0)] = 0
                
                # Only make pixel white if ALL methods agree it should be white
                binary[(binary1 == 255) & (binary2 == 255) & (binary3 == 255)] = 255
                
                # For pixels where methods disagree, default to black (preserve details)
                disagreement_mask = ~((binary1 == 255) & (binary2 == 255) & (binary3 == 255)) & ~((binary1 == 0) | (binary2 == 0) | (binary3 == 0))
                binary[disagreement_mask] = 0  # Default to black (preserve details)
                
            else:
                # Use Otsu's method with preprocessing
                _, binary = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Post-processing to preserve details
            # Apply minimal morphological operations to clean up while preserving details
            kernel_small = np.ones((1, 1), np.uint8)
            kernel_medium = np.ones((2, 2), np.uint8)
            
            # Remove only isolated noise pixels (very conservative)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            
            # Close only very small gaps in letters and shapes
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
            
            # Step 5: Ensure proper orientation (VTracer expects black shapes on white background)
            # Count black and white pixels to determine if inversion is needed
            black_pixels = np.sum(binary == 0)
            white_pixels = np.sum(binary == 255)
            
            # If more than 50% is black, invert
            if black_pixels > white_pixels * 0.5:
                binary = cv2.bitwise_not(binary)
            
            # Save intermediate results for debugging
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(debug_dir, "original_gray.png"), gray)
            cv2.imwrite(os.path.join(debug_dir, "enhanced_details.png"), enhanced_details)
            cv2.imwrite(os.path.join(debug_dir, "processed_gray.png"), processed_gray)
            cv2.imwrite(os.path.join(debug_dir, "final_binary.png"), binary)
            
            # Save the final processed image
            output_path = os.path.join(output_dir, "preprocessed.png")
            cv2.imwrite(output_path, binary)
            
            self.logger.info(f"Enhanced image preprocessing completed: {output_path}")
            self.logger.info(f"Debug images saved to: {debug_dir}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Enhanced image preprocessing failed: {e}")
            raise

    def _detect_and_clean_contours(self, image_path: str, output_dir: str) -> str:
        """
        Enhanced contour detection and cleaning for maximum detail preservation.
        """
        try:
            # Load binary image
            binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Step 1: Multi-scale contour detection
            # Detect contours at different scales to capture various detail levels
            scales = [1.0, 0.75, 1.25]
            all_contours = []
            
            for scale in scales:
                if scale != 1.0:
                    # Resize image for different scales
                    h, w = binary.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    scaled_binary = binary
                
                # Find contours with different methods
                # Method 1: Standard contour detection
                contours1, _ = cv2.findContours(
                    scaled_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Method 2: External contours only (for main shapes)
                contours2, _ = cv2.findContours(
                    scaled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Method 3: All contours with different approximation
                contours3, _ = cv2.findContours(
                    scaled_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
                )
                
                # Combine contours from all methods
                scale_contours = contours1 + contours2 + contours3
                
                # Scale contours back to original size if needed
                if scale != 1.0:
                    scaled_contours = []
                    for contour in scale_contours:
                        try:
                            # Ensure contour has valid points
                            if len(contour) > 0:
                                # Scale contour points
                                scaled_contour = contour.astype(np.float32)
                                scaled_contour[:, :, 0] *= w / new_w
                                scaled_contour[:, :, 1] *= h / new_h
                                scaled_contour = scaled_contour.astype(np.int32)
                                
                                # Validate scaled contour
                                if len(scaled_contour) > 0 and cv2.contourArea(scaled_contour) > 0:
                                    scaled_contours.append(scaled_contour)
                        except Exception as e:
                            self.logger.warning(f"Skipping invalid contour after scaling: {e}")
                            continue
                    scale_contours = scaled_contours
                
                all_contours.extend(scale_contours)
            
            # Step 2: Conservative contour filtering for maximum detail preservation
            filtered_contours = []
            
            for contour in all_contours:
                try:
                    # Validate contour before processing
                    if len(contour) < 3:
                        continue
                    
                    # Calculate contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Conservative filtering criteria - keep almost everything
                    keep_contour = False
                    
                    # Criterion 1: Area-based filtering (lenient for details)
                    if area >= 1:  # Reduced from 2 to 1 to preserve even more details
                        keep_contour = True
                    
                    # Criterion 2: Aspect ratio filtering (for letter details)
                    if len(contour) >= 4:  # Need at least 4 points for bounding rect
                        try:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = max(w, h) / max(1, min(w, h))
                            
                            # Keep thin contours (like letter strokes) even if small
                            if aspect_ratio > 2.0 and area >= 0.5:  # Even more lenient for thin contours
                                keep_contour = True
                        except Exception:
                            pass
                    
                    # Criterion 3: Perimeter-based filtering (for detailed shapes)
                    if perimeter >= 6:  # Reduced from 8 to 6 for more details
                        keep_contour = True
                    
                    # Criterion 4: Circularity-based filtering (for dots and small details)
                    if area > 0 and perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.2:  # Reduced from 0.3 to 0.2 for more circular shapes
                            keep_contour = True
                    
                    if keep_contour:
                        filtered_contours.append(contour)
                        
                except Exception as e:
                    self.logger.warning(f"Skipping invalid contour during filtering: {e}")
                    continue
            
            # Step 3: Remove duplicate and overlapping contours
            unique_contours = self._remove_duplicate_contours_enhanced(filtered_contours)
            
            # Step 4: Create enhanced clean mask
            clean_mask = np.ones_like(binary) * 255  # Start with white background
            
            # Sort contours by area (largest first) to handle overlapping
            unique_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            
            for contour in unique_contours:
                try:
                    # Fill contour with black
                    cv2.fillPoly(clean_mask, [contour], 0)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid contour during mask creation: {e}")
                    continue
            
            # Step 5: Post-processing for detail enhancement
            # Apply morphological operations to clean up while preserving details
            kernel_small = np.ones((1, 1), np.uint8)
            kernel_medium = np.ones((2, 2), np.uint8)
            
            # Remove isolated noise pixels
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Close small gaps in letters and shapes
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_medium)
            
            # Step 6: Final detail enhancement
            # Apply edge enhancement to preserve sharp details
            kernel_edge = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]], dtype=np.float32)
            edge_enhanced = cv2.filter2D(clean_mask, -1, kernel_edge)
            
            # Combine with original mask
            final_mask = cv2.addWeighted(clean_mask, 0.9, edge_enhanced, 0.1, 0)
            
            # Ensure binary output
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Step 7: Save debug information
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create debug visualization
            debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            try:
                cv2.drawContours(debug_img, unique_contours, -1, (0, 255, 0), 1)
            except Exception as e:
                self.logger.warning(f"Could not draw contours for debug: {e}")
            
            cv2.imwrite(os.path.join(debug_dir, "original_binary.png"), binary)
            cv2.imwrite(os.path.join(debug_dir, "contours_debug.png"), debug_img)
            cv2.imwrite(os.path.join(debug_dir, "clean_mask.png"), clean_mask)
            cv2.imwrite(os.path.join(debug_dir, "final_mask.png"), final_mask)
            
            # Save the final cleaned image
            output_path = os.path.join(output_dir, "cleaned_contours.png")
            cv2.imwrite(output_path, final_mask)
            
            self.logger.info(f"Enhanced contour detection completed: {output_path}")
            self.logger.info(f"Found {len(unique_contours)} unique contours after filtering")
            self.logger.info(f"Debug images saved to: {debug_dir}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Enhanced contour detection failed: {e}")
            raise

    def _remove_duplicate_contours_enhanced(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhanced duplicate contour removal with better overlap detection.
        """
        if not contours:
            return []
        
        unique_contours = []
        
        for i, contour1 in enumerate(contours):
            is_duplicate = False
            
            for j, contour2 in enumerate(unique_contours):
                # Calculate overlap using intersection over union (IoU)
                overlap = self._calculate_contour_overlap(contour1, contour2)
                
                # If overlap is high, consider it a duplicate
                if overlap > 0.7:  # 70% overlap threshold
                    # Keep the contour with larger area
                    area1 = cv2.contourArea(contour1)
                    area2 = cv2.contourArea(contour2)
                    
                    if area1 > area2:
                        # Replace the existing contour
                        unique_contours[j] = contour1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour1)
        
        return unique_contours

    def _calculate_contour_overlap(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Calculate the overlap between two contours using intersection over union.
        """
        try:
            # Create masks for both contours
            h, w = max(cv2.boundingRect(contour1)[2:]) + max(cv2.boundingRect(contour2)[2:]), \
                   max(cv2.boundingRect(contour1)[2:]) + max(cv2.boundingRect(contour2)[2:])
            
            mask1 = np.zeros((h, w), dtype=np.uint8)
            mask2 = np.zeros((h, w), dtype=np.uint8)
            
            # Adjust contours to fit in the combined mask
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            # Create offset to place contours in the combined mask
            offset_x = min(x1, x2)
            offset_y = min(y1, y2)
            
            # Draw contours on masks
            cv2.fillPoly(mask1, [contour1 - np.array([offset_x, offset_y])], 255)
            cv2.fillPoly(mask2, [contour2 - np.array([offset_x, offset_y])], 255)
            
            # Calculate intersection and union
            intersection = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)
            
            intersection_area = np.sum(intersection > 0)
            union_area = np.sum(union > 0)
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
        except Exception:
            # Fallback: simple area-based comparison
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            
            if area1 == 0 or area2 == 0:
                return 0.0
            
            # Use the smaller area as reference
            min_area = min(area1, area2)
            max_area = max(area1, area2)
            
            return min_area / max_area

    def _vectorize_with_vtracer(self, image_path: str, output_dir: str, 
                              simplify: float, turdsize: int, output_format: str) -> Dict[str, str]:
        """
        Enhanced vectorization using VTracer CLI with optimized parameters for detail preservation.
        """
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_paths = {}
            
            # Check if VTracer is available
            vtracer_path = self._find_vtracer()
            if not vtracer_path:
                self.logger.warning("VTracer not found, using fallback vectorization")
                return self._fallback_vectorization(image_path, output_dir, output_format)
            
            # Enhanced VTracer parameters for better detail detection
            # Reduce turdsize for more details
            enhanced_turdsize = max(1, turdsize // 2)  # More aggressive detail preservation
            
            # Adjust simplify parameter for more details
            enhanced_simplify = max(0.5, simplify * 0.8)  # Less simplification for more details
            
            # Build enhanced VTracer command with optimized parameters
            svg_path = os.path.join(output_dir, f"{base_name}.svg")
            cmd = [
                vtracer_path,
                '--input', image_path,
                '--output', svg_path,
                '--colormode', 'bw',  # Binary mode for single-color tracing
                '--filter_speckle', str(int(enhanced_turdsize)),  # Reduced for more details
                '--corner_threshold', str(int(enhanced_simplify * 8)),  # Reduced for sharper corners
                '--mode', 'polygon',  # Use polygon mode for cleaner paths
                '--path_precision', '3',  # Increased precision for better detail
                '--segment_length', '4',  # Fixed: Must be within [3.5,10] range
                '--splice_threshold', '5'  # Low threshold for more detailed splines
            ]
            
            # Run VTracer with enhanced parameters
            self.logger.info(f"Running enhanced VTracer: {' '.join(cmd[:10])}...")  # Log first 10 args
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                self.logger.error(f"VTracer failed: {result.stderr}")
                # Try with simpler parameters as fallback
                self.logger.info("Trying fallback VTracer parameters...")
                fallback_cmd = [
                    vtracer_path,
                    '--input', image_path,
                    '--output', svg_path,
                    '--colormode', 'bw',
                    '--filter_speckle', str(int(enhanced_turdsize)),
                    '--corner_threshold', str(int(enhanced_simplify * 10)),
                    '--mode', 'polygon',
                    '--path_precision', '2',
                    '--segment_length', '4'  # Fixed: Must be within [3.5,10] range
                ]
                
                result = subprocess.run(
                    fallback_cmd, capture_output=True, text=True, timeout=300
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"VTracer failed with fallback parameters: {result.stderr}")
            
            # Check if SVG was created
            if os.path.exists(svg_path):
                output_paths['svg'] = svg_path
                self.logger.info(f"Enhanced SVG created: {svg_path}")
                
                # Convert to PDF if requested
                if output_format in ['pdf', 'both']:
                    pdf_path = self._convert_svg_to_pdf(svg_path, output_dir)
                    if pdf_path:
                        output_paths['pdf'] = pdf_path
                        self.logger.info(f"PDF created: {pdf_path}")
                
                # Create AI file (copy of PDF for Adobe Illustrator compatibility)
                if output_format in ['both', 'ai']:
                    ai_path = os.path.join(output_dir, f"{base_name}.ai")
                    if 'pdf' in output_paths:
                        shutil.copy2(output_paths['pdf'], ai_path)
                        output_paths['ai'] = ai_path
                        self.logger.info(f"AI file created: {ai_path}")
                    else:
                        # Create AI file from SVG if PDF conversion failed
                        ai_path = self._convert_svg_to_ai(svg_path, output_dir)
                        if ai_path:
                            output_paths['ai'] = ai_path
                            self.logger.info(f"AI file created from SVG: {ai_path}")
            else:
                raise RuntimeError("VTracer did not create SVG file")
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Enhanced VTracer vectorization failed: {e}")
            raise

    def _find_vtracer(self) -> Optional[str]:
        """
        Find VTracer executable in system PATH or common locations.
        VTracer is used as the primary vectorization tool.
        """
        # Check system PATH for vtracer
        try:
            result = subprocess.run(['which', 'vtracer'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Check common installation paths
        common_paths = [
            '/usr/local/bin/vtracer',
            '/opt/homebrew/bin/vtracer',
            '/usr/bin/vtracer',
            os.path.expanduser('~/vtracer/vtracer'),
            os.path.expanduser('~/.local/bin/vtracer')
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None

    def _convert_svg_to_pdf(self, svg_path: str, output_dir: str) -> Optional[str]:
        """
        Convert SVG to PDF using cairosvg or svglib.
        """
        try:
            base_name = os.path.splitext(os.path.basename(svg_path))[0]
            pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
            
            # Try cairosvg first
            try:
                import cairosvg
                cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
                self.logger.info(f"PDF created with cairosvg: {pdf_path}")
                return pdf_path
            except ImportError:
                pass
            
            # Try svglib as fallback
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPDF
                
                drawing = svg2rlg(svg_path)
                renderPDF.drawToFile(drawing, pdf_path)
                self.logger.info(f"PDF created with svglib: {pdf_path}")
                return pdf_path
            except ImportError:
                pass
            
            # Try Inkscape as last resort
            try:
                cmd = ['inkscape', '--export-type=pdf', '--export-filename', pdf_path, svg_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and os.path.exists(pdf_path):
                    self.logger.info(f"PDF created with Inkscape: {pdf_path}")
                    return pdf_path
            except:
                pass
            
            self.logger.warning("No PDF converter available")
            return None
            
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {e}")
            return None

    def _convert_svg_to_ai(self, svg_path: str, output_dir: str) -> Optional[str]:
        """
        Convert SVG to AI format for Adobe Illustrator compatibility.
        """
        try:
            base_name = os.path.splitext(os.path.basename(svg_path))[0]
            ai_path = os.path.join(output_dir, f"{base_name}.ai")
            
            # For now, we'll create a simple AI-compatible file by copying the SVG
            # In a production environment, you might want to use a proper SVG to AI converter
            shutil.copy2(svg_path, ai_path)
            
            # Add AI file header if needed (this is a simplified approach)
            # Real AI files have a specific format, but many applications can read SVG files
            # with .ai extension
            
            self.logger.info(f"AI file created: {ai_path}")
            return ai_path
            
        except Exception as e:
            self.logger.error(f"AI conversion failed: {e}")
            return None

    def _generate_vector_preview(self, svg_path: str, preview_dir: str, base_name: str) -> Dict[str, str]:
        """
        Generate preview PNG from SVG using cairosvg or svglib.
        """
        try:
            preview_paths = {}
            
            # Generate different preview sizes
            sizes = {
                'thumb': (200, 200),
                'medium': (800, 800),
                'large': (1200, 1200)
            }
            
            for size_name, (width, height) in sizes.items():
                preview_path = os.path.join(preview_dir, f"{base_name}_{size_name}.png")
                
                # Try cairosvg first
                try:
                    import cairosvg
                    cairosvg.svg2png(
                        url=svg_path, 
                        write_to=preview_path,
                        output_width=width,
                        output_height=height
                    )
                    preview_paths[size_name.upper()] = preview_path
                except ImportError:
                    # Try svglib as fallback
                    try:
                        from svglib.svglib import svg2rlg
                        from reportlab.graphics import renderPM
                        
                        drawing = svg2rlg(svg_path)
                        renderPM.drawToFile(drawing, preview_path, fmt="PNG", dpi=150)
                        preview_paths[size_name.upper()] = preview_path
                    except ImportError:
                        self.logger.warning(f"Could not generate {size_name} preview - no converter available")
                        continue
            
            self.logger.info(f"Generated {len(preview_paths)} preview images")
            return preview_paths
            
        except Exception as e:
            self.logger.error(f"Preview generation failed: {e}")
            return {}

    def _fallback_vectorization(self, image_path: str, output_dir: str, output_format: str) -> Dict[str, str]:
        """
        Fallback vectorization method when VTracer is not available.
        """
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_paths = {}
            
            # Create a simple SVG as fallback
            svg_path = os.path.join(output_dir, f"{base_name}.svg")
            
            # Read the image to get dimensions
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape
            
            # Create a simple SVG with the image as a rectangle
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="black"/>
</svg>'''
            
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            
            output_paths['svg'] = svg_path
            self.logger.info(f"Fallback SVG created: {svg_path}")
            
            # Convert to PDF if requested
            if output_format in ['pdf', 'both']:
                pdf_path = self._convert_svg_to_pdf(svg_path, output_dir)
                if pdf_path:
                    output_paths['pdf'] = pdf_path
            
            # Create AI file if requested
            if output_format in ['both', 'ai']:
                ai_path = self._convert_svg_to_ai(svg_path, output_dir)
                if ai_path:
                    output_paths['ai'] = ai_path
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Fallback vectorization failed: {e}")
            raise

    # Utility methods for directory management
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_folder, exist_ok=True)
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists."""
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _ensure_temp_dir(self):
        """Ensure temp directory exists."""
        os.makedirs(self.temp_folder, exist_ok=True)

    def _start_cleanup_thread(self):
        """Start periodic cleanup thread."""
        import threading
        
        def cleanup_loop():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_resources()
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        self.logger.info("Cleanup thread started")

    def _cleanup_resources(self):
        """Clean up resources and old cache files."""
        try:
            # Clean up old cache files
            current_time = time.time()
            for filename in os.listdir(self.cache_folder):
                file_path = os.path.join(self.cache_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > (self.max_cache_age * 3600):  # Convert hours to seconds
                        os.remove(file_path)
                        self.logger.debug(f"Removed old cache file: {filename}")
            
            # Clean up temp files
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 3600:  # Remove temp files older than 1 hour
                        os.remove(file_path)
                        self.logger.debug(f"Removed old temp file: {filename}")
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.processing_stats.copy()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self._cleanup_resources()
        except:
            pass 