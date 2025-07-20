print("DEBUG: Loaded logo_processor.py from", __file__)
import os
import io
import sys
import time
import json
import gc
import traceback
import threading
import multiprocessing
import tempfile
import hashlib
import zipfile
import warnings
import datetime
import math
import random
import shutil
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple, List, Union, Any, Callable
from pathlib import Path

# Fix numpy.asscalar() deprecation issue in colormath
import numpy as np
if not hasattr(np, 'asscalar'):
    def asscalar(a):
        return a.item() if hasattr(a, 'item') else float(a)
    np.asscalar = asscalar

import cv2
import numpy as np
import psutil
import requests
import torch
import torch.nn as nn
from PIL import (
    Image, ImageDraw, ImageEnhance, ImageFilter, 
    ImageChops, ImageOps, ImageFile, ImageFont
)
from scipy import ndimage
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from tqdm import tqdm
from lxml import etree
import cairosvg
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import base64
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Try to import modern vector tracing libraries
try:
    import cv2
    import numpy as np
    from svgpathtools import Path as SVGPath, Line, CubicBezier
    import svgwrite
    from pathlib import Path
except ImportError as e:
    print(f"Warning: Some vector tracing libraries not available: {e}")

# Advanced computer vision imports for sophisticated contour detection
try:
    from skimage import measure, morphology, filters, segmentation, feature, util
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize, thin, remove_small_objects, remove_small_holes
    from skimage.filters import sobel, gaussian, threshold_otsu, threshold_local
    from skimage.segmentation import watershed, slic, felzenszwalb
    from skimage.feature import corner_harris, corner_peaks
    from skimage.util import img_as_ubyte, img_as_float
    from scipy import ndimage
    from scipy.spatial import distance
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    from scipy.signal import find_peaks
    from shapely.geometry import Polygon, MultiPolygon, Point, LineString
    from shapely.ops import unary_union, polygonize
    from shapely.validation import make_valid
    import mahotas as mh
    from imutils import contours as imutils_contours
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    ADVANCED_CV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced computer vision libraries not available: {e}")
    ADVANCED_CV_AVAILABLE = False

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
MAX_IMAGE_SIZE = 5000  # Maximum width or height in pixels
DEFAULT_QUALITY = 85  # Default quality for JPEG/WebP
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.svg']

class MemoryManager:
    """Manages memory usage to prevent OOM errors."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_limit = psutil.virtual_memory().total * 0.7  # 70% of total RAM
        self.process = psutil.Process()
        self.cache = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.process.memory_info().rss
        if current_usage > self.memory_limit:
            self.logger.warning(f"Memory usage high: {current_usage / (1024*1024):.2f} MB")
            self._cleanup_cache()
        return current_usage < self.memory_limit
    
    def optimize_memory(self):
        """Optimize memory usage by clearing caches."""
        import gc
        gc.collect()
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        # Remove old entries
        self.cache = {k: v for k, v in self.cache.items() 
                     if v['timestamp'] > current_time - self.cleanup_interval}
        
        # Clear PIL image cache
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        ImageFile.MAX_IMAGE_PIXELS = None
        
        # Clear OpenCV cache
        cv2.destroyAllWindows()
        
        self.last_cleanup = current_time
        self.logger.info(f"Cache cleaned up. Memory usage: {self.process.memory_info().rss / (1024*1024):.2f} MB")

class VectorTracingModel(nn.Module):
    """Neural network model for vector tracing."""
    def __init__(self):
        super(VectorTracingModel, self).__init__()
        # Model architecture here
        
    def forward(self, x):
        # Forward pass implementation
        return x

class LogoProcessor:
    """Processes and optimizes logo images with various effects."""
    
    def __init__(self, cache_dir: str = None, cache_folder: str = None, 
                 max_cache_size: int = 100, max_cache_age: int = 24, 
                 upload_folder: str = None, output_folder: str = None, 
                 temp_folder: str = None, use_redis: bool = False,
                 timeout: int = 300,  # 5 minute timeout
                 max_retries: int = 3):
        """Initialize the LogoProcessor with configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Support both cache_dir and cache_folder for compatibility
        if cache_folder is not None:
            cache_dir = cache_folder
            
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'logo_processor_cache')
        # Add cache_folder as an alias to cache_dir for compatibility with SAM model loading
        self.cache_folder = self.cache_dir
        self.max_cache_size = max_cache_size
        self.max_cache_age = max_cache_age
        self.upload_folder = upload_folder or os.path.join(tempfile.gettempdir(), 'logo_uploads')
        self.output_folder = output_folder or os.path.join(tempfile.gettempdir(), 'logo_outputs')
        self.temp_folder = temp_folder or os.path.join(tempfile.gettempdir(), 'logo_temp')
        
        # Initialize basic cache
        self.cache: Dict[str, Dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize directories
        self._ensure_cache_dir()
        self._ensure_upload_dir()
        self._ensure_output_dir()
        self._ensure_temp_dir()
        self._load_cache()
        self._cleanup_old_cache()
        
        # Initialize Redis if requested
        self.redis_client = None
        if use_redis:
            try:
                from redis import Redis
                self.redis_client = Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 0)),
                    password=os.getenv('REDIS_PASSWORD'),
                    socket_timeout=float(os.getenv('REDIS_SOCKET_TIMEOUT', 5.0)),
                    socket_connect_timeout=float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', 5.0)),
                    retry_on_timeout=True
                )
                self.redis_client.ping()
                self.logger.info("Redis cache initialized successfully")
            except Exception as e:
                self.logger.warning(f"Redis not available, using file-based cache: {str(e)}")
                self.redis_client = None
        
        # Initialize processing queue and progress tracking
        self.processing_queue = []
        self.progress_callbacks = {}
        self.current_process_id = None
        self.processing_status = {}
        self.processing_stats = {
            'total_processed': 0,
            'success': 0,
            'failures': 0,
            'avg_processing_time': 0
        }
        
        # Initialize memory management
        self.memory_manager = MemoryManager()
        self.memory_threshold = int(psutil.virtual_memory().total * 0.7)  # 70% of total RAM
        self.chunk_size = 1024 * 1024  # 1MB chunks for processing
        
        # Initialize thread pool for parallel processing with optimized configuration
        # Use more workers for I/O-bound tasks like image processing
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(128, cpu_count * 8)  # Significantly increased for better parallelization
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='logo_processor'
        )
        
        # Advanced parallel processing configuration
        self.parallel_config = {
            'max_workers': self.max_workers,
            'cpu_intensive_workers': min(32, cpu_count * 2),  # For vector processing
            'io_intensive_workers': min(64, cpu_count * 4),   # For image processing
            'memory_threshold_mb': 1024,  # 1GB memory threshold
            'task_timeout': 300,  # 5 minutes per task
            'batch_size': 8,  # Process tasks in batches
            'enable_multiprocessing': True,  # Use multiprocessing for CPU-intensive tasks
            'enable_priority_queue': True,  # Prioritize faster tasks
            'enable_memory_monitoring': True,  # Monitor memory usage
            'enable_adaptive_scaling': True,  # Scale workers based on load
        }
        
        # Performance monitoring with enhanced metrics
        self.processing_stats = {
            'total_processed': 0,
            'success': 0,
            'failures': 0,
            'avg_processing_time': 0,
            'parallel_processing_enabled': True,
            'max_workers': self.max_workers,
            'cpu_usage': 0,
            'memory_usage_mb': 0,
            'active_workers': 0,
            'queue_size': 0,
            'task_completion_times': {},
            'optimization_level': 'advanced'
        }
        
        # Task priority queue for better scheduling
        self.task_queue = []
        self.queue_lock = threading.Lock()
        
        # Memory monitoring
        self.memory_monitor = {
            'last_check': time.time(),
            'peak_usage': 0,
            'current_usage': 0,
            'threshold_exceeded': False
        }
        
        # Initialize locks
        self.status_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Lazy-loading for AI models to improve startup time and memory usage
        self.sd_pipeline = None
        self.sd_pipeline_lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"AI models will be loaded on first use. Using device: {self.device}")
        
        # Processing configuration
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = 1  # seconds between retries
        
        # Preview settings
        self.preview_size = (300, 300)
        self.preview_quality = 75
        self.preview_cache = {}
        self.preview_cache_timeout = 300
        
        # Social media sizes
        self.social_sizes = {
            'facebook_post': (1200, 630),
            'facebook_profile': (1080, 1080),
            'twitter_post': (1600, 900),
            'twitter_header': (1500, 500),
            'twitter_profile': (400, 400),
            'instagram_post': (1080, 1080),
            'instagram_story': (1080, 1920),
            'instagram_profile': (320, 320),
            'linkedin_post': (1200, 1200),
            'linkedin_banner': (1584, 396),
            'linkedin_profile': (400, 400),
            'pinterest_pin': (1000, 1500),
            'pinterest_profile': (165, 165),
            'pinterest_board': (222, 150),
            'tiktok_profile': (200, 200),
            'tiktok_video': (1080, 1920)
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()

    def generate_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        Main vector tracing method using high-quality VTracer-based processing.
        This provides superior vectorization results with advanced preprocessing.
        """
        start_time = time.time()
        
        try:
            # Extract options with defaults optimized for detail detection
            simplify = options.get('simplify', 0.6)  # Reduced from 0.8 for more details
            turdsize = options.get('turdsize', 1)    # Reduced from 2 for more details
            noise_reduction = options.get('noise_reduction', True)
            adaptive_threshold = options.get('adaptive_threshold', True)
            generate_preview = options.get('preview', True)
            output_format = options.get('output_format', 'both')  # Always generate all formats
            
            self.logger.info(f"Starting enhanced vector trace with VTracer")
            self.logger.info(f"Options: simplify={simplify}, turdsize={turdsize}, noise_reduction={noise_reduction}")
            self.logger.info(f"Detail preservation: Enhanced preprocessing and contour detection enabled")
            
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
                    'edge_enhancement': True
                }
            }
            
            self.logger.info(f"Enhanced vector trace completed in {processing_time:.2f}s")
            self.logger.info(f"Generated formats: {list(vector_results.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"High-quality vector trace failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'processing_time': time.time() - start_time
            }

    def _generate_cache_key(self, file_path: str, options: Dict) -> str:
        """Generate cache key for vector trace results."""
        import hashlib
        file_stat = os.stat(file_path)
        options_str = str(sorted(options.items()))
        content = f"{file_path}:{file_stat.st_mtime}:{file_stat.st_size}:{options_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_vector_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached vector trace result."""
        try:
            cache_file = os.path.join(self.cache_folder, f"vector_trace_{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if cache is still valid (24 hours)
                if time.time() - cached_data.get('timestamp', 0) < 86400:
                    return cached_data.get('result')
        except Exception as e:
            self.logger.warning(f"Cache read failed: {e}")
        return None

    def _cache_vector_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache vector trace result."""
        try:
            # Create a serializable copy of the result
            cacheable_result = self._make_result_cacheable(result)
            
            cache_file = os.path.join(self.cache_folder, f"vector_trace_{cache_key}.json")
            cache_data = {
                'result': cacheable_result,
                'timestamp': time.time()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            self.logger.warning(f"Cache write failed: {e}")

    def _make_result_cacheable(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to be JSON serializable."""
        cacheable_result = {}
        
        for key, value in result.items():
            if key == 'output_paths':
                # Keep output paths as strings
                cacheable_result[key] = value
            elif key == 'color_details':
                # Convert color details to be serializable
                cacheable_colors = []
                for color in value:
                    cacheable_color = {
                        'rgb': [int(x) for x in color['rgb']] if isinstance(color['rgb'], (tuple, list, np.ndarray)) else color['rgb'],
                        'hex': color['hex']
                    }
                    cacheable_colors.append(cacheable_color)
                cacheable_result[key] = cacheable_colors
            elif key == 'enhancements_applied':
                # Keep enhancements as is (should be serializable)
                cacheable_result[key] = value
            elif key == 'processing_time':
                # Convert to float if it's a numpy type
                cacheable_result[key] = float(value) if hasattr(value, 'item') else value
            else:
                # Convert numpy types to native Python types
                if isinstance(value, np.integer):
                    cacheable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    cacheable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    cacheable_result[key] = value.tolist()
                else:
                    cacheable_result[key] = value
        
        return cacheable_result

    def _analyze_logo_colors_ultra_fast(self, arr: np.ndarray, opts: Dict) -> Tuple[List[Dict], np.ndarray]:
        """
        Ultra-fast logo color analysis with enhanced detection for complex logos.
        """
        # Convert to RGB for color analysis
        rgb_arr = arr[:, :, :3]
        alpha_arr = arr[:, :, 3]
        
        # Enhanced background detection for complex logos
        if np.max(alpha_arr) == 0 or np.min(alpha_arr) == 255:
            logo_mask = self._detect_logo_background_enhanced(rgb_arr, opts)
        else:
            logo_mask = self._clean_alpha_mask_ultra_fast(alpha_arr, opts)
        
        # Extract logo pixels
        logo_pixels = rgb_arr[logo_mask > 0]
        
        if len(logo_pixels) == 0:
            self.logger.warning("No logo pixels detected")
            return [], logo_mask
        
        # Cluster colors
        colors = self._cluster_colors_ultra_fast(logo_pixels, opts)
        
        # Safety check: ensure all colors have required keys
        for color in colors:
            if 'pixel_count' not in color:
                color['pixel_count'] = 1000  # Default value
            if 'color_variance' not in color:
                color['color_variance'] = [30, 30, 30]
            if 'is_dominant' not in color:
                color['is_dominant'] = False
        
        self.logger.info(f"Detected {len(colors)} colors in logo")
        return colors, logo_mask

    def _detect_logo_background_enhanced(self, rgb_arr: np.ndarray, opts: Dict) -> np.ndarray:
        """
        Enhanced background detection for complex logos with sharp edges.
        """
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding for better edge detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _cluster_colors_enhanced(self, logo_pixels: np.ndarray, opts: Dict) -> List[Dict]:
        """
        Enhanced color clustering for complex logos with better precision.
        """
        if len(logo_pixels) == 0:
            return []
        
        # For black logos, use more precise clustering
        if np.mean(logo_pixels) < 50:  # Dark logo
            return self._cluster_black_colors_precise(logo_pixels, opts)
        else:
            return self._cluster_colors_ultra_fast(logo_pixels, opts)

    def _cluster_black_colors_precise(self, logo_pixels: np.ndarray, opts: Dict) -> List[Dict]:
        """
        Ultra-precise color clustering specifically for black logos with texture variations.
        """
        if len(logo_pixels) == 0:
            return []
        
        # For black logos, use ultra-precise color detection including texture variations
        # Group pixels by exact color values with minimal tolerance
        unique_colors, counts = np.unique(logo_pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        
        # Filter out very rare colors (noise) but keep texture variations
        min_count = max(1, len(logo_pixels) * 0.0005)  # At least 0.05% of pixels for texture
        valid_indices = counts >= min_count
        
        if not np.any(valid_indices):
            # If no colors meet the threshold, use the most common color
            valid_indices[0] = True
        
        colors = []
        for i, (color, count) in enumerate(zip(unique_colors[valid_indices], counts[valid_indices])):
            # Calculate color variance for this specific color
            color_mask = np.all(logo_pixels == color, axis=1)
            color_pixels = logo_pixels[color_mask]
            
            if len(color_pixels) > 0:
                variance = np.var(color_pixels, axis=0)
            else:
                variance = [0, 0, 0]
            
            colors.append({
                'rgb': color.tolist(),
                'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                'count': int(count),
                'percentage': float(count / len(logo_pixels) * 100),
                'color_variance': variance.tolist(),
                'is_dominant': i == 0,  # First color is dominant
                'precision': 'ultra_exact',  # Ultra-exact color matching
                'texture_variation': variance[0] + variance[1] + variance[2] > 0  # Indicates texture
            })
        
        return colors

    def _clean_alpha_mask_ultra_fast(self, alpha_arr: np.ndarray, opts: Dict) -> np.ndarray:
        """
        Ultra-fast alpha mask cleaning.
        """
        # Simple thresholding
        mask = (alpha_arr > 128).astype(np.uint8) * 255
        
        # Single morphological operation
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def _cluster_colors_ultra_fast(self, logo_pixels: np.ndarray, opts: Dict) -> List[Dict]:
        """
        Comprehensive color clustering for complex logos with multiple colors.
        """
        # Remove outliers using simple method
        cleaned_pixels = self._remove_color_outliers_ultra_fast(logo_pixels, opts)
        
        if len(cleaned_pixels) == 0:
            return []
        
        # Check for unique colors
        unique_colors = np.unique(cleaned_pixels, axis=0)
        if len(unique_colors) == 0:
            return []
        
        # For complex logos, detect more colors
        max_colors = min(12, len(unique_colors))  # Increased from 3 to 12
        
        if max_colors < 2:
            if max_colors == 1:
                color = unique_colors[0]
                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                return [{
                    'rgb': tuple(color),
                    'hex': color_hex,
                    'cluster_id': 0,
                    'pixel_count': len(cleaned_pixels),
                    'color_variance': [0, 0, 0],
                    'is_dominant': True
                }]
            else:
                return []
        
        # Use K-means with more iterations for better color detection
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=5, max_iter=100)  # Increased iterations
            kmeans.fit(cleaned_pixels)
            
            # Get cluster centers and labels
            cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
            labels = kmeans.labels_
            
            # Create color info
            colors = []
            for i, center in enumerate(cluster_centers):
                color_hex = f"#{center[0]:02x}{center[1]:02x}{center[2]:02x}"
                pixel_count = np.sum(labels == i)
                
                # Calculate actual variance for this cluster
                cluster_pixels = cleaned_pixels[labels == i]
                if len(cluster_pixels) > 1:
                    variance = np.std(cluster_pixels, axis=0)
                else:
                    variance = [30, 30, 30]
                
                colors.append({
                    'rgb': tuple(center),
                    'hex': color_hex,
                    'cluster_id': i,
                    'pixel_count': pixel_count,
                    'color_variance': variance.tolist() if hasattr(variance, 'tolist') else list(variance),
                    'is_dominant': pixel_count > len(cleaned_pixels) * 0.05  # Reduced threshold for more dominant colors
                })
            
            # Sort by pixel count
            colors.sort(key=lambda x: x['pixel_count'], reverse=True)
            
            # Filter out very small color clusters (less than 0.5% of pixels)
            min_pixels = len(cleaned_pixels) * 0.005
            colors = [c for c in colors if c['pixel_count'] >= min_pixels]
            
            self.logger.info(f"Detected {len(colors)} colors with {max_colors} clusters")
            
            # Safety check: ensure all colors have required keys
            for color in colors:
                if 'pixel_count' not in color:
                    color['pixel_count'] = 1000  # Default value
                if 'color_variance' not in color:
                    color['color_variance'] = [30, 30, 30]
                if 'is_dominant' not in color:
                    color['is_dominant'] = False
            
            return colors
            
        except Exception as e:
            self.logger.error(f"Comprehensive K-means clustering failed: {e}")
            # Fallback: return the most common colors
            from collections import Counter
            color_counts = Counter([tuple(x) for x in cleaned_pixels])
            most_common_colors = color_counts.most_common(min(6, len(color_counts)))
            
            colors = []
            for i, (color, count) in enumerate(most_common_colors):
                if count >= len(cleaned_pixels) * 0.01:  # At least 1% of pixels
                    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    colors.append({
                        'rgb': color,
                        'hex': color_hex,
                        'cluster_id': i,
                        'pixel_count': count,
                        'color_variance': [30, 30, 30],
                        'is_dominant': count > len(cleaned_pixels) * 0.1
                    })
            
            self.logger.info(f"Fallback detected {len(colors)} colors")
            
            # Safety check: ensure all colors have required keys
            for color in colors:
                if 'pixel_count' not in color:
                    color['pixel_count'] = 1000  # Default value
                if 'color_variance' not in color:
                    color['color_variance'] = [30, 30, 30]
                if 'is_dominant' not in color:
                    color['is_dominant'] = False
            
            return colors

    def _remove_color_outliers_ultra_fast(self, pixels: np.ndarray, opts: Dict) -> np.ndarray:
        """
        Ultra-fast color outlier removal using simple percentile method.
        """
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0)
        distances = np.linalg.norm(pixels - mean_color, axis=1)
        
        # Use simple percentile-based filtering
        threshold = np.percentile(distances, 90)  # Keep 90% of pixels for speed
        mask = distances <= threshold
        return pixels[mask]

    def _find_closest_pms_color(self, rgb_color: tuple) -> str:
        """
        Find the closest PMS color to the given RGB color.
        Returns the closest PMS color code.
        """
        # Common PMS colors with their RGB equivalents
        pms_colors = {
            'PMS Black': (0, 0, 0),
            'PMS White': (255, 255, 255),
            'PMS 185': (220, 20, 60),  # Bright Red
            'PMS 286': (0, 71, 171),   # Blue
            'PMS 354': (0, 135, 81),   # Green
            'PMS 116': (255, 205, 0),  # Yellow
            'PMS 137': (255, 140, 0),  # Orange
            'PMS 2592': (128, 0, 128), # Purple
            'PMS 7545': (128, 128, 128), # Gray
            'PMS 877': (192, 192, 192),  # Silver
            'PMS 871': (184, 115, 51),   # Bronze
            'PMS 872': (212, 175, 55),   # Gold
            'PMS 485': (255, 0, 0),      # Process Red
            'PMS 286': (0, 0, 255),      # Process Blue
            'PMS 354': (0, 255, 0),      # Process Green
        }
        
        # Calculate distance to each PMS color
        min_distance = float('inf')
        closest_pms = 'PMS Black'  # Default
        
        for pms_name, pms_rgb in pms_colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb_color, pms_rgb)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_pms = pms_name
        
        return closest_pms

    def _create_ultra_fast_potrace_paths(self, arr: np.ndarray, logo_mask: np.ndarray, 
                                       logo_colors: List[Dict], smoothness: float, min_area: int,
                                       use_bezier: bool, simplify_threshold: float,
                                       preserve_details: bool) -> List[Dict]:
        """
        Create vector paths using modern OpenCV-based approach for maximum speed and quality.
        Optimized for logos with multiple colors - uses main color for logos with >3 colors.
        """
        # Performance optimization: If more than 3 colors, use main color only
        if len(logo_colors) > 3:
            self.logger.info(f"Logo has {len(logo_colors)} colors, using main color only for faster processing")
            
            # Find the main (dominant) color
            main_color = None
            max_pixels = 0
            
            for color_info in logo_colors:
                # Safety check: ensure pixel_count exists
                pixel_count = color_info.get('pixel_count', 0)
                if color_info.get('is_dominant', False) or pixel_count > max_pixels:
                    main_color = color_info
                    max_pixels = pixel_count
            
            if main_color is None:
                # Fallback to first color
                main_color = logo_colors[0]
                # Ensure pixel_count exists in fallback
                if 'pixel_count' not in main_color:
                    main_color['pixel_count'] = 1000  # Default value
            
            # Find closest PMS color
            closest_pms = self._find_closest_pms_color(main_color['rgb'])
            pixel_count = main_color.get('pixel_count', 0)
            self.logger.info(f"Using main color: {main_color['hex']} (closest to {closest_pms}) with {pixel_count} pixels")
            
            # Create a simplified color mask using the main color
            rgb_color = np.array(main_color['rgb'], dtype=np.uint8)
            
            # Use a more generous tolerance to capture all logo elements
            tolerance = 100  # Higher tolerance to capture all variations
            color_diff = np.abs(arr[:, :, :3] - rgb_color).sum(axis=2)
            all_logo_pixels = (color_diff <= tolerance) & (logo_mask > 0)
            
            # Create unified mask
            unified_mask = np.zeros_like(logo_mask, dtype=np.uint8)
            unified_mask[all_logo_pixels] = 255
            
            # Clean the mask while preserving details
            if preserve_details:
                kernel = np.ones((2, 2), np.uint8)
                unified_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel)
                unified_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_OPEN, kernel)
            
            # Use advanced CV vectorization on unified mask
            paths = self._vectorize_with_enhanced_detail(unified_mask, arr.shape[1], arr.shape[0], "")
            
            # Convert paths to the expected format with main color
            result_paths = []
            for i, path_data in enumerate(paths):
                result_paths.append({
                    'path_data': path_data,
                    'color': main_color['hex'],
                    'fill_color': main_color['hex'],
                    'stroke_color': main_color['hex'],
                    'stroke_width': 0,
                    'fill_opacity': 1.0,
                    'stroke_opacity': 1.0,
                    'area': 1000,  # Placeholder
                    'is_dominant': True,
                    'pms_color': closest_pms  # Add PMS color information
                })
            
            self.logger.info(f"Generated {len(result_paths)} unified paths with {closest_pms}")
            return result_paths
        
        else:
            # For 3 or fewer colors, process each color individually (original approach)
            from concurrent.futures import ThreadPoolExecutor
            
            def process_color_with_modern_approach(color_info):
                return self._vectorize_color_with_modern_approach(
                    arr, logo_mask, color_info, 
                    smoothness, min_area, use_bezier, 
                    simplify_threshold, preserve_details
                )
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(len(logo_colors), 4)) as executor:
                results = list(executor.map(process_color_with_modern_approach, logo_colors))
            
            # Combine results
            all_paths = []
            for result in results:
                if result:
                    all_paths.extend(result)
            
            return all_paths

    def _vectorize_color_with_modern_approach(self, arr: np.ndarray, logo_mask: np.ndarray, 
                                            color_info: Dict, smoothness: float, min_area: int,
                                            use_bezier: bool, simplify_threshold: float,
                                            preserve_details: bool) -> List[Dict]:
        """
        Vectorize a specific color region using advanced computer vision approach.
        This is the primary method for color vectorization.
        """
        try:
            # Get the RGB color from the color info
            rgb_color = np.array(color_info['rgb'], dtype=np.uint8)
            
            # Create color mask with tolerance for texture variations
            color_variance = color_info.get('color_variance', [30, 30, 30])
            tolerance = max(15, min(80, int(np.mean(color_variance) * 2.0)))
            
            # Calculate color difference
            color_diff = np.abs(arr[:, :, :3] - rgb_color).sum(axis=2)
            color_pixels = (color_diff <= tolerance) & (logo_mask > 0)
            
            # Create color mask
            color_mask = np.zeros_like(logo_mask, dtype=np.uint8)
            color_mask[color_pixels] = 255
            
            # Clean the mask while preserving details
            if preserve_details:
                # Use morphological operations to preserve fine details
                kernel = np.ones((2, 2), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Use advanced CV vectorization
            paths = self._vectorize_with_enhanced_detail(color_mask, arr.shape[1], arr.shape[0], "")
            
            # Convert paths to the expected format
            result_paths = []
            for i, path_data in enumerate(paths):
                result_paths.append({
                    'path_data': path_data,
                    'color': color_info['hex'],
                    'fill_color': color_info['hex'],
                    'stroke_color': color_info['hex'],
                    'stroke_width': 0,
                    'fill_opacity': 1.0,
                    'stroke_opacity': 1.0,
                    'area': cv2.contourArea(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),  # Placeholder
                    'is_dominant': color_info.get('is_dominant', False)
                })
            
            return result_paths
            
        except Exception as e:
            self.logger.warning(f"Modern approach vectorization failed: {e}")
            return []

    def _trace_contour_with_enhanced_detail(self, contour: np.ndarray, image_shape: Tuple[int, int], 
                                          smoothness: float, use_bezier: bool, simplify_threshold: float) -> Optional[str]:
        """
        Trace a single contour with enhanced detail preservation for complex logos.
        """
        try:
            # Create a binary mask for this contour
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Use advanced CV vectorization approach
            paths = self._vectorize_with_enhanced_detail(mask, image_shape[1], image_shape[0], "")
            
            if paths:
                return paths[0]  # Return the first path
            else:
                # Fallback to basic path creation
                points = contour.reshape(-1, 2)
                if len(points) >= 3:
                    return self._create_advanced_path(contour)
                
        except Exception as e:
            self.logger.warning(f"Enhanced contour tracing failed: {e}")
        
        return None

    def _vectorize_with_enhanced_detail(self, mask: np.ndarray, width: int, height: int, temp_dir: str) -> List[str]:
        """
        Vectorize using ultra-enhanced OpenCV-based contour detection for maximum texture and detail preservation.
        Enhanced to capture internal details, holes, and nested contours.
        """
        try:
            # Use advanced computer vision libraries if available
            if ADVANCED_CV_AVAILABLE:
                return self._vectorize_with_advanced_cv(mask, width, height, temp_dir)
            else:
                return self._vectorize_with_basic_opencv(mask, width, height, temp_dir)
            
        except Exception as e:
            self.logger.warning(f"Ultra-enhanced vectorization failed: {e}")
            return self._vectorize_with_basic_opencv(mask, width, height, temp_dir)

    def _vectorize_with_advanced_cv(self, mask: np.ndarray, width: int, height: int, temp_dir: str) -> List[str]:
        """
        Sophisticated vectorization using advanced computer vision libraries.
        Provides much better contour detection and vector ability than basic OpenCV.
        Performance optimized to prevent stalling.
        """
        try:
            import time
            start_time = time.time()
            timeout = 30  # 30 second timeout
            
            # Convert mask to proper format
            mask_float = img_as_float(mask)
            
            # Performance optimization: Limit the number of detection methods for large images
            if width * height > 1000000:  # Large image (>1MP)
                self.logger.info("Large image detected, using optimized detection methods")
                # Use only the most effective methods for large images
                contours_multi_scale = self._detect_multi_scale_contours(mask_float)
                contours_advanced_edges = self._detect_advanced_edges(mask_float)
                
                all_contours = []
                all_contours.extend(contours_multi_scale)
                all_contours.extend(contours_advanced_edges)
            else:
                # For smaller images, use all detection methods
                contours_multi_scale = self._detect_multi_scale_contours(mask_float)
                contours_advanced_edges = self._detect_advanced_edges(mask_float)
                contours_watershed = self._detect_watershed_contours(mask_float)
                contours_regions = self._detect_region_contours(mask_float)
                contours_features = self._detect_feature_contours(mask_float)
                
                # Combine all detection methods
                all_contours = []
                all_contours.extend(contours_multi_scale)
                all_contours.extend(contours_advanced_edges)
                all_contours.extend(contours_watershed)
                all_contours.extend(contours_regions)
                all_contours.extend(contours_features)
            
            # Check timeout
            if time.time() - start_time > timeout:
                self.logger.warning("Advanced CV detection timed out, using basic OpenCV")
                return self._vectorize_with_basic_opencv(mask, width, height, temp_dir)
            
            # Performance optimization: Limit number of contours to prevent stalling
            max_contours = 1000  # Limit to prevent excessive processing
            if len(all_contours) > max_contours:
                self.logger.info(f"Limiting contours from {len(all_contours)} to {max_contours} for performance")
                # Sort by area and keep the largest contours
                all_contours.sort(key=cv2.contourArea, reverse=True)
                all_contours = all_contours[:max_contours]
            
            # Remove duplicates and filter
            unique_contours = self._remove_duplicate_contours_advanced(all_contours)
            
            # Check timeout again
            if time.time() - start_time > timeout:
                self.logger.warning("Advanced CV processing timed out, using basic OpenCV")
                return self._vectorize_with_basic_opencv(mask, width, height, temp_dir)
            
            # Process contours with advanced path generation
            paths = []
            for i, contour in enumerate(unique_contours):
                # Check timeout periodically
                if i % 100 == 0 and time.time() - start_time > timeout:
                    self.logger.warning("Advanced CV path generation timed out, stopping early")
                    break
                
                area = cv2.contourArea(contour)
                if area < 0.1:  # Ultra-small threshold for maximum detail
                    continue
                
                # Use advanced path generation
                path_data = self._create_advanced_path(contour)
                if path_data:
                    paths.append(path_data)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Advanced CV vectorization completed in {processing_time:.2f}s with {len(paths)} paths")
            return paths
            
        except Exception as e:
            self.logger.warning(f"Advanced CV vectorization failed: {e}")
            return self._vectorize_with_basic_opencv(mask, width, height, temp_dir)

    def _detect_multi_scale_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Multi-scale contour detection using different Gaussian blur levels.
        Performance optimized to prevent stalling.
        """
        contours = []
        
        # Performance optimization: Use fewer scales for large images
        if mask.shape[0] * mask.shape[1] > 500000:  # Large image
            scales = [1.0, 2.0]  # Fewer scales
            thresholds = [0.2, 0.4]  # Fewer thresholds
        else:
            scales = [0.5, 1.0, 1.5, 2.0, 3.0]
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for scale in scales:
            # Apply Gaussian blur at different scales
            blurred = gaussian(mask, sigma=scale)
            
            # Threshold at different levels
            for thresh in thresholds:
                binary = blurred > thresh
                
                # Find contours at this scale
                labeled = label(binary)
                regions = regionprops(labeled)
                
                # Limit number of regions to prevent stalling
                max_regions = 100
                if len(regions) > max_regions:
                    # Sort by area and keep largest regions
                    regions = sorted(regions, key=lambda x: x.area, reverse=True)[:max_regions]
                
                for region in regions:
                    if region.area > 5:  # Minimum area
                        # Get contour from region
                        coords = region.coords
                        if len(coords) > 2:
                            contour = np.array(coords, dtype=np.int32)
                            contours.append(contour)
        
        return contours

    def _detect_advanced_edges(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Advanced edge detection using multiple techniques.
        """
        contours = []
        
        # Sobel edge detection
        sobel_edges = sobel(mask)
        sobel_binary = sobel_edges > 0.1
        
        # Find contours from Sobel edges
        labeled_sobel = label(sobel_binary)
        regions_sobel = regionprops(labeled_sobel)
        
        for region in regions_sobel:
            if region.area > 3:
                coords = region.coords
                if len(coords) > 2:
                    contour = np.array(coords, dtype=np.int32)
                    contours.append(contour)
        
        # Canny-like edge detection using multiple thresholds
        edges_low = mask > 0.1
        edges_high = mask > 0.5
        
        # Find contours from edge maps
        for edge_map in [edges_low, edges_high]:
            labeled_edges = label(edge_map)
            regions_edges = regionprops(labeled_edges)
            
            for region in regions_edges:
                if region.area > 2:
                    coords = region.coords
                    if len(coords) > 2:
                        contour = np.array(coords, dtype=np.int32)
                        contours.append(contour)
        
        return contours

    def _detect_watershed_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Watershed segmentation for complex shape detection.
        """
        contours = []
        
        try:
            # Distance transform
            distance = ndimage.distance_transform_edt(mask)
            
            # Find local maxima using scipy
            from scipy.ndimage import maximum_filter
            local_maxi = maximum_filter(distance, size=5) == distance
            local_maxi = np.where(local_maxi)
            local_maxi = np.column_stack(local_maxi)
            
            if len(local_maxi) > 0:
                # Create markers
                markers = np.zeros_like(mask, dtype=int)
                for i, (y, x) in enumerate(local_maxi):
                    markers[y, x] = i + 1
                
                # Watershed segmentation
                labels = watershed(-distance, markers, mask=mask)
                
                # Extract contours from watershed regions
                for label_id in np.unique(labels):
                    if label_id == 0:
                        continue
                    
                    region_mask = labels == label_id
                    labeled_region = label(region_mask)
                    regions = regionprops(labeled_region)
                    
                    for region in regions:
                        if region.area > 5:
                            coords = region.coords
                            if len(coords) > 2:
                                contour = np.array(coords, dtype=np.int32)
                                contours.append(contour)
        
        except Exception as e:
            self.logger.warning(f"Watershed detection failed: {e}")
        
        return contours

    def _detect_region_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Region-based contour detection using SLIC and Felzenszwalb segmentation.
        Performance optimized to prevent stalling.
        """
        contours = []
        
        try:
            # Convert to RGB for segmentation
            mask_rgb = np.stack([mask, mask, mask], axis=-1)
            
            # Performance optimization: Use fewer segments for large images
            if mask.shape[0] * mask.shape[1] > 500000:  # Large image
                n_segments = 20  # Fewer segments
                scale = 50  # Lower scale
            else:
                n_segments = 50
                scale = 100
            
            # SLIC segmentation
            segments_slic = slic(mask_rgb, n_segments=n_segments, compactness=10)
            
            # Felzenszwalb segmentation
            segments_felz = felzenszwalb(mask_rgb, scale=scale, sigma=0.5, min_size=50)
            
            # Process SLIC segments
            for segment_id in np.unique(segments_slic):
                segment_mask = segments_slic == segment_id
                if np.sum(segment_mask) > 10:
                    labeled_segment = label(segment_mask)
                    regions = regionprops(labeled_segment)
                    
                    # Limit regions to prevent stalling
                    max_regions = 50
                    if len(regions) > max_regions:
                        regions = sorted(regions, key=lambda x: x.area, reverse=True)[:max_regions]
                    
                    for region in regions:
                        if region.area > 5:
                            coords = region.coords
                            if len(coords) > 2:
                                contour = np.array(coords, dtype=np.int32)
                                contours.append(contour)
            
            # Process Felzenszwalb segments
            for segment_id in np.unique(segments_felz):
                segment_mask = segments_felz == segment_id
                if np.sum(segment_mask) > 10:
                    labeled_segment = label(segment_mask)
                    regions = regionprops(labeled_segment)
                    
                    # Limit regions to prevent stalling
                    max_regions = 50
                    if len(regions) > max_regions:
                        regions = sorted(regions, key=lambda x: x.area, reverse=True)[:max_regions]
                    
                    for region in regions:
                        if region.area > 5:
                            coords = region.coords
                            if len(coords) > 2:
                                contour = np.array(coords, dtype=np.int32)
                                contours.append(contour)
        
        except Exception as e:
            self.logger.warning(f"Region-based detection failed: {e}")
        
        return contours

    def _detect_feature_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Feature-based contour detection using corner detection and feature points.
        """
        contours = []
        
        try:
            # Harris corner detection
            harris_response = corner_harris(mask, k=0.04)
            corners = corner_peaks(harris_response, min_distance=5, threshold_rel=0.1)
            
            if len(corners) > 2:
                # Create contours from corner points
                for i in range(0, len(corners), 3):
                    if i + 2 < len(corners):
                        corner_group = corners[i:i+3]
                        if len(corner_group) >= 3:
                            contour = np.array(corner_group, dtype=np.int32)
                            contours.append(contour)
            
            # Peak detection for feature points using scipy
            from scipy.ndimage import maximum_filter
            peaks = maximum_filter(mask, size=3) == mask
            peak_coords = np.where(peaks)
            peak_coords = np.column_stack(peak_coords)
            
            if len(peak_coords) > 2:
                # Create contours from peak points
                for i in range(0, len(peak_coords), 4):
                    if i + 3 < len(peak_coords):
                        peak_group = peak_coords[i:i+4]
                        if len(peak_group) >= 3:
                            contour = np.array(peak_group, dtype=np.int32)
                            contours.append(contour)
        
        except Exception as e:
            self.logger.warning(f"Feature-based detection failed: {e}")
        
        return contours

    def _remove_duplicate_contours_advanced(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Advanced duplicate removal using shape similarity and spatial clustering.
        """
        if len(contours) <= 1:
            return contours
        
        unique_contours = []
        
        for contour in contours:
            is_duplicate = False
            
            for existing in unique_contours:
                # Multiple similarity checks
                similarity_cv = cv2.matchShapes(contour, existing, cv2.CONTOURS_MATCH_I1, 0)
                
                # Area similarity
                area_ratio = cv2.contourArea(contour) / max(cv2.contourArea(existing), 1)
                area_similar = 0.8 < area_ratio < 1.2
                
                # Centroid distance
                M_contour = cv2.moments(contour)
                M_existing = cv2.moments(existing)
                
                if M_contour['m00'] > 0 and M_existing['m00'] > 0:
                    cx_contour = int(M_contour['m10'] / M_contour['m00'])
                    cy_contour = int(M_contour['m01'] / M_contour['m00'])
                    cx_existing = int(M_existing['m10'] / M_existing['m00'])
                    cy_existing = int(M_existing['m01'] / M_existing['m00'])
                    
                    centroid_distance = np.sqrt((cx_contour - cx_existing)**2 + (cy_contour - cy_existing)**2)
                    centroid_similar = centroid_distance < 10
                else:
                    centroid_similar = False
                
                # Combined similarity check
                if similarity_cv < 0.01 and area_similar and centroid_similar:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
        
        return unique_contours

    def _create_advanced_path(self, contour: np.ndarray) -> str:
        """
        Create advanced SVG path using sophisticated curve fitting and optimization.
        """
        try:
            if len(contour) < 3:
                return ""
            
            # Convert to points
            points = contour.reshape(-1, 2)
            
            # Use Shapely for advanced geometry processing
            if ADVANCED_CV_AVAILABLE:
                return self._create_shapely_optimized_path(points)
            else:
                return self._create_ultra_detailed_path(points)
            
        except Exception as e:
            self.logger.warning(f"Advanced path creation failed: {e}")
            return self._create_ultra_detailed_path(contour.reshape(-1, 2))

    def _create_shapely_optimized_path(self, points: np.ndarray) -> str:
        """
        Create optimized SVG path using Shapely geometry operations.
        """
        try:
            # Create Shapely polygon
            polygon = Polygon(points)
            
            # Simplify and optimize the polygon
            simplified = polygon.simplify(tolerance=0.1, preserve_topology=True)
            
            # Get exterior coordinates
            if hasattr(simplified, 'exterior'):
                coords = list(simplified.exterior.coords)
            else:
                coords = list(simplified.coords)
            
            if len(coords) < 3:
                return ""
            
            # Create SVG path
            path_data = f"M {coords[0][0]:.1f},{coords[0][1]:.1f}"
            
            for i in range(1, len(coords)):
                path_data += f" L {coords[i][0]:.1f},{coords[i][1]:.1f}"
            
            path_data += " Z"
            return path_data
            
        except Exception as e:
            self.logger.warning(f"Shapely path creation failed: {e}")
            return self._create_ultra_detailed_path(points)

    def _vectorize_with_basic_opencv(self, mask: np.ndarray, width: int, height: int, temp_dir: str) -> List[str]:
        """
        Fallback to basic OpenCV vectorization if advanced libraries are not available.
        """
        try:
            # Find ALL contours including internal details and holes
            contours_external, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            contours_tree, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            contours_list, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            contours_ccomp, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            
            # Combine all contours for maximum coverage
            all_contours = []
            all_contours.extend(contours_external)
            all_contours.extend(contours_tree)
            all_contours.extend(contours_list)
            all_contours.extend(contours_ccomp)
            
            # Remove duplicates and very similar contours
            unique_contours = self._remove_duplicate_contours(all_contours)
            
            # Process contours with hierarchy information for internal details
            paths = []
            for i, contour in enumerate(unique_contours):
                # Filter out very small contours but keep internal details
                area = cv2.contourArea(contour)
                if area < 0.5:  # Ultra-small threshold for internal details
                    continue
                
                # Use ultra-minimal approximation to preserve all texture
                epsilon = 0.0001 * cv2.arcLength(contour, True)  # Extremely small epsilon
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                # Convert to points
                points = approx.reshape(-1, 2)
                
                # Create ultra-detailed SVG path with texture preservation
                path_data = self._create_ultra_detailed_path(points)
                if path_data:
                    paths.append(path_data)
            
            # Also process internal details using hierarchy
            if hierarchy is not None:
                internal_paths = self._process_internal_details(mask, hierarchy, contours_tree)
                paths.extend(internal_paths)
            
            return paths
            
        except Exception as e:
            self.logger.warning(f"Basic OpenCV vectorization failed: {e}")
            return []

    def _process_internal_details(self, mask: np.ndarray, hierarchy: np.ndarray, contours: List[np.ndarray]) -> List[str]:
        """
        Process internal details, holes, and nested contours using hierarchy information.
        """
        paths = []
        
        try:
            # Process hierarchy to find internal details
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # Check if this is an internal contour (hole)
                if h[3] >= 0:  # Has parent (internal detail)
                    area = cv2.contourArea(contour)
                    if area > 0.1:  # Small threshold for internal details
                        # Use ultra-minimal approximation
                        epsilon = 0.0001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(approx) >= 3:
                            points = approx.reshape(-1, 2)
                            path_data = self._create_ultra_detailed_path(points)
                            if path_data:
                                paths.append(path_data)
            
            # Also find internal details using flood fill approach
            internal_details = self._find_internal_details_flood_fill(mask)
            for detail_contour in internal_details:
                area = cv2.contourArea(detail_contour)
                if area > 0.1:
                    epsilon = 0.0001 * cv2.arcLength(detail_contour, True)
                    approx = cv2.approxPolyDP(detail_contour, epsilon, True)
                    
                    if len(approx) >= 3:
                        points = approx.reshape(-1, 2)
                        path_data = self._create_ultra_detailed_path(points)
                        if path_data:
                            paths.append(path_data)
            
        except Exception as e:
            self.logger.warning(f"Internal detail processing failed: {e}")
        
        return paths

    def _find_internal_details_flood_fill(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find internal details using flood fill approach to detect holes and internal shapes.
        """
        internal_contours = []
        
        try:
            # Create a copy of the mask
            working_mask = mask.copy()
            
            # Find external contours first
            external_contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # For each external contour, look for internal details
            for ext_contour in external_contours:
                # Create a mask for this external contour
                contour_mask = np.zeros_like(working_mask)
                cv2.fillPoly(contour_mask, [ext_contour], 255)
                
                # Find internal contours within this external contour
                internal_contours_in_ext, _ = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                
                for internal_contour in internal_contours_in_ext:
                    area = cv2.contourArea(internal_contour)
                    if 0.1 < area < cv2.contourArea(ext_contour) * 0.95:  # Internal detail, not the main contour
                        internal_contours.append(internal_contour)
            
            # Also use morphological operations to find internal details
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            internal_mask = cv2.subtract(mask, eroded)
            
            if np.any(internal_mask):
                internal_contours_from_morph, _ = cv2.findContours(internal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                for contour in internal_contours_from_morph:
                    area = cv2.contourArea(contour)
                    if area > 0.1:
                        internal_contours.append(contour)
            
        except Exception as e:
            self.logger.warning(f"Flood fill internal detail detection failed: {e}")
        
        return internal_contours

    def _detect_enhanced_contours(self, color_info: Dict) -> List[np.ndarray]:
        """
        Enhanced contour detection using advanced CV approach.
        Simplified since the advanced CV system handles this better.
        """
        # This method is kept for compatibility but the advanced CV system is preferred
        # The advanced CV system in _vectorize_with_advanced_cv handles all contour detection
        return []

    def _extract_internal_details_from_hierarchy(self, contours: List[np.ndarray], hierarchy: np.ndarray) -> List[np.ndarray]:
        """
        Extract internal details from contour hierarchy.
        Simplified since advanced CV handles this better.
        """
        internal_contours = []
        
        try:
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # Check if this is an internal contour (has parent)
                if h[3] >= 0:  # Has parent
                    area = cv2.contourArea(contour)
                    if area > 0.05:  # Small threshold for internal details
                        internal_contours.append(contour)
        
        except Exception as e:
            self.logger.warning(f"Hierarchy extraction failed: {e}")
        
        return internal_contours

    def _create_ultra_detailed_path(self, points: np.ndarray) -> str:
        """
        Create ultra-detailed SVG path using advanced CV approach.
        """
        return self._create_advanced_path(np.array(points).reshape(-1, 1, 2))

    def _create_enhanced_detail_path(self, points: np.ndarray) -> str:
        """
        Create enhanced detail SVG path using advanced CV approach.
        """
        return self._create_advanced_path(np.array(points).reshape(-1, 1, 2))

    def _create_enhanced_backup_paths(self, contours: List[np.ndarray], color_info: Dict, 
                                    min_area: int, smoothness: float, use_bezier: bool) -> List[Dict]:
        """
        Enhanced backup method for creating vector paths when primary method fails.
        This method is kept as a fallback but the advanced CV system is preferred.
        """
        paths = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Use advanced path creation
            path_data = self._create_advanced_path(contour)
            
            if path_data:
                paths.append({
                    'path_data': path_data,
                    'color': color_info['hex'],
                    'area': area,
                    'is_dominant': color_info.get('is_dominant', False),
                    'quality': 'backup',
                    'contour_index': i
                })
        
        return paths

    def _preprocess_contour_for_maximum_detail(self, contour: np.ndarray, smoothness: float) -> np.ndarray:
        """
        Contour preprocessing for maximum detail preservation.
        Simplified version since advanced CV handles this better.
        """
        if len(contour) < 3:
            return contour
        
        # Use minimal simplification to preserve details
        arc_length = cv2.arcLength(contour, True)
        epsilon = max(0.01, arc_length * 0.001 * smoothness)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        return simplified

    def _remove_ultra_minimal_redundant_points(self, points: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Remove redundant points while preserving details.
        Simplified version since advanced CV handles this better.
        """
        if len(points) < 4:
            return points
        
        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        
        # Keep points that are far enough apart
        keep_indices = [0]
        for i, dist in enumerate(distances):
            if dist > threshold:
                keep_indices.append(i + 1)
        
        if len(points) - 1 not in keep_indices:
            keep_indices.append(len(points) - 1)
        
        return points[keep_indices]

    def _create_maximum_detail_path(self, contour: np.ndarray, smoothness: float, use_bezier: bool) -> str:
        """
        Create maximum detail path using advanced CV approach.
        """
        return self._create_advanced_path(contour)

    def _create_simple_paths_fallback(self, contours: List[np.ndarray], color_info: Dict, min_area: int) -> List[Dict]:
        """
        Simple fallback method for basic path creation.
        """
        paths = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            path_data = self._create_advanced_path(contour)
            
            if path_data:
                paths.append({
                    'path_data': path_data,
                    'color': color_info['hex'],
                    'area': area,
                    'is_dominant': color_info.get('is_dominant', False)
                })
        
        return paths

    def _create_parallel_color_paths(self, arr: np.ndarray, logo_mask: np.ndarray, 
                                   logo_colors: List[Dict], smoothness: float, min_area: int,
                                   use_bezier: bool, simplify_threshold: float,
                                   preserve_details: bool) -> List[Dict]:
        """
        Create vector paths using parallel processing for multiple colors.
        Simplified to use the advanced CV approach.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        def process_color(color_info):
            return self._vectorize_color_with_modern_approach(
                arr, logo_mask, color_info, 
                smoothness, min_area, use_bezier, 
                simplify_threshold, preserve_details
            )
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(logo_colors), 4)) as executor:
            results = list(executor.map(process_color, logo_colors))
        
        # Combine results
        all_paths = []
        for result in results:
            if result:
                all_paths.extend(result)
        
        return all_paths

    def _create_optimized_color_paths(self, arr: np.ndarray, logo_mask: np.ndarray, 
                                    color_info: Dict, smoothness: float, min_area: int,
                                    use_bezier: bool, simplify_threshold: float,
                                    preserve_details: bool) -> List[Dict]:
        """
        Create optimized vector paths using advanced CV approach.
        """
        # Use the modern approach which leverages advanced CV
        return self._vectorize_color_with_modern_approach(
            arr, logo_mask, color_info, 
            smoothness, min_area, use_bezier, 
            simplify_threshold, preserve_details
        )

    def _clean_color_mask_fast(self, mask: np.ndarray, preserve_details: bool) -> np.ndarray:
        """
        Fast color mask cleaning while preserving details.
        """
        if preserve_details:
            # Use minimal morphological operations
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def _create_optimized_path(self, contour: np.ndarray, smoothness: float, 
                             use_bezier: bool, simplify_threshold: float,
                             preserve_details: bool) -> str:
        """
        Create optimized path using advanced CV approach.
        """
        return self._create_advanced_path(contour)

    def _create_simple_bezier_path(self, points: np.ndarray, smoothness: float) -> str:
        """
        Create simple Bezier path. Simplified since advanced CV handles this better.
        """
        if len(points) < 3:
            return ""
        
        # Use line segments for better detail preservation
        path_data = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
        for point in points[1:]:
            path_data += f" L {point[0]:.1f},{point[1]:.1f}"
        path_data += " Z"
        
        return path_data

    def _create_fast_linear_path(self, points: np.ndarray, simplify_threshold: float) -> str:
        """
        Create fast linear path. Simplified since advanced CV handles this better.
        """
        if len(points) < 3:
            return ""
        
        # Use line segments for detail preservation
        path_data = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
        for point in points[1:]:
            path_data += f" L {point[0]:.1f},{point[1]:.1f}"
        path_data += " Z"
        
        return path_data

    def _build_optimized_svg(self, width: int, height: int, paths: List[Dict], opts: Dict) -> str:
        """
        Build optimized SVG with minimal content for speed.
        """
        # Sort paths by area and dominance
        paths.sort(key=lambda x: (x.get('is_dominant', False), x.get('area', 0)), reverse=True)
        
        # Optimized SVG with minimal styling
        svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      path {{ stroke: none; }}
    </style>
  </defs>
'''
        
        for path_info in paths:
            svg_content += f'  <path d="{path_info["path_data"]}" fill="{path_info["color"]}" />\n'
        
        svg_content += '</svg>'
        return svg_content

    def process_logo(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        Main entry point for processing a logo with parallel processing for all variations.
        """
        start_time = time.time()
        self.logger.info(f"Starting logo processing for {file_path}")
        self.logger.info(f"Options received: {options}")
        all_outputs = {}
        errors = []

        # Check if any processing option is selected
        has_vector_trace = options.get('vector_trace', False)
        has_full_color_vector_trace = options.get('full_color_vector_trace', False)
        has_contour_cut = options.get('contour_cut', False)
        has_transparent_png = options.get('transparent_png', False)
        has_black_version = options.get('black_version', False)
        has_distressed_effect = options.get('distressed_effect', False)
        has_favicon = options.get('favicon', False)  # Add favicon option
        social_options = options.get('social_formats', {})
        has_social_media = any(social_options.values()) if isinstance(social_options, dict) else False
        has_color_separations = options.get('color_separations', False)
        
        # Debug logging for option detection
        self.logger.info(f"Option detection:")
        self.logger.info(f"  vector_trace: {has_vector_trace}")
        self.logger.info(f"  full_color_vector_trace: {has_full_color_vector_trace}")
        self.logger.info(f"  contour_cut: {has_contour_cut}")
        self.logger.info(f"  transparent_png: {has_transparent_png}")
        self.logger.info(f"  black_version: {has_black_version}")
        self.logger.info(f"  distressed_effect: {has_distressed_effect}")
        self.logger.info(f"  social_media: {has_social_media}")
        self.logger.info(f"  color_separations: {has_color_separations}")
        self.logger.info(f"  favicon: {has_favicon}")
        
        if not (has_vector_trace or has_full_color_vector_trace or has_contour_cut or has_transparent_png or has_black_version or has_distressed_effect or has_social_media or has_color_separations or has_favicon):
            self.logger.error("No processing options selected - all options are False")
            return {'success': False, 'message': 'No processing options selected. Please select at least one option.'}

        # Prepare tasks for parallel processing
        tasks = []
        
        # Add basic variations (fast processing)
        if has_transparent_png:
            tasks.append(('transparent_png', self._process_transparent_png_wrapper, file_path))
        
        if has_black_version:
            tasks.append(('black_version', self._process_black_version_wrapper, file_path))
            
        if has_distressed_effect:
            tasks.append(('distressed_effect', self._process_distressed_effect_wrapper, file_path))
            
        if has_color_separations:
            tasks.append(('color_separations', self._process_color_separations_wrapper, file_path))
            
        if has_contour_cut:
            tasks.append(('contour_cut', self._process_contour_cutline_wrapper, file_path))
            
        if has_favicon:
            tasks.append(('favicon', self._process_favicon_wrapper, file_path))

        # Add vector variations (heavier processing)
        if has_vector_trace:
            self.logger.info("Adding vector_trace task to processing queue")
            vector_options = options.get('vector_trace_options', {})
            self.logger.info(f"Vector trace options: {vector_options}")
            tasks.append(('vector_trace', self._process_vector_trace_wrapper, file_path, vector_options))
            
        if has_full_color_vector_trace:
            self.logger.info("Adding full_color_vector_trace task to processing queue")
            full_color_options = options.get('full_color_vector_trace_options', {})
            self.logger.info(f"Full color vector trace options: {full_color_options}")
            tasks.append(('full_color_vector_trace', self._process_full_color_vector_trace_wrapper, file_path, full_color_options))

        self.logger.info(f"Total tasks prepared: {len(tasks)}")
        for task in tasks:
            self.logger.info(f"  Task: {task[0]}")

        # Process all tasks in parallel with advanced optimization
        if tasks:
            self.logger.info(f"Processing {len(tasks)} variations with intelligent parallel processing")
            parallel_start_time = time.time()
            
            # Determine optimal processing strategy
            cpu_count = multiprocessing.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Sort tasks by priority for better scheduling
            tasks.sort(key=lambda x: self._get_task_priority(x[0]), reverse=True)
            
            # Get optimal processing strategy
            strategy = self._get_optimal_processing_strategy(tasks, cpu_count, memory_gb)
            self.logger.info(f"Selected strategy: {strategy['description']}")
            
            # Process tasks using the selected strategy
            if strategy['strategy'] == 'thread_pool':
                # Use ThreadPoolExecutor for I/O bound tasks
                max_workers = strategy['max_workers']
                self.logger.info(f"Using ThreadPoolExecutor with {max_workers} workers")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {executor.submit(task[1], *task[2:]): task[0] for task in tasks}
                    
                    batch_results = {}
                    completed_tasks = 0
                    
                    for future in as_completed(future_to_task, timeout=self.parallel_config['task_timeout']):
                        task_name = future_to_task[future]
                        completed_tasks += 1
                        
                        try:
                            result = future.result(timeout=30)
                            batch_results[task_name] = result
                            self.logger.info(f"✅ ThreadPool completed {task_name} ({completed_tasks}/{len(tasks)})")
                        except Exception as e:
                            error_msg = f"ThreadPool task {task_name} failed: {str(e)}"
                            self.logger.error(error_msg)
                            errors.append(error_msg)
                            batch_results[task_name] = None
            
            elif strategy['strategy'] == 'thread_pool_optimized':
                # Use optimized ThreadPoolExecutor for maximum concurrency
                max_workers = strategy['max_workers']
                self.logger.info(f"Using optimized ThreadPoolExecutor with {max_workers} workers")
                batch_results = self._process_with_thread_pool_optimized(tasks, max_workers)
            
            elif strategy['strategy'] == 'hybrid':
                # Use hybrid approach for mixed workloads
                self.logger.info(f"Using hybrid strategy with {strategy['thread_workers']} thread workers and {strategy['process_workers']} CPU workers")
                batch_results = self._process_with_hybrid_strategy(tasks, strategy)
            
            elif strategy['strategy'] == 'async_single':
                # Use async processing for single tasks
                self.logger.info("Using async single-threaded processing")
                batch_results = self._process_with_async_strategy(tasks)
            
            parallel_time = time.time() - parallel_start_time
            self.logger.info(f"Intelligent parallel processing completed in {parallel_time:.2f}s using {strategy['strategy']} strategy")
            
            # Process batch results
            for task_name, result in batch_results.items():
                self.logger.info(f"Processing result for {task_name}: {type(result)}")
                self.logger.info(f"Result content for {task_name}: {result}")
                
                if result and isinstance(result, dict):
                    # Handle different result formats
                    if task_name == 'black_version' and 'smooth_gray' in result:
                        all_outputs['smooth_gray_version'] = result['smooth_gray']
                    if task_name == 'black_version' and 'pure_black' in result:
                        all_outputs['black_version'] = result['pure_black']
                    elif task_name == 'vector_trace' and result.get('status') == 'success':
                        self.logger.info(f"Vector trace success detected for {task_name}")
                        output_paths = result.get('output_paths', {})
                        self.logger.info(f"Vector trace output paths: {output_paths}")
                        if output_paths.get('svg'):
                            all_outputs['vector_trace_svg'] = output_paths['svg']
                            self.logger.info(f"Added vector_trace_svg: {output_paths['svg']}")
                        if output_paths.get('pdf'):
                            all_outputs['vector_trace_pdf'] = output_paths['pdf']
                            self.logger.info(f"Added vector_trace_pdf: {output_paths['pdf']}")
                        if output_paths.get('ai'):
                            all_outputs['vector_trace_ai'] = output_paths['ai']
                            self.logger.info(f"Added vector_trace_ai: {output_paths['ai']}")
                    elif task_name == 'vector_trace':
                        self.logger.error(f"Vector trace failed - status: {result.get('status')}, message: {result.get('message')}")
                        error_msg = f"Vector trace processing failed: {result.get('message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                    elif task_name == 'full_color_vector_trace' and result.get('status') == 'success':
                        output_paths = result.get('output_paths', {})
                        if output_paths.get('svg'):
                            all_outputs['full_color_vector_trace_svg'] = output_paths['svg']
                        if output_paths.get('pdf'):
                            all_outputs['full_color_vector_trace_pdf'] = output_paths['pdf']
                        if output_paths.get('ai'):
                            all_outputs['full_color_vector_trace_ai'] = output_paths['ai']
                        colors_used = result.get('colors_used', 0)
                        all_outputs['full_color_vector_trace_info'] = {'colors_used': colors_used}
                    elif task_name == 'contour_cut' and isinstance(result, dict):
                        all_outputs['contour_cut'] = result
                    elif task_name == 'color_separations' and isinstance(result, dict):
                        all_outputs['color_separations'] = result
                    else:
                        error_msg = f"{task_name} processing did not produce valid output"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                elif result and isinstance(result, str):  # Simple path result (transparent_png, distressed_effect, favicon)
                    if task_name == 'favicon':
                        all_outputs['favicon'] = result
                    else:
                        all_outputs[task_name] = result
                else:
                    error_msg = f"{task_name} processing did not produce valid output"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

        # Process social media variations with advanced parallel processing
        if has_social_media:
            platforms_to_process = [platform for platform, enabled in social_options.items() if enabled]
            self.logger.info(f"Processing social media variations for: {platforms_to_process}")
            social_start_time = time.time()

            # Use optimized worker allocation for social media processing
            optimal_workers = self._get_optimal_worker_count('social_media')

            with ThreadPoolExecutor(max_workers=optimal_workers, thread_name_prefix='logo_processor_social') as executor:
                future_to_platform = {
                    executor.submit(self.process_social_variation, file_path, platform, options): platform
                    for platform in platforms_to_process
                }

                # Collect results with timeout
                for future in as_completed(future_to_platform):
                    platform = future_to_platform[future]
                    try:
                        result = future.result(timeout=self.parallel_config['task_timeout'])
                        if result and result.get('status') == 'success':
                            output_key = f"social_{platform}"
                            all_outputs[output_key] = result.get('output_path')
                            
                            # Add additional outputs (favicon, webp, etc.)
                            additional_outputs = result.get('additional_outputs', {})
                            for output_type, output_path in additional_outputs.items():
                                if output_type == 'favicon':
                                    all_outputs[f"social_{platform}_favicon"] = output_path
                                elif output_type == 'webp':
                                    all_outputs[f"social_{platform}_webp"] = output_path
                        else:
                            error_msg = f"Failed to process social variation for {platform}: {result.get('message', 'Unknown error')}"
                            self.logger.error(error_msg)
                            errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Exception processing social variation for {platform}: {e}"
                        self.logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)

            social_time = time.time() - social_start_time
            self.logger.info(f"Social media processing completed in {social_time:.2f}s for {len(platforms_to_process)} platforms")

        # Update performance stats
        total_time = time.time() - start_time
        with self.stats_lock:
            self.processing_stats['total_processed'] += 1
            if not errors:
                self.processing_stats['success'] += 1
            else:
                self.processing_stats['failures'] += 1

            # Update average processing time
            current_avg = self.processing_stats['avg_processing_time']
            total_processed = self.processing_stats['total_processed']
            self.processing_stats['avg_processing_time'] = (current_avg * (total_processed - 1) + total_time) / total_processed

        # Final result
        if not all_outputs:
            final_message = 'Processing failed. No variations were generated.'
            if errors:
                final_message += " Details: " + "; ".join(errors)
            return {'success': False, 'message': final_message, 'errors': errors}

        success_message = f"Successfully generated {len(all_outputs)} variations using parallel processing in {total_time:.2f}s."
        if errors:
            success_message += f" However, {len(errors)} error(s) occurred: " + "; ".join(errors)

        self.logger.info(f"Total processing time: {total_time:.2f}s for {len(all_outputs)} outputs")

        return {
            'success': True,
            'message': success_message,
            'outputs': all_outputs,
            'errors': errors,
            'processing_time': total_time,
            'parallel_processing_used': True
        }

    def _process_transparent_png_wrapper(self, file_path: str) -> str:
        """Wrapper for transparent PNG processing."""
        try:
            return self._create_transparent_png(file_path)
        except Exception as e:
            self.logger.error(f"Error in transparent PNG wrapper: {e}", exc_info=True)
            return None

    def _process_black_version_wrapper(self, file_path: str) -> dict:
        """Wrapper for black version processing."""
        try:
            return self._create_black_version(file_path)
        except Exception as e:
            self.logger.error(f"Error in black version wrapper: {e}", exc_info=True)
            return None

    def _process_distressed_effect_wrapper(self, file_path: str) -> str:
        """Wrapper for distressed effect processing."""
        try:
            return self._create_distressed_version(file_path)
        except Exception as e:
            self.logger.error(f"Error in distressed effect wrapper: {e}", exc_info=True)
            return None

    def _process_color_separations_wrapper(self, file_path: str) -> dict:
        """Wrapper for color separations processing."""
        try:
            return self._create_color_separations(file_path)
        except Exception as e:
            self.logger.error(f"Error in color separations wrapper: {e}", exc_info=True)
            return None

    def _process_contour_cutline_wrapper(self, file_path: str) -> dict:
        """Wrapper for contour cutline processing."""
        try:
            return self._create_contour_cutline(file_path)
        except Exception as e:
            self.logger.error(f"Error in contour cutline wrapper: {e}", exc_info=True)
            return None

    def _process_vector_trace_wrapper(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Wrapper for vector trace processing."""
        try:
            self.logger.info(f"Vector trace wrapper called with file: {file_path}")
            self.logger.info(f"Vector trace wrapper options: {options}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Call the vector trace method
            result = self.generate_vector_trace(file_path, options)
            self.logger.info(f"Vector trace wrapper result: {result}")
            self.logger.info(f"Vector trace result type: {type(result)}")
            self.logger.info(f"Vector trace result status: {result.get('status') if isinstance(result, dict) else 'Not a dict'}")
            
            # Validate the result
            if not result:
                error_msg = "Vector trace returned None result"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            if not isinstance(result, dict):
                error_msg = f"Vector trace returned non-dict result: {type(result)}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            if result.get('status') != 'success':
                error_msg = f"Vector trace failed with status: {result.get('status')}, message: {result.get('message')}"
                self.logger.error(error_msg)
                return result
            
            # Check if output paths exist
            output_paths = result.get('output_paths', {})
            for path_type, path in output_paths.items():
                if path and not os.path.exists(path):
                    error_msg = f"Vector trace output file not found: {path_type} = {path}"
                    self.logger.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
            
            self.logger.info(f"Vector trace wrapper returning successful result")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in vector trace wrapper: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _process_full_color_vector_trace_wrapper(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Wrapper for full color vector trace processing."""
        try:
            return self.generate_full_color_vector_trace(file_path, options)
        except Exception as e:
            self.logger.error(f"Error in full color vector trace wrapper: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: {self.cache_dir}")

    def _ensure_upload_dir(self):
        """Ensure the upload directory exists."""
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
            self.logger.info(f"Created upload directory: {self.upload_folder}")

    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.logger.info(f"Created output directory: {self.output_folder}")

    def _ensure_temp_dir(self):
        """Ensure the temp directory exists."""
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
            self.logger.info(f"Created temp directory: {self.temp_folder}")

    def _create_transparent_png(self, file_path: str) -> str:
        """
        Create a high-quality transparent PNG using smart background removal (rembg if available).
        Returns the output file path.
        """
        from PIL import Image
        import os
        try:
            img = Image.open(file_path).convert("RGBA")
            output_path = os.path.join(self.output_folder, f"transparent_{os.path.basename(file_path)}")
            try:
                from rembg import remove
                img_no_bg = remove(img)
                img_no_bg.save(output_path, format="PNG", optimize=True)
            except ImportError:
                # Fallback: naive alpha mask for white backgrounds
                datas = img.getdata()
                newData = []
                for item in datas:
                    if item[0] > 240 and item[1] > 240 and item[2] > 240:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)
                img.putdata(newData)
                img.save(output_path, format="PNG", optimize=True)
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating transparent PNG: {str(e)}", exc_info=True)
            return None

    def _create_black_version(self, file_path: str) -> dict:
        """
        Create both a high-quality smooth grayscale version and a pure black version 
        that maintains logo integrity and smoothness using advanced thresholding and smoothing.
        Returns a dict with both output paths.
        """
        from PIL import Image, ImageFilter, ImageOps, ImageEnhance
        import numpy as np
        import os
        try:
            img = Image.open(file_path).convert("RGBA")
            w, h = img.size
            
            # Create grayscale version with proper alpha handling
            # Use weighted RGB to grayscale conversion for better quality
            r, g, b, a = img.split()
            # Weighted grayscale conversion (luminance-preserving)
            gray = Image.fromarray(
                (0.299 * np.array(r) + 0.587 * np.array(g) + 0.114 * np.array(b)).astype(np.uint8),
                mode='L'
            )
            
            # Apply gentle smoothing to reduce noise while preserving edges
            smooth_gray = gray.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Enhance contrast slightly for better definition
            enhanced_gray = ImageEnhance.Contrast(smooth_gray).enhance(1.1)
            
            # Create smooth grayscale version (preserves all detail)
            smooth_output_path = os.path.join(self.output_folder, f"smooth_gray_{os.path.basename(file_path)}")
            # Create RGBA version with original alpha
            smooth_rgba = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            smooth_rgba.paste(enhanced_gray, (0, 0), a)  # Use original alpha channel
            smooth_rgba.save(smooth_output_path, format="PNG", optimize=True)
            
            # Create pure black version using advanced thresholding
            # Use Otsu's method for automatic threshold determination
            gray_array = np.array(enhanced_gray)
            alpha_array = np.array(a)
            
            # Only process pixels where alpha > 0
            valid_pixels = gray_array[alpha_array > 0]
            if len(valid_pixels) == 0:
                # Fallback if no valid pixels
                threshold = 128
            else:
                # Use Otsu's method for optimal threshold
                from skimage.filters import threshold_otsu
                try:
                    threshold = threshold_otsu(valid_pixels)
                except:
                    # Fallback to mean-based threshold
                    threshold = np.mean(valid_pixels)
            
            # Apply threshold with slight bias towards preserving detail
            # Use a slightly lower threshold to preserve more of the logo
            adjusted_threshold = max(threshold * 0.9, threshold - 10)
            
            # Create binary mask
            binary_mask = (gray_array > adjusted_threshold).astype(np.uint8) * 255
            
            # Clean up the binary mask with morphological operations
            kernel = np.ones((2, 2), np.uint8)
            # Remove small noise
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            # Fill small holes
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply anti-aliasing to smooth edges
            # Use a small Gaussian blur and re-threshold for smooth edges
            smooth_mask = cv2.GaussianBlur(cleaned_mask.astype(np.float32), (3, 3), 0.5)
            smooth_mask = (smooth_mask > 127).astype(np.uint8) * 255
            
            # Create final black version
            result_arr = np.zeros((h, w, 4), dtype=np.uint8)
            result_arr[..., 0:3] = 0  # Black
            # Only set alpha where original alpha > 0 AND our mask is black (logo area)
            result_arr[..., 3] = np.where((smooth_mask == 0) & (alpha_array > 0), 255, 0)
            
            result = Image.fromarray(result_arr, mode="RGBA")
            black_output_path = os.path.join(self.output_folder, f"black_{os.path.basename(file_path)}")
            result.save(black_output_path, format="PNG", optimize=True)
            
            self.logger.info(f"Created both smooth grayscale and pure black versions")
            return {
                'smooth_gray': smooth_output_path,
                'pure_black': black_output_path
            }

        except Exception as e:
            self.logger.error(f"Error creating black version PNG: {str(e)}", exc_info=True)
            return None

    def _create_contour_cutline(self, file_path: str) -> dict:
        """
        Generates BOTH:
        1. A PNG/PDF with the magenta cut path drawn on the raster image for preview.
        2. A vector SVG/PDF with the raster image as background and a true spot color vector path for RIP/cutting.
        Returns a dict with keys: 'png', 'pdf_preview', 'svg', 'pdf_vector'.
        Now includes smart object detection for logos with backgrounds and smart contour hierarchy detection (holes/cutouts).
        Always auto-removes background before contour detection.
        """
        import io
        from PIL import ImageDraw
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        import cairosvg
        import base64
        try:
            # 1. Load image
            img = Image.open(file_path).convert("RGBA")
            img_np = np.array(img)
            width, height = img.size
            # --- Always auto-remove background ---
            alpha_channel = img_np[:, :, 3]
            if np.max(alpha_channel) == 0 or np.min(alpha_channel) == 255:
                corners = [img_np[0,0,:3], img_np[0,-1,:3], img_np[-1,0,:3], img_np[-1,-1,:3]]
                flat = img_np[:,:,:3].reshape(-1, 3)
                from collections import Counter
                most_common = Counter([tuple(x) for x in flat]).most_common(1)[0][0]
                bg_candidates = corners + [most_common]
                bg_color = Counter([tuple(x) for x in bg_candidates]).most_common(1)[0][0]
                tolerance = 18
                diff = np.abs(img_np[:,:,:3] - np.array(bg_color)).sum(axis=2)
                new_alpha = np.where(diff <= tolerance, 0, 255).astype(np.uint8)
                img_np[:, :, 3] = new_alpha
                img = Image.fromarray(img_np, 'RGBA')
                alpha_channel = img_np[:, :, 3]
            # --- Create mask for logo ---
            mask = (alpha_channel > 0).astype(np.uint8) * 255
            # Clean up mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            # --- Find contours and hierarchy ---
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours or hierarchy is None:
                self.logger.error("No contours found for cutline.")
                return None
            hierarchy = hierarchy[0]
            # Filter out contours that are the image border
            def is_border_contour(cnt, w, h, tol=2):
                x, y, cw, ch = cv2.boundingRect(cnt)
                return x <= tol and y <= tol and abs((x+cw)-w) <= tol and abs((y+ch)-h) <= tol
            filtered = [(i, c) for i, c in enumerate(contours) if not is_border_contour(c, width, height)]
            if not filtered:
                filtered = list(enumerate(contours))  # fallback: use all
            # --- 1. Raster preview: draw all contours (outer and holes) ---
            overlay = img.copy()
            draw = ImageDraw.Draw(overlay)
            magenta = (255, 0, 255, 255)
            stroke_width = max(1, int(0.25 * (width / 1000)))
            for idx, c in filtered:
                points = [(int(p[0][0]), int(p[0][1])) for p in c]
                if len(points) > 1:
                    draw.line(points + [points[0]], fill=magenta, width=stroke_width)
            # Save as PNG
            png_filename = f"contour_cutline_{int(time.time())}.png"
            png_output_path = os.path.join(self.output_folder, png_filename)
            overlay.save(png_output_path)
            # Save as raster PDF
            pdf_preview_filename = f"contour_cutline_{int(time.time())}_preview.pdf"
            pdf_preview_output_path = os.path.join(self.output_folder, pdf_preview_filename)
            c = canvas.Canvas(pdf_preview_output_path, pagesize=(width, height))
            c.drawImage(ImageReader(png_output_path), 0, 0, width=width, height=height)
            c.showPage()
            c.save()
            # --- 2. Vector SVG/PDF: raster image as background, vector path on top ---
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_b64}"
            svg_header = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" fill-rule="evenodd">
  <defs>
    <linearGradient id="CutContour" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#FF00FF"/>
    </linearGradient>
  </defs>
  <image href="{img_data_url}" x="0" y="0" width="{width}" height="{height}"/>
'''
            # --- Smart hierarchy: build compound path for outer and holes ---
            svg_paths = []
            used = set()
            for idx, c in filtered:
                if hierarchy[idx][3] == -1:  # Outer contour (no parent)
                    # Start compound path
                    path_data = "M " + " L ".join(f"{p[0][0]},{p[0][1]}" for p in c) + " Z"
                    # Find all holes (children)
                    child_idx = hierarchy[idx][2]
                    while child_idx != -1:
                        child = contours[child_idx]
                        child_path = "M " + " L ".join(f"{p[0][0]},{p[0][1]}" for p in child) + " Z"
                        path_data += " " + child_path
                        used.add(child_idx)
                        child_idx = hierarchy[child_idx][0]  # Next child
                    svg_paths.append(f'<path d="{path_data}" fill="none" stroke="url(#CutContour)" stroke-width="0.25pt" />')
            # Add any remaining holes as separate paths (if not already included)
            for idx, c in filtered:
                if idx in used:
                    continue
                if hierarchy[idx][3] != -1:  # Child contour (hole)
                    path_data = "M " + " L ".join(f"{p[0][0]},{p[0][1]}" for p in c) + " Z"
                    svg_paths.append(f'<path d="{path_data}" fill="none" stroke="url(#CutContour)" stroke-width="0.25pt" />')
            svg_content = (svg_header + "".join(svg_paths) + "</svg>").encode('utf-8')
            # Save SVG file
            svg_filename = f"contour_cutline_{int(time.time())}.svg"
            svg_output_path = os.path.join(self.output_folder, svg_filename)
            with open(svg_output_path, 'wb') as svg_file:
                svg_file.write(svg_content)
            # Convert SVG to vector PDF
            pdf_vector_filename = f"contour_cutline_{int(time.time())}_vector.pdf"
            pdf_vector_output_path = os.path.join(self.output_folder, pdf_vector_filename)
            cairosvg.svg2pdf(bytestring=svg_content, write_to=pdf_vector_output_path)
            self.logger.info(f"Contour cutline PNG file saved to {png_output_path}")
            self.logger.info(f"Contour cutline PDF preview file saved to {pdf_preview_output_path}")
            self.logger.info(f"Contour cutline SVG file saved to {svg_output_path}")
            self.logger.info(f"Contour cutline vector PDF file saved to {pdf_vector_output_path}")
            return {
                'png': png_output_path,
                'pdf_preview': pdf_preview_output_path,
                'svg': svg_output_path,
                'pdf_vector': pdf_vector_output_path
            }
        except Exception as e:
            self.logger.error(f"Error creating contour cutline outputs: {str(e)}", exc_info=True)
            return None

    def _load_cache(self):
        """Load cache from a file."""
        cache_file = os.path.join(self.cache_dir, 'cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                self.logger.info(f"Loaded {len(self.cache)} items from cache file.")
            except (IOError, json.JSONDecodeError) as e:
                self.logger.warning(f"Could not load cache file: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        """Save the current cache state to a file."""
        cache_file = os.path.join(self.cache_dir, 'cache.json')
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except IOError as e:
            self.logger.error(f"Could not save cache file: {e}")

    def _cleanup_old_cache(self, force: bool = False):
        """Clean up old entries from the cache."""
        now = time.time()
        if not force and (now - getattr(self, '_last_cleanup_time', 0)) < 3600: # run at most every hour
            return

        self.logger.info("Running cache cleanup...")
        cleaned_count = 0
        
        for key in list(self.cache.keys()):
            entry = self.cache.get(key)
            if entry and 'timestamp' in entry:
                age_hours = (now - entry['timestamp']) / 3600
                if age_hours > self.max_cache_age:
                    del self.cache[key]
                    cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Removed {cleaned_count} old entries from cache.")
            self._save_cache()
        
        self._last_cleanup_time = now
        
    def _update_status(self, process_id: Optional[str], status: str, progress: int, message: str):
        """Update processing status.
        
        Args:
            process_id: Unique ID for the process
            status: Current status (e.g., 'loading', 'processing', 'completed', 'error')
            progress: Progress percentage (0-100)
            message: Status message to display
        """
        if process_id is None:
            return
            
        with self.status_lock:
            self.processing_status[process_id] = {
                'status': status,
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }
            
            # Call callback if registered
            if process_id in self.progress_callbacks:
                try:
                    self.progress_callbacks[process_id](self.processing_status[process_id])
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {str(e)}")
                    
    def _start_cleanup_thread(self):
        """Start a background thread for periodic cleanup."""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_resources()
                    time.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {str(e)}")
                    time.sleep(60)  # Wait longer on error
        
        cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name='resource_cleanup'
        )
        cleanup_thread.start()

    def _cleanup_resources(self):
        """Periodically clean up resources like caches and old files."""
        self.logger.info("Running periodic resource cleanup...")
        # Clean up old file-based cache
        self._cleanup_old_cache()

        # Clean up old preview cache entries
        now = time.time()
        cleaned_previews = 0
        for key in list(self.preview_cache.keys()):
            timestamp, _ = self.preview_cache.get(key, (0, None))
            if (now - timestamp) > self.preview_cache_timeout:
                del self.preview_cache[key]
                cleaned_previews += 1
        if cleaned_previews > 0:
            self.logger.info(f"Cleaned {cleaned_previews} expired preview cache entries.")

        # Optional: check memory and trigger GC
        self.memory_manager.check_memory()
        gc.collect()

    def _find_content_bbox(self, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds the bounding box of the main content in an image using transparency or contrast.
        """
        if img.mode in ('RGBA', 'LA', 'P'):
            # Attempt to use alpha channel for accurate bounding box
            try:
                alpha = img.convert('RGBA').split()[-1]
                bbox = alpha.getbbox()
                if bbox:
                    self.logger.info(f"Found content bbox using alpha channel: {bbox}")
                    return bbox
            except Exception:
                pass  # Fallback to other methods if alpha channel fails

        # Fallback for opaque images (e.g., JPG) or images without clear alpha
        self.logger.info("Falling back to contrast-based content detection.")
        grayscale_img = img.convert('L')
        
        # Apply a gentle blur to reduce noise
        blurred_img = grayscale_img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Use Otsu's thresholding to automatically determine the best threshold
        try:
            threshold_value = threshold_otsu(np.array(blurred_img))
            # Invert the image if the content is darker than the background (typical for logos)
            if np.mean(np.array(blurred_img)) > 127:
                 binary_img = blurred_img.point(lambda p: 255 if p < threshold_value else 0, '1')
            else:
                 binary_img = blurred_img.point(lambda p: 255 if p > threshold_value else 0, '1')
            
            bbox = binary_img.getbbox()
            if bbox:
                 self.logger.info(f"Found content bbox using contrast method: {bbox}")
            return bbox
        except Exception as e:
            self.logger.error(f"Could not determine content bbox using contrast: {e}", exc_info=True)
            return None

    def _get_sd_pipeline(self):
        """Lazily loads the Stable Diffusion pipeline."""
        with self.sd_pipeline_lock:
            if self.sd_pipeline is None:
                try:
                    self.logger.info("Initializing Stable Diffusion pipeline for the first time...")
                    from diffusers import StableDiffusionInpaintingPipeline
                    model_path = "runwayml/stable-diffusion-inpainting"
                    self.sd_pipeline = StableDiffusionInpaintingPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self.logger.info("Stable Diffusion pipeline loaded successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to load Stable Diffusion pipeline: {e}", exc_info=True)
            return self.sd_pipeline

    def _center_crop_fallback(self, img: Image.Image, target_width: int, target_height: int, output_path: str) -> Optional[str]:
        """A robust fallback to center-crop an image."""
        self.logger.warning(f"Falling back to simple center crop for {output_path}")
        try:
            cropped_img = ImageOps.fit(img, (target_width, target_height), Image.LANCZOS)
            cropped_img.save(output_path)
            return output_path
        except Exception as e:
            self.logger.error(f"Center crop fallback failed: {e}", exc_info=True)
            return None

    def _repurpose_for_social(self, file_path: str, platform: str, options: Dict) -> Optional[str]:
        """
        Intelligently repurposes an image for a social media platform using a tiered approach.
        This version uses transparency/contrast-based cropping instead of SAM.
        """
        target_width, target_height = self.social_sizes[platform]
        focal_point = options.get('focal_point')

        try:
            with Image.open(file_path) as img:
                img = img.convert("RGBA")
                original_width, original_height = img.size
                original_aspect = original_width / original_height
                target_aspect = target_width / target_height
                aspect_diff = abs(original_aspect - target_aspect)

                output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{platform}.png"
                output_path = os.path.join(self.output_folder, output_filename)

                # Strategy 1: Generative Fill for significant aspect ratio changes
                if aspect_diff > 0.5 and (target_width > original_width or target_height > original_height):
                    self.logger.info(f"Using generative fill for {platform} due to large aspect ratio difference.")
                    filled_img = self._generative_fill_background(img, target_width, target_height)
                    if filled_img:
                        filled_img.save(output_path, 'PNG')
                        return output_path
                    else:
                        self.logger.warning("Generative fill failed. Falling back.")

                # Strategy 2: Smart Cropping based on Content Box
                try:
                    self.logger.info(f"Using smart cropping for {platform}.")
                    content_bbox = self._find_content_bbox(img)
                    
                    if content_bbox:
                        if focal_point:
                            center_x, center_y = focal_point
                        else:
                            # Center of the content's bounding box
                            center_x = content_bbox[0] + (content_bbox[2] - content_bbox[0]) / 2
                            center_y = content_bbox[1] + (content_bbox[3] - content_bbox[1]) / 2

                        crop_width, crop_height = original_width, original_height
                        if original_aspect > target_aspect: # original is wider
                            crop_width = int(original_height * target_aspect)
                        else: # original is taller
                            crop_height = int(original_width / target_aspect)

                        # Calculate crop box centered on the focal point
                        left = max(0, center_x - crop_width / 2)
                        top = max(0, center_y - crop_height / 2)
                        
                        # Ensure the crop box does not go out of bounds
                        if left + crop_width > original_width:
                            left = original_width - crop_width
                        if top + crop_height > original_height:
                            top = original_height - crop_height
                        
                        right = left + crop_width
                        bottom = top + crop_height

                        cropped_img = img.crop((left, top, right, bottom))
                        resized_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)
                        resized_img.save(output_path, 'PNG')
                        return output_path
                    else:
                        self.logger.warning("Could not find content box. Falling back to center crop.")
                except Exception as e:
                    self.logger.error(f"Smart cropping failed: {e}", exc_info=True)

                # Strategy 3: Fallback to simple center crop
                self.logger.info(f"Falling back to center crop for {platform}.")
                return self._center_crop_fallback(img, target_width, target_height, output_path)

        except Exception as e:
            self.logger.error(f"Failed to repurpose image for {platform}: {e}", exc_info=True)
            return None

    def generate_preview(self, image_path: str) -> Optional[str]:
        """
        Generates a smaller preview image for quick display.

        Args:
            image_path: The path to the full-size image.

        Returns:
            The path to the generated preview image, or None on failure.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Cannot generate preview. File not found: {image_path}")
            return None

        # Check cache first
        cache_key = hashlib.md5(image_path.encode()).hexdigest()
        cached_preview = self.preview_cache.get(cache_key)
        if cached_preview:
            timestamp, preview_path = cached_preview
            if (time.time() - timestamp) < self.preview_cache_timeout and os.path.exists(preview_path):
                self.logger.info(f"Returning cached preview for {image_path}")
                return preview_path

        try:
            with Image.open(image_path) as img:
                img.thumbnail(self.preview_size, Image.LANCZOS)
                
                original_filename = os.path.basename(image_path)
                preview_filename = f"preview_{cache_key}_{original_filename}"
                preview_path = os.path.join(self.temp_folder, preview_filename)
                
                img.save(preview_path, quality=self.preview_quality, optimize=True)
                
                self.preview_cache[cache_key] = (time.time(), preview_path)
                
                self.logger.info(f"Generated preview for {image_path} at {preview_path}")
                return preview_path
        except Exception as e:
            self.logger.error(f"Failed to generate preview for {image_path}: {e}", exc_info=True)
            return None

    def process_social_variation(self, file_path: str, platform: str, options: Dict = None) -> Dict[str, str]:
        """
        Process a logo for a specific social media platform with optimizations.
        """
        options = options or {}
        start_time = time.time()

        try:
            # The core repurposing logic is now more robust and doesn't use SAM
            output_path = self._repurpose_for_social(file_path, platform, options)

            if not output_path or not os.path.exists(output_path):
                raise ValueError("Social media variation processing failed to produce an output file.")

            # Generate additional formats for social media
            additional_outputs = {}
            
            # Generate favicon for web-related platforms
            if platform in ['facebook_profile', 'twitter_profile', 'instagram_profile', 'linkedin_profile']:
                try:
                    with Image.open(output_path) as img:
                        favicon_path = self._create_favicon(img)
                        if favicon_path:
                            additional_outputs['favicon'] = favicon_path
                except Exception as e:
                    self.logger.warning(f"Failed to generate favicon for {platform}: {e}")
            
            # Generate WebP version for web optimization
            try:
                with Image.open(output_path) as img:
                    webp_path = self._create_webp_version(img)
                    if webp_path:
                        additional_outputs['webp'] = webp_path
            except Exception as e:
                self.logger.warning(f"Failed to generate WebP for {platform}: {e}")

            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed social variation for {platform} in {processing_time:.2f}s")

            # Generate a preview for immediate feedback
            preview_path = self.generate_preview(output_path)

            return {
                'status': 'success',
                'original_path': file_path,
                'output_path': output_path,
                'preview_path': preview_path,
                'platform': platform,
                'processing_time': round(processing_time, 2),
                'additional_outputs': additional_outputs
            }
        except Exception as e:
            self.logger.error(f"Error processing social variation for {platform}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'platform': platform
            }

    def process_all_social_variations(self, file_path: str, options: Dict = None) -> Dict[str, Dict[str, str]]:
        """
        Generates all social media variations for a logo in parallel using a thread pool.

        Args:
            file_path: Path to the source logo image.
            options: A dictionary of processing options.

        Returns:
            A dictionary where keys are platform names and values are the results
            from process_social_variation.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found for social variations: {file_path}")
            return {}

        options = options or {}
        results = {}
        # Use the class's thread pool to manage concurrent executions
        with self.thread_pool as executor:
            future_to_platform = {executor.submit(self.process_social_variation, file_path, platform, options): platform for platform in self.social_sizes.keys()}
            
            for future in as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    result = future.result()
                    if result:
                        results[platform] = result
                except Exception as e:
                    self.logger.error(f"A social variation task for {platform} failed: {e}", exc_info=True)
                    results[platform] = {'status': 'error', 'message': str(e), 'platform': platform}
        
        return results

    def _generative_fill_background(self, img: Image.Image, target_width: int, target_height: int) -> Optional[Image.Image]:
        """
        Use Stable Diffusion to fill the background, optimized for speed.
        """
        pipeline = self._get_sd_pipeline()
        if not pipeline:
            return None

        try:
            new_img = Image.new("RGBA", (target_width, target_height), (255, 255, 255, 0))
            paste_x = (target_width - img.width) // 2
            paste_y = (target_height - img.height) // 2
            new_img.paste(img, (paste_x, paste_y), img)

            mask = Image.new("L", (target_width, target_height), 255)
            alpha_mask = Image.new("L", (target_width, target_height), 0)
            alpha_mask.paste(img.getchannel('A'), (paste_x, paste_y))
            mask = ImageChops.subtract(mask, alpha_mask)

            prompt = "professional background, clean, high quality, seamless"
            
            # Optimized for speed: fewer steps, no guidance scale for faster, more creative fill
            filled_img = pipeline(
                prompt=prompt,
                image=new_img.convert("RGB"),
                mask_image=mask,
                num_inference_steps=20,  # Fewer steps for faster generation
                guidance_scale=7.0
            ).images[0]

            final_image = Image.new("RGBA", (target_width, target_height))
            final_image.paste(filled_img, (0, 0))
            final_image.paste(img, (paste_x, paste_y), img)

            return final_image

        except Exception as e:
            self.logger.error(f"Error during generative fill: {e}", exc_info=True)
            return None

    def _create_comparison_preview(self, original_path: str, processed_path: str) -> Optional[str]:
        """Creates a side-by-side comparison image of the original and processed images."""
        try:
            original_img = Image.open(original_path).convert("RGBA")
            processed_img = Image.open(processed_path).convert("RGBA")

            # Resize both to a common height for comparison
            height = 400
            orig_width = int(original_img.width * (height / original_img.height))
            proc_width = int(processed_img.width * (height / processed_img.height))
            
            original_img = original_img.resize((orig_width, height), Image.LANCZOS)
            processed_img = processed_img.resize((proc_width, height), Image.LANCZOS)

            # Create a new canvas to hold both images with a small gap
            gap = 20
            preview_width = orig_width + proc_width + gap
            preview_img = Image.new("RGBA", (preview_width, height), (230, 230, 230, 255))

            preview_img.paste(original_img, (0, 0))
            preview_img.paste(processed_img, (orig_width + gap, 0))

            # Add labels
            draw = ImageDraw.Draw(preview_img)
            try:
                # Use a universal truetype font if available, otherwise default
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 15)
                except IOError:
                    try:
                        font = ImageFont.truetype("arial.ttf", 15)
                    except IOError:
                        font = ImageFont.load_default()
            except IOError:
                font = ImageFont.load_default()
            
            draw.text((5, 5), "Original", fill="black", font=font)
            draw.text((orig_width + gap + 5, 5), "Repurposed", fill="black", font=font)

            filename = os.path.basename(processed_path)
            preview_filename = f"{Path(filename).stem}_preview.png"
            preview_path = os.path.join(self.output_folder, preview_filename)
            preview_img.save(preview_path)
            self.logger.info(f"Comparison preview saved to {preview_path}")
            return preview_path

        except Exception as e:
            self.logger.error(f"Failed to create comparison preview: {e}")
            return None

    def _create_webp_version(self, img: Image.Image) -> Optional[str]:
        """Create a WebP version of the logo with optimized settings."""
        try:
            output_path = os.path.join(
                self.output_folder,
                f"logo_{int(time.time())}.webp"
            )
            # Use method=4 for a good balance of speed and quality.
            img.save(output_path, 'WEBP', quality=85, method=4, lossless=False)
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating WebP version: {str(e)}", exc_info=True)
            return None

    def _create_favicon(self, img: Image.Image) -> Optional[str]:
        """Create a favicon.png file from the logo."""
        try:
            output_path = os.path.join(
                self.output_folder,
                f"favicon_{int(time.time())}.png"  # Use PNG instead of ICO
            )

            # Create a 32x32 favicon (standard size)
            favicon_size = (32, 32)
            img_copy = img.copy()
            img_copy.thumbnail(favicon_size, Image.Resampling.LANCZOS)

            new_img = Image.new('RGBA', favicon_size, (0, 0, 0, 0))
            new_img.paste(
                img_copy,
                ((favicon_size[0] - img_copy.size[0]) // 2, (favicon_size[1] - img_copy.size[1]) // 2)
            )

            # Save as PNG instead of ICO to avoid format issues
            new_img.save(output_path, 'PNG', optimize=True)
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating favicon: {str(e)}", exc_info=True)
            return None
    
    def _create_email_header(self, img: Image.Image) -> Optional[str]:
        """Create an email header version of the logo (recommended size: 600x200)."""
        try:
            # Create output filename
            output_path = os.path.join(
                self.output_folder,
                f"email_header_{int(time.time())}.png"
            )
            
            # Create a new image with email header dimensions (600x200)
            email_header = Image.new('RGBA', (600, 200), (255, 255, 255, 255))
            
            # Calculate size to fit the logo within the header while maintaining aspect ratio
            img_copy = img.copy()
            img_ratio = img_copy.width / img_copy.height
            header_ratio = 600 / 200
            
            if img_ratio > header_ratio:
                # Image is wider than header, fit to width
                new_width = 500  # Leave some padding
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller than header, fit to height
                new_height = 150  # Leave some padding
                new_width = int(new_height * img_ratio)
            
            # Resize the image
            img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Calculate position to center the image
            x = (600 - new_width) // 2
            y = (200 - new_height) // 2
            
            # Paste the image onto the header
            email_header.paste(img_copy, (x, y), img_copy if img_copy.mode == 'RGBA' else None)
            
            # Save the result
            email_header.save(output_path, 'PNG', quality=95, optimize=True)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating email header: {str(e)}")
            return None
    
    def _create_color_separations(self, file_path: str) -> dict:
        """
        Create high-quality color separations as individual PNGs (with registration marks) and a layered PSD.
        If the logo can be represented with 4 or fewer distinct PMS colors (after merging similar clusters), generate those PMS separations. Otherwise, generate only CMYK separations.
        Returns a dict with keys: 'pngs' (list of (path, label)), 'psd' (PSD file path), 'labels' (list of labels), 'process' (PMS or CMYK).
        """
        from PIL import Image, ImageDraw
        import numpy as np
        import os
        # Minimal PMS dataset (expand as needed)
        PMS_COLORS = [
            {"name": "White", "rgb": (255, 255, 255)},
            {"name": "Black", "rgb": (0, 0, 0)},
            {"name": "PMS 186 C", "rgb": (200, 16, 46)},
            {"name": "PMS 300 C", "rgb": (0, 114, 206)},
            {"name": "PMS 355 C", "rgb": (0, 175, 72)},
            {"name": "PMS 123 C", "rgb": (255, 199, 44)},
            {"name": "PMS 165 C", "rgb": (255, 80, 0)},
            {"name": "PMS 2587 C", "rgb": (102, 45, 145)},
            {"name": "PMS Black C", "rgb": (45, 41, 38)},
            {"name": "PMS Cool Gray 7 C", "rgb": (151, 153, 155)},
            {"name": "PMS 485 C", "rgb": (220, 20, 60)},
            {"name": "PMS 286 C", "rgb": (0, 71, 171)},
            {"name": "PMS 347 C", "rgb": (0, 150, 57)},
            {"name": "PMS 116 C", "rgb": (255, 205, 0)},
            {"name": "PMS 151 C", "rgb": (255, 102, 0)},
            {"name": "PMS 2592 C", "rgb": (142, 68, 173)},
            {"name": "PMS 7545 C", "rgb": (128, 130, 133)},
            {"name": "PMS 7546 C", "rgb": (98, 100, 102)},
            {"name": "PMS 7547 C", "rgb": (68, 70, 72)},
            {"name": "PMS 7548 C", "rgb": (45, 47, 49)},
            {"name": "PMS 7549 C", "rgb": (25, 27, 29)},
            {"name": "PMS 7550 C", "rgb": (15, 17, 19)},
        ]
        CMYK_LABELS = ["Cyan", "Magenta", "Yellow", "Black"]
        from colormath.color_objects import sRGBColor, LabColor
        from colormath.color_conversions import convert_color
        from colormath.color_diff import delta_e_cie2000
        try:
            img = Image.open(file_path).convert('RGBA')
            arr = np.array(img)
            
            # --- Enhanced Background Detection for Color Separations ---
            alpha_channel = arr[:, :, 3]
            
            # Create a mask for logo content (similar to vector trace)
            if np.max(alpha_channel) == 0 or np.min(alpha_channel) == 255:
                # No alpha or fully opaque: need to detect background
                self.logger.info("Color separations: No valid alpha channel, detecting background...")
                
                # Convert to grayscale for better analysis
                gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
                
                # Use Otsu's method to find optimal threshold
                otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                self.logger.info(f"Color separations: Otsu threshold: {otsu_threshold}")
                
                # Create initial mask using Otsu threshold
                content_mask = (gray > otsu_threshold).astype(np.uint8) * 255
                
                # If mask is mostly empty or mostly full, try alternative methods
                mask_ratio = np.sum(content_mask > 0) / content_mask.size
                self.logger.info(f"Color separations: Initial mask ratio: {mask_ratio:.3f}")
                
                if mask_ratio < 0.01 or mask_ratio > 0.99:
                    # Try adaptive thresholding
                    self.logger.info("Color separations: Trying adaptive thresholding...")
                    adaptive_mask = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    adaptive_ratio = np.sum(adaptive_mask > 0) / adaptive_mask.size
                    self.logger.info(f"Color separations: Adaptive mask ratio: {adaptive_ratio:.3f}")
                    
                    if 0.01 < adaptive_ratio < 0.99:
                        content_mask = adaptive_mask
                    else:
                        # Try edge detection approach
                        self.logger.info("Color separations: Trying edge detection approach...")
                        edges = cv2.Canny(gray, 50, 150)
                        kernel = np.ones((3,3), np.uint8)
                        edges = cv2.dilate(edges, kernel, iterations=1)
                        edges = cv2.erode(edges, kernel, iterations=1)
                        
                        # Fill contours
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        edge_mask = np.zeros_like(gray)
                        for contour in contours:
                            if cv2.contourArea(contour) > 100:  # Filter small contours
                                cv2.fillPoly(edge_mask, [contour], 255)
                        
                        edge_ratio = np.sum(edge_mask > 0) / edge_mask.size
                        self.logger.info(f"Color separations: Edge mask ratio: {edge_ratio:.3f}")
                        
                        if 0.01 < edge_ratio < 0.99:
                            content_mask = edge_mask
                        else:
                            # Last resort: try rembg
                            try:
                                from rembg import remove
                                self.logger.info("Color separations: Using rembg as last resort...")
                                img_no_bg = remove(img)
                                arr_no_bg = np.array(img_no_bg)
                                content_mask = (arr_no_bg[:, :, 3] > 0).astype(np.uint8) * 255
                            except ImportError:
                                # Final fallback: use contrast-based threshold
                                self.logger.info("Color separations: Using contrast-based threshold...")
                                # Find the most common color (likely background)
                                flat_colors = arr[:,:,:3].reshape(-1, 3)
                                from collections import Counter
                                most_common_color = Counter([tuple(x) for x in flat_colors]).most_common(1)[0][0]
                                
                                # Create mask for non-background pixels with higher tolerance
                                tolerance = 30  # Increased tolerance for dark backgrounds
                                diff = np.abs(arr[:,:,:3] - np.array(most_common_color)).sum(axis=2)
                                content_mask = (diff > tolerance).astype(np.uint8) * 255
            else:
                # Use alpha channel for mask
                content_mask = (alpha_channel > 0).astype(np.uint8) * 255
            
            # Clean up mask with morphological operations
            kernel = np.ones((3,3), np.uint8)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
            
            # Get only pixels from logo content (not background)
            content_pixels = arr[content_mask > 0][:, :3].reshape(-1, 3)
            
            # Skip processing if no valid pixels found
            if len(content_pixels) == 0:
                self.logger.warning("No valid pixels found for color separation after background removal")
                return None
                
            from sklearn.cluster import KMeans
            n_colors = min(6, len(content_pixels))  # Don't try to cluster more colors than pixels
            if n_colors < 2:
                n_colors = 2  # Minimum 2 colors for separation
                
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(content_pixels)
            cluster_centers = kmeans.cluster_centers_.astype(int)
            
            # Create labels for the entire image, but only where content exists
            labels = np.full(arr.shape[:2], -1)  # Initialize with -1 for background
            labels[content_mask > 0] = kmeans.predict(content_pixels)
            
            # --- Merge similar clusters (ΔE < 10) ---
            merged_clusters = []
            merged_indices = []
            for i, color1 in enumerate(cluster_centers):
                lab1 = convert_color(sRGBColor(*color1, is_upscaled=True), LabColor)
                found = False
                for j, (rep_color, group) in enumerate(merged_clusters):
                    lab2 = convert_color(sRGBColor(*rep_color, is_upscaled=True), LabColor)
                    # Fix numpy compatibility issue
                    try:
                        delta_e = delta_e_cie2000(lab1, lab2)
                        # Handle both scalar and array results
                        if hasattr(delta_e, 'item'):
                            delta_e_val = delta_e.item()
                        else:
                            delta_e_val = float(delta_e)
                    except Exception as e:
                        self.logger.warning(f"Error calculating color difference: {e}")
                        delta_e_val = 20  # Default to not merging
                        
                    if delta_e_val < 10:
                        merged_clusters[j][1].append(i)
                        found = True
                        break
                if not found:
                    merged_clusters.append([color1, [i]])
                    
            # --- PMS Matching for merged clusters ---
            pms_labels = []
            pms_indices = []
            for color, group in merged_clusters:
                # Robust white/black detection
                if np.allclose(color, [255, 255, 255], atol=12):
                    pms_labels.append('White')
                    pms_indices.append(group)
                    continue
                if np.allclose(color, [0, 0, 0], atol=12):
                    pms_labels.append('Black')
                    pms_indices.append(group)
                    continue
                lab1 = convert_color(sRGBColor(*color, is_upscaled=True), LabColor)
                min_dist = float('inf')
                best = None
                for pms in PMS_COLORS:
                    lab2 = convert_color(sRGBColor(*pms['rgb'], is_upscaled=True), LabColor)
                    try:
                        dist = delta_e_cie2000(lab1, lab2)
                        if hasattr(dist, 'item'):
                            dist_val = dist.item()
                        else:
                            dist_val = float(dist)
                    except Exception as e:
                        self.logger.warning(f"Error calculating PMS distance: {e}")
                        dist_val = float('inf')
                    if dist_val < min_dist:
                        min_dist = dist_val
                        best = pms['name']
                pms_labels.append(best or 'Unknown PMS')
                pms_indices.append(group)
                
            unique_pms = list(dict.fromkeys(pms_labels))
            # --- Decide PMS or CMYK ---
            use_cmyk = len(unique_pms) > 4
            sep_pngs = []
            sep_labels = []
            
            if not use_cmyk:
                # --- Generate PMS separations ---
                for idx, (pms_label, group) in enumerate(zip(pms_labels, pms_indices)):
                    mask = np.isin(labels, group).astype(np.uint8) * 255
                    color_layer = np.zeros_like(arr)
                    color_layer[:, :, :3] = cluster_centers[group[0]].astype(np.uint8)
                    color_layer[:, :, 3] = mask
                    sep_img = Image.fromarray(color_layer, 'RGBA')
                    # Draw registration marks
                    draw = ImageDraw.Draw(sep_img)
                    w, h = sep_img.size
                    reg_size = max(20, int(min(w, h) * 0.03))
                    reg_color = (0, 0, 0, 255)
                    draw.line([(5, 5), (5 + reg_size, 5)], fill=reg_color, width=2)
                    draw.line([(5, 5), (5, 5 + reg_size)], fill=reg_color, width=2)
                    draw.line([(w - 5, 5), (w - 5 - reg_size, 5)], fill=reg_color, width=2)
                    draw.line([(w - 5, 5), (w - 5, 5 + reg_size)], fill=reg_color, width=2)
                    draw.line([(5, h - 5), (5 + reg_size, h - 5)], fill=reg_color, width=2)
                    draw.line([(5, h - 5), (5, h - 5 - reg_size)], fill=reg_color, width=2)
                    draw.line([(w - 5, h - 5), (w - 5 - reg_size, h - 5)], fill=reg_color, width=2)
                    draw.line([(w - 5, h - 5), (w - 5, h - 5 - reg_size)], fill=reg_color, width=2)
                    safe_label = pms_label.replace(' ', '').replace('/', '').replace('(', '').replace(')', '')
                    png_path = os.path.join(self.output_folder, f"color_sep_{safe_label}_{os.path.basename(file_path)}")
                    sep_img.save(png_path, format="PNG", optimize=True)
                    sep_pngs.append((png_path, pms_label))
                    sep_labels.append(pms_label)
                process_type = 'PMS'
            else:
                # --- Generate CMYK separations ---
                def rgb_to_cmyk(r, g, b):
                    """Convert RGB to CMYK using improved color space conversion"""
                    if (r, g, b) == (0, 0, 0):
                        return 0, 0, 0, 100
                    
                    # Normalize RGB values
                    r_norm = r / 255.0
                    g_norm = g / 255.0
                    b_norm = b / 255.0
                    
                    # Calculate CMY
                    c = 1.0 - r_norm
                    m = 1.0 - g_norm
                    y = 1.0 - b_norm
                    
                    # Calculate K (black)
                    k = min(c, m, y)
                    
                    # Calculate final CMYK values
                    if k < 1.0:
                        c_final = (c - k) / (1.0 - k)
                        m_final = (m - k) / (1.0 - k)
                        y_final = (y - k) / (1.0 - k)
                    else:
                        c_final = 0.0
                        m_final = 0.0
                        y_final = 0.0
                    
                    # Convert to percentages and ensure values are in valid range
                    c_pct = max(0, min(100, int(c_final * 100)))
                    m_pct = max(0, min(100, int(m_final * 100)))
                    y_pct = max(0, min(100, int(y_final * 100)))
                    k_pct = max(0, min(100, int(k * 100)))
                    
                    return c_pct, m_pct, y_pct, k_pct
                    
                cmyk_layers = [np.zeros(arr.shape[:2], dtype=np.uint8) for _ in range(4)]
                for i, color in enumerate(cluster_centers):
                    c, m, y, k = rgb_to_cmyk(*color)
                    mask = (labels == i).astype(np.uint8) * 255
                    cmyk_vals = [c, m, y, k]
                    for j in range(4):
                        cmyk_layers[j] = np.maximum(cmyk_layers[j], (mask if cmyk_vals[j] > 0 else 0))
                        
                for j, cmyk_label in enumerate(CMYK_LABELS):
                    color_layer = np.zeros_like(arr)
                    color_layer[:, :, :3] = 255
                    color_layer[:, :, 3] = cmyk_layers[j]
                    sep_img = Image.fromarray(color_layer, 'RGBA')
                    # Draw registration marks
                    draw = ImageDraw.Draw(sep_img)
                    w, h = sep_img.size
                    reg_size = max(20, int(min(w, h) * 0.03))
                    reg_color = (0, 0, 0, 255)
                    draw.line([(5, 5), (5 + reg_size, 5)], fill=reg_color, width=2)
                    draw.line([(5, 5), (5, 5 + reg_size)], fill=reg_color, width=2)
                    draw.line([(w - 5, 5), (w - 5 - reg_size, 5)], fill=reg_color, width=2)
                    draw.line([(w - 5, 5), (w - 5, 5 + reg_size)], fill=reg_color, width=2)
                    draw.line([(5, h - 5), (5 + reg_size, h - 5)], fill=reg_color, width=2)
                    draw.line([(5, h - 5), (5, h - 5 - reg_size)], fill=reg_color, width=2)
                    draw.line([(w - 5, h - 5), (w - 5 - reg_size, h - 5)], fill=reg_color, width=2)
                    draw.line([(w - 5, h - 5), (w - 5, h - 5 - reg_size)], fill=reg_color, width=2)
                    label = f"{cmyk_label}"
                    safe_label = label.replace(' ', '').replace('/', '').replace('(', '').replace(')', '')
                    png_path = os.path.join(self.output_folder, f"color_sep_{safe_label}_{os.path.basename(file_path)}")
                    sep_img.save(png_path, format="PNG", optimize=True)
                    sep_pngs.append((png_path, label))
                    sep_labels.append(label)
                process_type = 'CMYK'
                
            # PSD export (if psd-tools is available)
            try:
                from psd_tools import PSDImage, Group, Layer
                layers = [Layer.from_image(Image.open(p[0])) for p in sep_pngs]
                psd = PSDImage.new(mode='RGBA', size=img.size, layers=layers)
                psd_path = os.path.join(self.output_folder, f"color_separations_{os.path.basename(file_path)}.psd")
                psd.save(psd_path)
            except ImportError:
                psd_path = None
                
            return {'pngs': sep_pngs, 'psd': psd_path, 'labels': sep_labels, 'process': process_type}
        except Exception as e:
            self.logger.error(f"Error in color separations: {str(e)}", exc_info=True)
            return None

    def _create_distressed_version(self, file_path: str) -> str:
        """
        Create a high-quality grunge distressed effect version of the logo.
        Only distress the non-transparent (logo) area; background remains transparent and RGB is not altered.
        Returns the output file path.
        """
        from PIL import Image, ImageFilter, ImageChops, ImageEnhance
        import numpy as np
        import os
        try:
            img = Image.open(file_path).convert("RGBA")
            w, h = img.size
            # Generate procedural noise mask for grunge
            noise = np.random.normal(loc=128, scale=60, size=(h, w)).astype(np.uint8)
            noise_img = Image.fromarray(noise, mode="L").filter(ImageFilter.GaussianBlur(radius=2))
            # Create a thresholded mask for random speckle
            threshold = 140
            grunge_mask = noise_img.point(lambda p: 255 if p > threshold else 0, mode='1').convert("L")
            # Optionally, blend with a subtle edge erosion for more realism
            edge_mask = img.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 30 else 0, mode='1').convert("L")
            combined_mask = ImageChops.lighter(grunge_mask, edge_mask)
            # Apply the mask ONLY to the alpha channel where the logo is opaque
            r, g, b, a = img.split()
            # Only distress where original alpha > 0
            distressed_alpha = ImageChops.multiply(a, combined_mask)
            # Where original alpha == 0, keep fully transparent
            distressed_alpha_np = np.array(distressed_alpha)
            a_np = np.array(a)
            distressed_alpha_np[a_np == 0] = 0
            distressed_alpha = Image.fromarray(distressed_alpha_np, mode="L")
            distressed_img = Image.merge("RGBA", (r, g, b, distressed_alpha))
            # Optionally, enhance contrast for a bolder effect
            distressed_img = ImageEnhance.Contrast(distressed_img).enhance(1.2)
            output_path = os.path.join(self.output_folder, f"distressed_{os.path.basename(file_path)}")
            distressed_img.save(output_path, format="PNG", optimize=True)
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating distressed effect: {str(e)}", exc_info=True)
            return None

    def generate_full_color_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        Convert a raster image to a full-color, layered, and editable vector graphic.
        This creates separate vector paths for each color region, preserving the original color palette.
        Returns SVG, PDF, and AI files with full color support.
        """
        import tempfile
        import subprocess
        import shutil
        self.logger.info(f"Starting full-color vector trace for {file_path}")
        
        try:
            # --- 1. Set Parameters --- 
            opts = options.get('full_color_vector_trace_options', {})
            num_colors = opts.get('num_colors', 8)  # More colors for full-color vectorization
            min_shape_area = opts.get('min_shape_area', 50)
            path_smoothness = opts.get('path_smoothness', 1.0)
            use_potrace = opts.get('use_potrace', True)  # Whether to use Potrace for each color region

            output_filename_base = f"{Path(file_path).stem}_fullcolor_vectorized"
            output_svg_path = os.path.join(self.output_folder, f"{output_filename_base}.svg")

            # --- 2. Load and preprocess image ---
            img = Image.open(file_path).convert('RGBA')
            img_w, img_h = img.size
            
            # Ensure we have content to vectorize
            if img_w < 10 or img_h < 10:
                raise ValueError("Image too small for vectorization")
            
            # Convert to numpy array
            arr = np.array(img)
            
            # --- Enhanced Background Detection for Full Color Vector Trace ---
            alpha_channel = arr[:, :, 3]
            
            # Create a mask for logo content (similar to vector trace)
            if np.max(alpha_channel) == 0 or np.min(alpha_channel) == 255:
                # No alpha or fully opaque: need to detect background
                self.logger.info("Full color vector trace: No valid alpha channel, detecting background...")
                
                # Convert to grayscale for better analysis
                gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
                
                # Use Otsu's method to find optimal threshold
                otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                self.logger.info(f"Full color vector trace: Otsu threshold: {otsu_threshold}")
                
                # Create initial mask using Otsu threshold
                content_mask = (gray > otsu_threshold).astype(np.uint8) * 255
                
                # If mask is mostly empty or mostly full, try alternative methods
                mask_ratio = np.sum(content_mask > 0) / content_mask.size
                self.logger.info(f"Full color vector trace: Initial mask ratio: {mask_ratio:.3f}")
                
                if mask_ratio < 0.01 or mask_ratio > 0.99:
                    # Try adaptive thresholding
                    self.logger.info("Full color vector trace: Trying adaptive thresholding...")
                    adaptive_mask = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    adaptive_ratio = np.sum(adaptive_mask > 0) / adaptive_mask.size
                    self.logger.info(f"Full color vector trace: Adaptive mask ratio: {adaptive_ratio:.3f}")
                    
                    if 0.01 < adaptive_ratio < 0.99:
                        content_mask = adaptive_mask
                    else:
                        # Try edge detection approach
                        self.logger.info("Full color vector trace: Trying edge detection approach...")
                        edges = cv2.Canny(gray, 50, 150)
                        kernel = np.ones((3,3), np.uint8)
                        edges = cv2.dilate(edges, kernel, iterations=1)
                        edges = cv2.erode(edges, kernel, iterations=1)
                        
                        # Fill contours
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        edge_mask = np.zeros_like(gray)
                        for contour in contours:
                            if cv2.contourArea(contour) > 100:  # Filter small contours
                                cv2.fillPoly(edge_mask, [contour], 255)
                        
                        edge_ratio = np.sum(edge_mask > 0) / edge_mask.size
                        self.logger.info(f"Full color vector trace: Edge mask ratio: {edge_ratio:.3f}")
                        
                        if 0.01 < edge_ratio < 0.99:
                            content_mask = edge_mask
                        else:
                            # Last resort: try rembg
                            try:
                                from rembg import remove
                                self.logger.info("Full color vector trace: Using rembg as last resort...")
                                img_no_bg = remove(img)
                                arr_no_bg = np.array(img_no_bg)
                                content_mask = (arr_no_bg[:, :, 3] > 0).astype(np.uint8) * 255
                            except ImportError:
                                # Final fallback: use contrast-based threshold
                                self.logger.info("Full color vector trace: Using contrast-based threshold...")
                                # Find the most common color (likely background)
                                flat_colors = arr[:,:,:3].reshape(-1, 3)
                                from collections import Counter
                                most_common_color = Counter([tuple(x) for x in flat_colors]).most_common(1)[0][0]
                                
                                # Create mask for non-background pixels with higher tolerance
                                tolerance = 30  # Increased tolerance for dark backgrounds
                                diff = np.abs(arr[:,:,:3] - np.array(most_common_color)).sum(axis=2)
                                content_mask = (diff > tolerance).astype(np.uint8) * 255
            else:
                # Use alpha channel for mask
                content_mask = (alpha_channel > 0).astype(np.uint8) * 255
            
            # Clean up mask with morphological operations
            kernel = np.ones((3,3), np.uint8)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
            
            # Check if we have any content to vectorize
            if np.sum(content_mask) == 0:
                raise ValueError("No visible content found in image after background removal")
            
            # --- 3. Color Quantization ---
            # Get only pixels from logo content (not background)
            valid_pixels = arr[content_mask > 0][:, :3]
            if len(valid_pixels) == 0:
                raise ValueError("No valid pixels found for color quantization")
            
            # Limit number of colors to number of unique pixels
            num_colors = min(num_colors, len(np.unique(valid_pixels, axis=0)))
            if num_colors < 2:
                num_colors = 2
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10).fit(valid_pixels)
            cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
            
            # Create labels for the entire image, but only where content exists
            labels = np.full(arr.shape[:2], -1)  # Initialize with -1 for background
            labels[content_mask > 0] = kmeans.predict(valid_pixels)
            
            # --- 4. Create vector paths for each color ---
            svg_paths = []
            temp_dir = None
            
            for i, color in enumerate(cluster_centers):
                # Create mask for this color
                mask = (labels == i).astype(np.uint8) * 255
                
                # Clean up mask
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Skip if mask is too small
                if np.sum(mask) < min_shape_area:
                    continue
        
                # Vectorize this color region
                color_paths = self._vectorize_color_region(mask, img_w, img_h, color, use_potrace, temp_dir)
                if color_paths:
                    svg_paths.extend(color_paths)
            
            if not svg_paths:
                raise ValueError("No vector paths generated from any color region")
            
            # --- 5. Assemble final SVG ---
            svg_content = f'''<svg width="{img_w}" height="{img_h}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      path {{ stroke: none; }}
    </style>
  </defs>
'''
            # Sort paths by area (largest first) for better layering
            svg_paths.sort(key=lambda x: x.get('area', 0), reverse=True)
            
            for path_info in svg_paths:
                svg_content += f'  <path d="{path_info["path_data"]}" fill="{path_info["color"]}" />\n'
            
            svg_content += '</svg>'
            
            # --- 6. Save SVG file ---
            with open(output_svg_path, 'w') as f:
                f.write(svg_content)
            
            # --- 7. Convert to PDF/AI ---
            pdf_output_path = os.path.join(self.output_folder, f"{output_filename_base}.pdf")
            ai_output_path = os.path.join(self.output_folder, f"{output_filename_base}.ai")
            
            try:
                cairosvg.svg2pdf(bytestring=svg_content.encode('utf-8'), write_to=pdf_output_path)
                shutil.copy(pdf_output_path, ai_output_path)
            except Exception as e:
                self.logger.warning(f"PDF conversion failed: {e}")
                # Create a simple PDF as fallback
                self._create_simple_pdf(output_svg_path, pdf_output_path)
                shutil.copy(pdf_output_path, ai_output_path)
            
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            self.logger.info(f"Full-color vector trace successful. Outputs saved to {self.output_folder}")
            return {
                'status': 'success',
                'output_paths': {
                    'svg': output_svg_path,
                    'pdf': pdf_output_path,
                    'ai': ai_output_path
                },
                'colors_used': len(cluster_centers)
            }
            
        except Exception as e:
            self.logger.error(f"Error during full-color vector tracing for {file_path}: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _vectorize_with_potrace(self, mask: np.ndarray, width: int, height: int, temp_dir: str) -> List[str]:
        """
        Vectorize using modern OpenCV-based approach (replaces outdated potrace).
        """
        return self._vectorize_with_modern_approach(mask, width, height, temp_dir)

    def _vectorize_color_region(self, mask: np.ndarray, width: int, height: int, color: np.ndarray, use_potrace: bool, temp_dir: str) -> List[Dict]:
        """Vectorize a single color region and return path information"""
        paths = []
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        try:
            # Use modern vectorization approach
            modern_paths = self._vectorize_with_modern_approach(mask, width, height, temp_dir)
            if modern_paths:
                for path_data in modern_paths:
                    # Calculate area for the path
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    area = sum(cv2.contourArea(contour) for contour in contours)
                    
                    paths.append({
                        'path_data': path_data,
                        'color': color_hex,
                        'area': area
                    })
                return paths
            
            # Fallback to OpenCV contours if modern approach fails
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Skip very small contours
                    continue
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                # Convert to SVG path
                points = approx.reshape(-1, 2)
                path_data = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    path_data += f" L {point[0]},{point[1]}"
                path_data += " Z"
                
                paths.append({
                    'path_data': path_data,
                    'color': color_hex,
                    'area': area
                })
            
            return paths
            
        except Exception as e:
            self.logger.warning(f"Error vectorizing color region: {e}")
            return paths

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.stats_lock:
            stats = self.processing_stats.copy()
            stats['memory_usage_mb'] = psutil.Process().memory_info().rss / (1024 * 1024)
            stats['cpu_count'] = multiprocessing.cpu_count()
            stats['active_threads'] = len(threading.enumerate())
            return stats

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current performance."""
        stats = self.get_performance_stats()
        recommendations = {
            'current_config': self.parallel_config.copy(),
            'recommendations': [],
            'performance_metrics': {
                'avg_processing_time': stats.get('avg_processing_time', 0),
                'success_rate': (stats.get('success', 0) / max(stats.get('total_processed', 1), 1)) * 100,
                'memory_efficiency': stats.get('memory_usage_mb', 0) / max(stats.get('total_processed', 1), 1),
                'worker_utilization': (stats.get('active_workers', 0) / stats.get('max_workers', 1)) * 100
            }
        }
        
        # Analyze performance and provide recommendations
        avg_time = stats.get('avg_processing_time', 0)
        memory_usage = stats.get('memory_usage_mb', 0)
        success_rate = recommendations['performance_metrics']['success_rate']
        
        if avg_time > 30:  # If average processing time is over 30 seconds
            recommendations['recommendations'].append({
                'type': 'performance',
                'priority': 'high',
                'message': 'Consider increasing CPU-intensive workers for vector processing',
                'action': 'Increase cpu_intensive_workers in parallel_config'
            })
        
        if memory_usage > 2048:  # If memory usage is over 2GB
            recommendations['recommendations'].append({
                'type': 'memory',
                'priority': 'high',
                'message': 'High memory usage detected. Consider reducing batch size or enabling memory monitoring',
                'action': 'Reduce batch_size or enable memory monitoring'
            })
        
        if success_rate < 95:  # If success rate is below 95%
            recommendations['recommendations'].append({
                'type': 'reliability',
                'priority': 'medium',
                'message': 'Success rate below 95%. Consider increasing task timeout or reducing worker count',
                'action': 'Increase task_timeout or reduce max_workers'
            })
        
        # Check if we can optimize worker allocation
        cpu_count = multiprocessing.cpu_count()
        if cpu_count > 8 and self.parallel_config['max_workers'] < cpu_count * 6:
            recommendations['recommendations'].append({
                'type': 'scaling',
                'priority': 'medium',
                'message': f'System has {cpu_count} CPUs but only using {self.parallel_config["max_workers"]} workers. Consider increasing max_workers.',
                'action': 'Increase max_workers to cpu_count * 6'
            })
        
        return recommendations

    def optimize_configuration(self, target_performance: str = 'balanced') -> Dict[str, Any]:
        """Optimize configuration based on target performance profile."""
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if target_performance == 'speed':
            # Optimize for maximum speed
            new_config = {
                'max_workers': min(256, cpu_count * 12),
                'cpu_intensive_workers': min(64, cpu_count * 4),
                'io_intensive_workers': min(128, cpu_count * 8),
                'memory_threshold_mb': int(memory_gb * 512),  # 50% of total RAM
                'task_timeout': 600,  # 10 minutes
                'batch_size': 16,  # Larger batches
                'enable_multiprocessing': True,
                'enable_priority_queue': True,
                'enable_memory_monitoring': False,  # Disable for speed
                'enable_adaptive_scaling': True,
            }
        elif target_performance == 'memory':
            # Optimize for memory efficiency
            new_config = {
                'max_workers': min(32, cpu_count * 2),
                'cpu_intensive_workers': min(8, cpu_count),
                'io_intensive_workers': min(16, cpu_count * 2),
                'memory_threshold_mb': int(memory_gb * 256),  # 25% of total RAM
                'task_timeout': 300,  # 5 minutes
                'batch_size': 4,  # Smaller batches
                'enable_multiprocessing': False,  # Disable for memory efficiency
                'enable_priority_queue': True,
                'enable_memory_monitoring': True,
                'enable_adaptive_scaling': True,
            }
        else:  # balanced
            # Balanced optimization
            new_config = {
                'max_workers': min(128, cpu_count * 8),
                'cpu_intensive_workers': min(32, cpu_count * 2),
                'io_intensive_workers': min(64, cpu_count * 4),
                'memory_threshold_mb': int(memory_gb * 384),  # 37.5% of total RAM
                'task_timeout': 300,  # 5 minutes
                'batch_size': 8,  # Medium batches
                'enable_multiprocessing': True,
                'enable_priority_queue': True,
                'enable_memory_monitoring': True,
                'enable_adaptive_scaling': True,
            }
        
        # Apply new configuration
        self.parallel_config.update(new_config)
        self.max_workers = new_config['max_workers']
        
        # Recreate thread pool with new settings
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='logo_processor'
        )
        
        self.logger.info(f"Configuration optimized for {target_performance} performance")
        return {
            'target_performance': target_performance,
            'new_config': new_config,
            'system_info': {
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'max_workers': self.max_workers
            }
        }

    def optimize_for_parallel_processing(self):
        """Optimize settings for parallel processing."""
        # Increase thread pool size for better parallelization
        cpu_count = multiprocessing.cpu_count()
        new_max_workers = min(128, cpu_count * 8)
        
        if new_max_workers != self.max_workers:
            self.logger.info(f"Optimizing thread pool: {self.max_workers} -> {new_max_workers} workers")
            self.max_workers = new_max_workers
            
            # Create new thread pool with optimized settings
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='logo_processor'
            )
            
            with self.stats_lock:
                self.processing_stats['max_workers'] = self.max_workers

    def _get_task_priority(self, task_name: str) -> int:
        """
        Get task priority for intelligent scheduling.
        Higher numbers = higher priority.
        """
        priority_map = {
            # High priority - fast tasks that can be done quickly
            'transparent_png': 10,
            'favicon': 9,
            'distressed_effect': 8,
            
            # Medium priority - moderate complexity
            'black_version': 7,
            'contour_cut': 6,
            
            # Lower priority - heavy computational tasks
            'vector_trace': 5,
            'full_color_vector_trace': 4,
            'color_separations': 3,
            
            # Social media variations
            'social_instagram': 6,
            'social_facebook': 6,
            'social_twitter': 6,
            'social_linkedin': 6,
            'social_youtube': 6,
            'social_tiktok': 6,
        }
        return priority_map.get(task_name, 5)

    def _get_optimal_processing_strategy(self, tasks: List[tuple], cpu_count: int, memory_gb: float) -> Dict[str, Any]:
        """
        Determine the optimal processing strategy based on tasks and system resources.
        """
        heavy_tasks = sum(1 for task in tasks if task[0] in ['vector_trace', 'full_color_vector_trace', 'color_separations'])
        light_tasks = len(tasks) - heavy_tasks
        total_tasks = len(tasks)
        
        # Strategy 1: Pure ThreadPoolExecutor (best for I/O bound tasks and when ProcessPool fails)
        if light_tasks > heavy_tasks or total_tasks <= 3:
            return {
                'strategy': 'thread_pool',
                'max_workers': min(cpu_count * 2, total_tasks * 2),
                'description': 'ThreadPoolExecutor for I/O bound tasks and small workloads'
            }
        
        # Strategy 2: Hybrid approach (best for mixed workloads)
        elif total_tasks > 3 and heavy_tasks > 0:
            return {
                'strategy': 'hybrid',
                'thread_workers': min(cpu_count, light_tasks * 2),
                'process_workers': min(cpu_count // 2, heavy_tasks),
                'description': 'Hybrid ThreadPool + optimized processing for mixed workload'
            }
        
        # Strategy 3: Optimized ThreadPool (fallback for all cases)
        else:
            return {
                'strategy': 'thread_pool_optimized',
                'max_workers': min(cpu_count * 3, total_tasks * 3),
                'description': 'Optimized ThreadPoolExecutor for maximum concurrency'
            }

    def _process_with_hybrid_strategy(self, tasks: List[tuple], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tasks using hybrid ThreadPool + optimized processing strategy.
        """
        # Separate tasks by type
        cpu_bound_tasks = [task for task in tasks if task[0] in ['vector_trace', 'full_color_vector_trace', 'color_separations']]
        io_bound_tasks = [task for task in tasks if task[0] not in ['vector_trace', 'full_color_vector_trace', 'color_separations']]
        
        results = {}
        
        # Process I/O-bound tasks with ThreadPoolExecutor (fast tasks first)
        if io_bound_tasks:
            with ThreadPoolExecutor(max_workers=strategy['thread_workers']) as thread_executor:
                thread_futures = {thread_executor.submit(task[1], *task[2:]): task[0] for task in io_bound_tasks}
                
                for future in as_completed(thread_futures):
                    task_name = thread_futures[future]
                    try:
                        result = future.result(timeout=30)
                        results[task_name] = result
                        self.logger.info(f"✅ ThreadPool completed {task_name}")
                    except Exception as e:
                        self.logger.error(f"ThreadPool task {task_name} failed: {e}")
                        results[task_name] = None
        
        # Process CPU-bound tasks with optimized ThreadPool (avoid ProcessPool pickling issues)
        if cpu_bound_tasks:
            # Use dedicated ThreadPool for CPU-bound tasks with fewer workers
            with ThreadPoolExecutor(max_workers=strategy['process_workers']) as cpu_executor:
                cpu_futures = {cpu_executor.submit(task[1], *task[2:]): task[0] for task in cpu_bound_tasks}
                
                for future in as_completed(cpu_futures):
                    task_name = cpu_futures[future]
                    try:
                        result = future.result(timeout=120)  # Longer timeout for CPU-bound tasks
                        results[task_name] = result
                        self.logger.info(f"✅ CPU ThreadPool completed {task_name}")
                    except Exception as e:
                        self.logger.error(f"CPU ThreadPool task {task_name} failed: {e}")
                        results[task_name] = None
        
        return results

    def _process_with_thread_pool_optimized(self, tasks: List[tuple], max_workers: int) -> Dict[str, Any]:
        """
        Process tasks using optimized ThreadPoolExecutor with intelligent scheduling.
        """
        results = {}
        
        # Sort tasks by priority for better scheduling
        tasks.sort(key=lambda x: self._get_task_priority(x[0]), reverse=True)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(task[1], *task[2:]): task[0] for task in tasks}
            
            completed_tasks = 0
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                completed_tasks += 1
                
                try:
                    # Use different timeouts based on task type
                    timeout = 120 if task_name in ['vector_trace', 'full_color_vector_trace', 'color_separations'] else 30
                    result = future.result(timeout=timeout)
                    results[task_name] = result
                    self.logger.info(f"✅ Optimized ThreadPool completed {task_name} ({completed_tasks}/{len(tasks)})")
                except Exception as e:
                    self.logger.error(f"Optimized ThreadPool task {task_name} failed: {e}")
                    results[task_name] = None
        
        return results

    def _process_with_process_pool(self, tasks: List[tuple], max_workers: int) -> Dict[str, Any]:
        """
        Process tasks using ProcessPoolExecutor for CPU-bound operations.
        """
        from concurrent.futures import ProcessPoolExecutor
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(task[1], *task[2:]): task[0] for task in tasks}
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result(timeout=60)
                    results[task_name] = result
                    self.logger.info(f"✅ ProcessPool completed {task_name}")
                except Exception as e:
                    self.logger.error(f"ProcessPool task {task_name} failed: {e}")
                    results[task_name] = None
        
        return results

    def _process_with_async_strategy(self, tasks: List[tuple]) -> Dict[str, Any]:
        """
        Process tasks using async/await for maximum concurrency.
        """
        import asyncio
        import concurrent.futures
        
        async def process_task_async(task):
            task_name, task_func, *task_args = task
            
            # Run CPU-bound task in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, task_func, *task_args)
            
            return task_name, result
        
        async def process_all_tasks_async():
            # Create tasks
            async_tasks = [process_task_async(task) for task in tasks]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Async task failed: {result}")
                else:
                    task_name, task_result = result
                    processed_results[task_name] = task_result
                    self.logger.info(f"✅ Async completed {task_name}")
            
            return processed_results
        
        # Run async processing
        try:
            results = asyncio.run(process_all_tasks_async())
            return results
        except Exception as e:
            self.logger.error(f"Async processing failed: {e}")
            return {}

    def _check_memory_usage(self) -> bool:
        """Check current memory usage and return if within limits."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_usage_mb = memory_info.rss / (1024 * 1024)
            
            with self.memory_lock:
                self.memory_monitor['current_usage'] = current_usage_mb
                self.memory_monitor['peak_usage'] = max(self.memory_monitor['peak_usage'], current_usage_mb)
                self.memory_monitor['last_check'] = time.time()
                
                # Check if we're approaching memory limits
                threshold = self.parallel_config['memory_threshold_mb']
                if current_usage_mb > threshold:
                    self.memory_monitor['threshold_exceeded'] = True
                    self.logger.warning(f"Memory usage high: {current_usage_mb:.1f}MB > {threshold}MB")
                    return False
                else:
                    self.memory_monitor['threshold_exceeded'] = False
                    return True
        except Exception as e:
            self.logger.warning(f"Error checking memory usage: {e}")
            return True  # Assume OK if we can't check

    def _get_optimal_worker_count(self, task_type: str) -> int:
        """Get optimal worker count based on task type and system resources."""
        cpu_count = multiprocessing.cpu_count()
        
        if task_type in ['vector_trace', 'full_color_vector_trace']:
            # CPU-intensive tasks
            return min(self.parallel_config['cpu_intensive_workers'], cpu_count * 2)
        else:
            # I/O-intensive tasks
            return min(self.parallel_config['io_intensive_workers'], cpu_count * 4)

    def _process_with_multiprocessing(self, func, *args, **kwargs):
        """Process CPU-intensive tasks using multiprocessing."""
        if not self.parallel_config['enable_multiprocessing']:
            return func(*args, **kwargs)
        
        try:
            from multiprocessing import Pool, cpu_count
            from functools import partial
            
            # Create a partial function with the arguments
            partial_func = partial(func, *args, **kwargs)
            
            # Use a small pool for CPU-intensive tasks
            pool_size = min(4, cpu_count())
            with Pool(processes=pool_size) as pool:
                result = pool.apply(partial_func)
            return result
        except Exception as e:
            self.logger.warning(f"Multiprocessing failed, falling back to threading: {e}")
            return func(*args, **kwargs)

    def _batch_process_tasks(self, tasks: list, batch_size: int = None) -> dict:
        """Process tasks in batches for better resource management."""
        if batch_size is None:
            batch_size = self.parallel_config['batch_size']
        
        all_results = {}
        total_tasks = len(tasks)
        
        for i in range(0, total_tasks, batch_size):
            batch = tasks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size} ({len(batch)} tasks)")
            
            # Process batch with optimal worker count
            batch_results = self._process_task_batch(batch)
            all_results.update(batch_results)
            
            # Check memory and take a small break if needed
            if not self._check_memory_usage():
                self.logger.info("Memory threshold exceeded, taking a brief pause...")
                time.sleep(0.5)
        
        return all_results

    def _process_task_batch(self, tasks: list) -> dict:
        """Process a batch of tasks with optimized worker allocation."""
        if not tasks:
            return {}
        
        # Determine optimal worker count for this batch
        task_types = [task[0] for task in tasks]
        cpu_intensive = any(t in ['vector_trace', 'full_color_vector_trace'] for t in task_types)
        optimal_workers = self._get_optimal_worker_count('vector_trace' if cpu_intensive else 'transparent_png')
        
        # Sort tasks by priority if enabled
        if self.parallel_config['enable_priority_queue']:
            tasks.sort(key=lambda x: self._get_task_priority(x[0]))
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=optimal_workers, thread_name_prefix='logo_processor_batch') as executor:
            # Submit all tasks in the batch
            future_to_task = {}
            for task in tasks:
                task_name = task[0]
                func = task[1]
                args = task[2:]
                
                # Use multiprocessing for CPU-intensive tasks
                if task_name in ['vector_trace', 'full_color_vector_trace'] and self.parallel_config['enable_multiprocessing']:
                    future = executor.submit(self._process_with_multiprocessing, func, *args)
                else:
                    future = executor.submit(func, *args)
                
                future_to_task[future] = task_name
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result(timeout=self.parallel_config['task_timeout'])
                    results[task_name] = result
                    
                    # Update performance stats
                    with self.stats_lock:
                        self.processing_stats['task_completion_times'][task_name] = time.time() - start_time
                        
                except Exception as e:
                    self.logger.error(f"Error processing {task_name}: {e}", exc_info=True)
                    results[task_name] = None
        
        return results

    def _create_simple_pdf(self, svg_path: str, pdf_path: str):
        """Create a simple PDF as fallback"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.drawString(100, 750, "Vector trace generated")
            c.drawString(100, 730, "Original SVG file included in package")
            c.save()
        except Exception as e:
            self.logger.warning(f"Simple PDF creation failed: {e}")
            # Create empty PDF file
            with open(pdf_path, 'w') as f:
                f.write("%PDF-1.4\n")

    def _cpu_fallback_processing(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """CPU fallback processing when GPU is not available."""
        try:
            # Create a standard LogoProcessor instance for CPU processing
            processor = LogoProcessor()
            return processor.process_logo(file_path, options)
        except Exception as e:
            self.logger.error(f"CPU fallback processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def _process_favicon_wrapper(self, file_path: str) -> str:
        """Wrapper for favicon processing."""
        try:
            return self._create_favicon(Image.open(file_path))
        except Exception as e:
            self.logger.error(f"Error in favicon wrapper: {e}", exc_info=True)
            return None

    def _process_vector_trace_wrapper(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Wrapper for vector trace processing."""
        try:
            self.logger.info(f"Vector trace wrapper called with file: {file_path}")
            self.logger.info(f"Vector trace wrapper options: {options}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Call the vector trace method
            result = self.generate_vector_trace(file_path, options)
            self.logger.info(f"Vector trace wrapper result: {result}")
            self.logger.info(f"Vector trace result type: {type(result)}")
            self.logger.info(f"Vector trace result status: {result.get('status') if isinstance(result, dict) else 'Not a dict'}")
            
            # Validate the result
            if not result:
                error_msg = "Vector trace returned None result"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            if not isinstance(result, dict):
                error_msg = f"Vector trace returned non-dict result: {type(result)}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            if result.get('status') != 'success':
                error_msg = f"Vector trace failed with status: {result.get('status')}, message: {result.get('message')}"
                self.logger.error(error_msg)
                return result
            
            # Check if output paths exist
            output_paths = result.get('output_paths', {})
            for path_type, path in output_paths.items():
                if path and not os.path.exists(path):
                    error_msg = f"Vector trace output file not found: {path_type} = {path}"
                    self.logger.error(error_msg)
                    return {'status': 'error', 'message': error_msg}
            
            self.logger.info(f"Vector trace wrapper returning successful result")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in vector trace wrapper: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _create_adobe_quality_backup_paths(self, contours: List[np.ndarray], color_info: Dict, 
                                         min_area: int, smoothness: float, use_bezier: bool) -> List[Dict]:
        """
        Create Adobe/Canva quality vector paths using comprehensive contour detection.
        This method provides professional-grade vectorization when Potrace fails.
        """
        paths = []
        
        # If no contours provided, perform comprehensive contour detection
        if not contours:
            self.logger.info(f"No contours provided for color {color_info['hex']}, performing comprehensive detection")
            contours = self._detect_comprehensive_contours(color_info)
        
        self.logger.info(f"Processing {len(contours)} contours for color {color_info['hex']}")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Advanced contour preprocessing for Adobe quality
            processed_contour = self._preprocess_contour_for_quality(contour, smoothness)
            
            if len(processed_contour) < 3:
                continue
            
            # Create high-quality path using multiple algorithms
            path_data = self._create_advanced_bezier_path(processed_contour, smoothness, use_bezier)
            
            if path_data:
                paths.append({
                    'path_data': path_data,
                    'color': color_info['hex'],
                    'area': area,
                    'is_dominant': color_info.get('is_dominant', False),
                    'quality': 'adobe_grade',
                    'contour_index': i
                })
                self.logger.debug(f"Created path {i+1}/{len(contours)} for color {color_info['hex']} with area {area:.1f}")
        
        self.logger.info(f"Generated {len(paths)} high-quality paths for color {color_info['hex']}")
        return paths

    def _detect_comprehensive_contours(self, color_info: Dict) -> List[np.ndarray]:
        """
        Comprehensive contour detection for complex logos with multiple shapes.
        """
        # Get the original image array
        if not hasattr(self, '_current_image_array'):
            self.logger.warning("No image array available for contour detection")
            return []
        
        arr = self._current_image_array
        rgb_arr = arr[:, :, :3]
        target_color = np.array(color_info['rgb'])
        
        # Create color mask with adaptive tolerance
        color_variance = color_info.get('color_variance', [30, 30, 30])
        if isinstance(color_variance, list):
            tolerance = max(15, min(100, int(np.mean(color_variance) * 2.5)))  # Increased tolerance range
        else:
            tolerance = 60
        
        # Calculate color difference with more flexible matching
        color_diff = np.abs(rgb_arr - target_color).sum(axis=2)
        color_mask = (color_diff <= tolerance).astype(np.uint8) * 255
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Find all contours (both external and internal)
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours with more lenient criteria
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:  # Reduced minimum area threshold
                # Check if contour is not too simple (avoid single pixels)
                if len(contour) > 2:  # Reduced minimum points
                    valid_contours.append(contour)
        
        # If no contours found, try with more aggressive tolerance
        if len(valid_contours) == 0:
            self.logger.debug(f"No contours found for color {color_info['hex']}, trying aggressive detection")
            tolerance = min(150, tolerance * 2)  # Double the tolerance
            color_mask = (color_diff <= tolerance).astype(np.uint8) * 255
            
            # Apply more aggressive cleaning
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 3:  # Even more lenient
                    if len(contour) > 2:
                        valid_contours.append(contour)
        
        self.logger.info(f"Detected {len(valid_contours)} valid contours for color {color_info['hex']} with tolerance {tolerance}")
        return valid_contours

    def _preprocess_contour_for_quality(self, contour: np.ndarray, smoothness: float) -> np.ndarray:
        """
        Advanced contour preprocessing for Adobe-quality vectorization.
        """
        if len(contour) < 3:
            return contour
        
        # Step 1: Convert to float for precision
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        # Step 2: Douglas-Peucker simplification with adaptive epsilon
        arc_length = cv2.arcLength(contour, True)
        epsilon = max(0.5, arc_length * 0.002 * smoothness)  # More conservative epsilon
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        # Step 3: Remove redundant points with adaptive threshold
        cleaned_points = self._remove_redundant_points(simplified.reshape(-1, 2), threshold=1.5)
        
        # Step 4: Smooth corners for better curves (only if enough points)
        if len(cleaned_points) > 4:
            smoothed_points = self._smooth_corners(cleaned_points, smoothness)
        else:
            smoothed_points = cleaned_points
        
        # Step 5: Ensure minimum number of points for Bezier curves
        if len(smoothed_points) < 3:
            # If we have too few points, add intermediate points
            if len(contour_points) >= 3:
                # Use original contour with minimal processing
                return contour
            else:
                return contour
        
        return smoothed_points.reshape(-1, 1, 2).astype(np.int32)

    def _remove_redundant_points(self, points: np.ndarray, threshold: float = 1.5) -> np.ndarray:
        """
        Remove redundant points that are too close together with improved logic.
        """
        if len(points) < 4:
            return points
        
        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        
        # Keep points that are far enough apart
        keep_indices = [0]  # Always keep first point
        for i, dist in enumerate(distances):
            if dist > threshold:
                keep_indices.append(i + 1)
        
        # Always keep the last point
        if len(points) - 1 not in keep_indices:
            keep_indices.append(len(points) - 1)
        
        # Ensure we have at least 3 points
        if len(keep_indices) < 3:
            # If we removed too many points, keep every other point
            if len(points) > 4:
                step = max(1, len(points) // 4)
                keep_indices = list(range(0, len(points), step))
                if len(points) - 1 not in keep_indices:
                    keep_indices.append(len(points) - 1)
            else:
                keep_indices = [0, len(points) // 2, len(points) - 1]
        
        return points[keep_indices]

    def _smooth_corners(self, points: np.ndarray, smoothness: float) -> np.ndarray:
        """
        Smooth sharp corners to create more natural curves with improved algorithm.
        """
        if len(points) < 5:
            return points
        
        smoothed = []
        n_points = len(points)
        
        for i in range(n_points):
            if i == 0:
                # First point
                p1 = points[-1]  # Wrap around
                p2 = points[i]
                p3 = points[i + 1]
            elif i == n_points - 1:
                # Last point
                p1 = points[i - 1]
                p2 = points[i]
                p3 = points[0]  # Wrap around
            else:
                p1 = points[i - 1]
                p2 = points[i]
                p3 = points[i + 1]
            
            # Calculate angle between segments
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            # Calculate angle
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
            angle = np.arccos(cos_angle)
            
            # If angle is sharp (less than 135 degrees), smooth it
            if angle < np.pi * 3/4:  # 135 degrees
                # Create smooth curve using quadratic interpolation
                t = smoothness * 0.3  # Reduced smoothing factor
                smoothed_point = p2 * (1 - t) + (p1 + p3) * 0.5 * t
                smoothed.append(smoothed_point.astype(np.int32))
            else:
                smoothed.append(p2)
        
        return np.array(smoothed)

    def _create_advanced_bezier_path(self, contour: np.ndarray, smoothness: float, use_bezier: bool) -> str:
        """
        Create advanced Bezier curve path with Adobe-quality algorithms.
        """
        points = contour.reshape(-1, 2)
        
        if len(points) < 3:
            return None
        
        if use_bezier:
            return self._create_adaptive_bezier_path(points, smoothness)
        else:
            return self._create_optimized_linear_path(points, smoothness)

    def _create_adaptive_bezier_path(self, points: np.ndarray, smoothness: float) -> str:
        """
        Create adaptive Bezier curves that adjust based on curvature.
        """
        if len(points) < 3:
            return None

        path_data = f"M {points[0][0]},{points[0][1]}"
        
        if len(points) == 3:
            # Simple triangle - use quadratic curve
            cp_x = points[1][0]
            cp_y = points[1][1]
            path_data += f" Q {cp_x},{cp_y} {points[2][0]},{points[2][1]}"
        else:
            # Complex shape - use cubic Bezier curves
            for i in range(1, len(points)):
                if i == 1:
                    # First curve
                    cp1_x = points[0][0] + (points[1][0] - points[0][0]) * smoothness
                    cp1_y = points[0][1] + (points[1][1] - points[0][1]) * smoothness
                    cp2_x = points[1][0] - (points[2][0] - points[0][0]) * smoothness * 0.3
                    cp2_y = points[1][1] - (points[2][1] - points[0][1]) * smoothness * 0.3
                elif i == len(points) - 1:
                    # Last curve
                    cp1_x = points[i-1][0] + (points[i][0] - points[i-2][0]) * smoothness * 0.3
                    cp1_y = points[i-1][1] + (points[i][1] - points[i-2][1]) * smoothness * 0.3
                    cp2_x = points[i][0] - (points[i][0] - points[i-1][0]) * smoothness
                    cp2_y = points[i][1] - (points[i][1] - points[i-1][1]) * smoothness
                else:
                    # Middle curves with adaptive control points
                    prev_point = points[i-2] if i > 1 else points[0]
                    curr_point = points[i-1]
                    next_point = points[i]
                    next_next_point = points[i+1] if i < len(points)-1 else points[i]
                    
                    # Calculate curvature
                    v1 = curr_point - prev_point
                    v2 = next_point - curr_point
                    v3 = next_next_point - next_point
                    
                    # Adaptive smoothness based on curvature
                    curvature = self._calculate_curvature(v1, v2, v3)
                    adaptive_smoothness = smoothness * (1 - curvature * 0.5)
                    
                    cp1_x = curr_point[0] + (next_point[0] - prev_point[0]) * adaptive_smoothness * 0.3
                    cp1_y = curr_point[1] + (next_point[1] - prev_point[1]) * adaptive_smoothness * 0.3
                    cp2_x = next_point[0] - (next_next_point[0] - curr_point[0]) * adaptive_smoothness * 0.3
                    cp2_y = next_point[1] - (next_next_point[1] - curr_point[1]) * adaptive_smoothness * 0.3
                
                path_data += f" C {cp1_x:.0f},{cp1_y:.0f} {cp2_x:.0f},{cp2_y:.0f} {points[i][0]},{points[i][1]}"
        
        # Close path
        path_data += " Z"
        return path_data

    def _calculate_curvature(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """
        Calculate curvature between three vectors.
        """
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        v3_norm = v3 / (np.linalg.norm(v3) + 1e-8)
        
        # Calculate angles
        cos_angle1 = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        cos_angle2 = np.clip(np.dot(v2_norm, v3_norm), -1, 1)
        
        angle1 = np.arccos(cos_angle1)
        angle2 = np.arccos(cos_angle2)
        
        # Curvature is the change in angle
        curvature = abs(angle2 - angle1) / np.pi
        return min(curvature, 1.0)

    def _create_optimized_linear_path(self, points: np.ndarray, smoothness: float) -> str:
        """
        Create optimized linear path with intelligent point reduction.
        """
        if len(points) < 3:
            return None
        
        # Start path
        path_data = f"M {points[0][0]},{points[0][1]}"
        
        # Adaptive point selection based on smoothness
        step = max(1, int(1 / smoothness)) if smoothness > 0 else 1
        
        for i in range(step, len(points), step):
            if i > 1:
                # Check if this point is significantly different from the previous
                prev_point = points[i-step]
                curr_point = points[i]
                distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
                
                # Skip points that are too close together
                if distance < 5:  # Minimum distance threshold
                    continue
            
            path_data += f" L {points[i][0]},{points[i][1]}"
        
        # Always include the last point
        if len(points) > 1:
            path_data += f" L {points[-1][0]},{points[-1][1]}"
        
        # Close path
        path_data += " Z"
        return path_data

    def _vectorize_with_modern_approach(self, mask: np.ndarray, width: int, height: int, temp_dir: str) -> List[str]:
        """
        Vectorize using modern OpenCV-based contour detection and SVG path generation.
        Enhanced for complex, jagged logos with sharp edges and intricate details.
        """
        try:
            # Find contours in the mask with more detailed detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            paths = []
            for contour in contours:
                # Filter out very small contours
                area = cv2.contourArea(contour)
                if area < 5:  # Reduced threshold for more detail
                    continue
                
                # Use more detailed contour approximation for complex shapes
                epsilon = 0.005 * cv2.arcLength(contour, True)  # Much smaller epsilon for detail
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                # Convert to points
                points = approx.reshape(-1, 2)
                
                # Create high-detail SVG path
                path_data = self._create_high_detail_svg_path(points)
                if path_data:
                    paths.append(path_data)
            
            return paths
            
        except Exception as e:
            self.logger.warning(f"Modern vectorization failed: {e}")
            return []
    
    def _create_high_detail_svg_path(self, points: np.ndarray) -> str:
        """
        Create a high-detail SVG path that preserves sharp edges and complex geometry.
        """
        if len(points) < 3:
            return ""
        
        try:
            # Start path
            path_data = f"M {points[0][0]},{points[0][1]}"
            
            # For complex, jagged logos, use line segments to preserve sharp edges
            for i in range(1, len(points)):
                path_data += f" L {points[i][0]},{points[i][1]}"
            
            # Close path
            path_data += " Z"
            return path_data
            
        except Exception as e:
            self.logger.warning(f"Error creating high-detail path: {e}")
            return self._create_simple_path(points)

    def _remove_duplicate_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove duplicate contours using advanced similarity metrics.
        This method is kept for compatibility but the advanced CV system handles this better.
        """
        if len(contours) <= 1:
            return contours
        
        unique_contours = []
        
        for contour in contours:
            is_duplicate = False
            
            for existing in unique_contours:
                # Basic similarity check
                similarity = cv2.matchShapes(contour, existing, cv2.CONTOURS_MATCH_I1, 0)
                area_ratio = cv2.contourArea(contour) / max(cv2.contourArea(existing), 1)
                
                if similarity < 0.01 and 0.8 < area_ratio < 1.2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
        
        return unique_contours

    def _process_internal_details(self, mask: np.ndarray, hierarchy: np.ndarray, contours: List[np.ndarray]) -> List[str]:
        """
        Process internal details using advanced CV approach.
        Simplified since the advanced CV system handles internal details better.
        """
        paths = []
        
        try:
            # Use advanced CV for internal detail detection
            if ADVANCED_CV_AVAILABLE:
                # Convert mask to float for advanced processing
                mask_float = img_as_float(mask)
                
                # Use watershed and region-based detection for internal details
                internal_contours = self._detect_watershed_contours(mask_float)
                internal_contours.extend(self._detect_region_contours(mask_float))
                
                # Process internal contours
                for contour in internal_contours:
                    area = cv2.contourArea(contour)
                    if area > 0.1:  # Small threshold for internal details
                        path_data = self._create_advanced_path(contour)
                        if path_data:
                            paths.append(path_data)
            else:
                # Fallback to basic hierarchy processing
                if hierarchy is not None:
                    for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                        if h[3] >= 0:  # Has parent (internal contour)
                            area = cv2.contourArea(contour)
                            if area > 0.1:
                                path_data = self._create_advanced_path(contour)
                                if path_data:
                                    paths.append(path_data)
        
        except Exception as e:
            self.logger.warning(f"Internal detail processing failed: {e}")
        
        return paths

    def generate_high_quality_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        High-quality single-color vector tracing using VTracer with advanced preprocessing.
        
        Args:
            file_path: Path to input image
            options: Processing options including:
                - simplify: Simplification factor (0.1-2.0)
                - turdsize: Minimum contour size
                - noise_reduction: Apply noise reduction
                - adaptive_threshold: Use adaptive thresholding
                - preview: Generate preview PNG
                - output_format: 'svg', 'pdf', or 'both'
        
        Returns:
            Dict with status, output paths, and processing details
        """
        start_time = time.time()
        
        try:
            # Extract options with defaults optimized for detail detection
            simplify = options.get('simplify', 0.6)  # Reduced from 0.8 for more details
            turdsize = options.get('turdsize', 1)    # Reduced from 2 for more details
            noise_reduction = options.get('noise_reduction', True)
            adaptive_threshold = options.get('adaptive_threshold', True)
            generate_preview = options.get('preview', True)
            output_format = options.get('output_format', 'both')  # Always generate all formats
            
            self.logger.info(f"Starting enhanced vector trace with VTracer")
            self.logger.info(f"Options: simplify={simplify}, turdsize={turdsize}, noise_reduction={noise_reduction}")
            self.logger.info(f"Detail preservation: Enhanced preprocessing and contour detection enabled")
            
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
                    'edge_enhancement': True
                }
            }
            
            self.logger.info(f"Enhanced vector trace completed in {processing_time:.2f}s")
            self.logger.info(f"Generated formats: {list(vector_results.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"High-quality vector trace failed: {e}", exc_info=True)
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
                
                # Combine the results (majority voting)
                binary = np.zeros_like(binary1)
                binary[(binary1 == 0) & (binary2 == 0)] = 0  # Black if both adaptive methods agree
                binary[(binary1 == 255) & (binary2 == 255)] = 255  # White if both adaptive methods agree
                binary[(binary1 == 0) & (binary3 == 0)] = 0  # Black if adaptive and Otsu agree
                binary[(binary1 == 255) & (binary3 == 255)] = 255  # White if adaptive and Otsu agree
                
                # For pixels where methods disagree, use the more conservative approach
                # (preserve more details by keeping them as black)
                disagreement_mask = (binary1 != binary2) & (binary1 != binary3) & (binary2 != binary3)
                binary[disagreement_mask] = 0  # Default to black (preserve details)
                
            else:
                # Use Otsu's method with preprocessing
                _, binary = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Post-processing to enhance details
            # Apply morphological operations to clean up while preserving details
            kernel_small = np.ones((1, 1), np.uint8)
            kernel_medium = np.ones((2, 2), np.uint8)
            
            # Remove isolated pixels (noise) while preserving details
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            
            # Close small gaps in letters and shapes
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
            
            # Step 5: Ensure proper orientation (VTracer expects black shapes on white background)
            # Count black and white pixels to determine if inversion is needed
            black_pixels = np.sum(binary == 0)
            white_pixels = np.sum(binary == 255)
            
            # If more than 60% is black, invert (assuming we want black shapes on white background)
            if black_pixels > white_pixels * 0.6:
                binary = cv2.bitwise_not(binary)
            
            # Step 6: Final detail enhancement
            # Apply one more pass of detail enhancement
            kernel_detail = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]], dtype=np.float32)
            enhanced = cv2.filter2D(binary, -1, kernel_detail)
            
            # Threshold the enhanced result
            _, final_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            
            # Save intermediate results for debugging
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(debug_dir, "original_gray.png"), gray)
            cv2.imwrite(os.path.join(debug_dir, "enhanced_details.png"), enhanced_details)
            cv2.imwrite(os.path.join(debug_dir, "processed_gray.png"), processed_gray)
            cv2.imwrite(os.path.join(debug_dir, "final_binary.png"), final_binary)
            
            # Save the final processed image
            output_path = os.path.join(output_dir, "preprocessed.png")
            cv2.imwrite(output_path, final_binary)
            
            self.logger.info(f"Enhanced image preprocessing completed: {output_path}")
            self.logger.info(f"Debug images saved to: {debug_dir}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Enhanced image preprocessing failed: {e}")
            raise

    def _detect_and_clean_contours(self, image_path: str, output_dir: str) -> str:
        """
        Enhanced contour detection and cleaning for optimal vectorization with detail preservation.
        """
        try:
            # Load binary image
            binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Step 1: Multi-scale contour detection
            # Detect contours at different scales to capture various detail levels
            scales = [1.0, 0.75, 1.25]
            all_contours = []
            all_hierarchies = []
            
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
                contours1, hierarchy1 = cv2.findContours(
                    scaled_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Method 2: External contours only (for main shapes)
                contours2, hierarchy2 = cv2.findContours(
                    scaled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Method 3: All contours with different approximation
                contours3, hierarchy3 = cv2.findContours(
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
                all_hierarchies.extend([hierarchy1, hierarchy2, hierarchy3])
            
            # Step 2: Advanced contour filtering and cleaning
            filtered_contours = []
            
            for contour in all_contours:
                try:
                    # Validate contour before processing
                    if len(contour) < 3:
                        continue
                    
                    # Calculate contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Enhanced filtering criteria
                    # Keep contours based on multiple criteria
                    keep_contour = False
                    
                    # Criterion 1: Area-based filtering (more lenient for details)
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
                    '--path_precision', '2'
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
                    preview_paths[size_name] = preview_path
                    continue
                except ImportError:
                    pass
                
                # Try svglib as fallback
                try:
                    from svglib.svglib import svg2rlg
                    from reportlab.graphics import renderPM
                    
                    drawing = svg2rlg(svg_path)
                    if drawing:
                        drawing.width = width
                        drawing.height = height
                        renderPM.drawToFile(drawing, preview_path, fmt="PNG")
                        preview_paths[size_name] = preview_path
                        continue
                except ImportError:
                    pass
                
                # Try Inkscape as last resort
                try:
                    cmd = [
                        'inkscape', 
                        '--export-type=png',
                        '--export-filename', preview_path,
                        '--export-width', str(width),
                        '--export-height', str(height),
                        svg_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0 and os.path.exists(preview_path):
                        preview_paths[size_name] = preview_path
                        continue
                except:
                    pass
            
            self.logger.info(f"Generated {len(preview_paths)} preview images")
            return preview_paths
            
        except Exception as e:
            self.logger.error(f"Preview generation failed: {e}")
            return {}

    def _fallback_vectorization(self, image_path: str, output_dir: str, output_format: str) -> Dict[str, str]:
        """
        Fallback vectorization using OpenCV when VTracer is not available.
        """
        try:
            self.logger.info("Using fallback vectorization with OpenCV")
            
            # Load image
            binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create SVG
            height, width = binary.shape
            svg_parts = []
            svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
            svg_parts.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
            svg_parts.append('  <path d="')
            
            # Convert contours to SVG path
            path_parts = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 10:
                    continue
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to path
                if len(approx) > 2:
                    path_data = "M"
                    for j, point in enumerate(approx):
                        x, y = point[0]
                        if j == 0:
                            path_data += f"{x},{y}"
                        else:
                            path_data += f" L{x},{y}"
                    path_data += " Z"
                    path_parts.append(path_data)
            
            svg_parts.append(" ".join(path_parts))
            svg_parts.append('" fill="black" stroke="none"/>')
            svg_parts.append('</svg>')
            
            svg_content = "\n".join(svg_parts)
            
            # Save SVG
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            svg_path = os.path.join(output_dir, f"{base_name}_fallback.svg")
            
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            
            output_paths = {'svg': svg_path}
            
            # Convert to PDF if requested
            if output_format in ['pdf', 'both']:
                pdf_path = self._convert_svg_to_pdf(svg_path, output_dir)
                if pdf_path:
                    output_paths['pdf'] = pdf_path
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Fallback vectorization failed: {e}")
            raise

    # NOTE: generate_high_quality_vector_trace functionality has been moved to generate_vector_trace
    # This method is now deprecated and will be removed in a future version.
    def generate_high_quality_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        DEPRECATED: This method has been moved to generate_vector_trace.
        Use generate_vector_trace instead for high-quality vector tracing.
        """
        return self.generate_vector_trace(file_path, options)

    def generate_ultra_high_quality_vector_trace(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """
        Ultra-high-quality vector tracing for maximum detail preservation.
        Designed to achieve almost mirror image quality for complex logos and lettering.
        """
        start_time = time.time()
        
        try:
            # Extract options with ultra-high-quality defaults
            simplify = options.get('simplify', 0.3)  # Much less simplification for maximum detail
            turdsize = options.get('turdsize', 0)    # No speckle filtering to preserve all details
            noise_reduction = options.get('noise_reduction', False)  # Disable noise reduction to preserve texture
            adaptive_threshold = options.get('adaptive_threshold', True)
            generate_preview = options.get('preview', True)
            output_format = options.get('output_format', 'both')
            preserve_texture = options.get('preserve_texture', True)  # New option for brush stroke preservation
            ultra_detail_mode = options.get('ultra_detail_mode', True)  # New ultra-detail mode
            
            self.logger.info(f"Starting ultra-high-quality vector trace")
            self.logger.info(f"Options: simplify={simplify}, turdsize={turdsize}, ultra_detail_mode={ultra_detail_mode}")
            self.logger.info(f"Ultra-high-quality mode: Maximum detail preservation enabled")
            
            # Create organized output structure
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(self.output_folder, f"{base_name}_ultra_quality_vector_trace")
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
            
            # Step 1: Ultra-High-Quality Image Preprocessing
            processed_image_path = self._ultra_high_quality_preprocessing(
                file_path, processed_dir, noise_reduction, adaptive_threshold, preserve_texture, ultra_detail_mode
            )
            
            # Step 2: Ultra-Detailed Contour Detection
            cleaned_image_path = self._ultra_detailed_contour_detection(
                processed_image_path, processed_dir, ultra_detail_mode
            )
            
            # Step 3: Ultra-High-Quality Vectorization
            vector_results = self._ultra_high_quality_vectorization(
                cleaned_image_path, vector_dir, simplify, turdsize, output_format, ultra_detail_mode
            )
            
            # Step 4: Generate Preview
            preview_paths = {}
            if generate_preview and 'svg' in vector_results:
                preview_paths = self._generate_ultra_quality_preview(
                    vector_results['svg'], preview_dir, base_name
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare ultra-high-quality result
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
                    'preserve_texture': preserve_texture,
                    'ultra_detail_mode': ultra_detail_mode,
                    'quality_level': 'ultra_high'
                },
                'organized_structure': {
                    'original': original_dir,
                    'processed': processed_dir,
                    'vector': vector_dir,
                    'preview': preview_dir
                },
                'quality_analysis': {
                    'ultra_high_quality_preprocessing': True,
                    'multi_scale_ultra_detail_detection': True,
                    'texture_preservation': preserve_texture,
                    'brush_stroke_enhancement': True,
                    'fine_detail_retention': True,
                    'curve_optimization': True
                }
            }
            
            self.logger.info(f"Ultra-high-quality vector trace completed in {processing_time:.2f}s")
            self.logger.info(f"Generated formats: {list(vector_results.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"Ultra-high-quality vector trace failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'processing_time': time.time() - start_time
            }

    def _ultra_high_quality_preprocessing(self, file_path: str, output_dir: str, 
                                        noise_reduction: bool, adaptive_threshold: bool,
                                        preserve_texture: bool, ultra_detail_mode: bool) -> str:
        """
        Ultra-high-quality image preprocessing for maximum detail preservation.
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
            
            # Ultra-high-quality preprocessing
            processed_gray = gray.copy()
            
            if ultra_detail_mode:
                # Step 1: Multi-scale ultra-detail enhancement
                scales = [1.0, 1.25, 1.5, 2.0, 2.5]  # More scales for ultra detail
                enhanced_details = np.zeros_like(gray, dtype=np.float32)
                
                for scale in scales:
                    if scale != 1.0:
                        # Resize for different scales
                        h, w = gray.shape
                        new_h, new_w = int(h * scale), int(w * scale)
                        scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Apply ultra-sharp masking to enhance details
                        blurred = cv2.GaussianBlur(scaled, (0, 0), 1.5)  # Reduced blur for sharper details
                        sharpened = cv2.addWeighted(scaled, 2.0, blurred, -1.0, 0)  # Increased sharpening
                        
                        # Resize back to original size
                        sharpened = cv2.resize(sharpened, (w, h), interpolation=cv2.INTER_CUBIC)
                    else:
                        # Apply ultra-sharp masking to original
                        blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
                        sharpened = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
                    
                    # Add to enhanced details
                    enhanced_details += sharpened.astype(np.float32)
                
                # Average the enhanced details
                enhanced_details = (enhanced_details / len(scales)).astype(np.uint8)
                
                # Step 2: Texture preservation for brush strokes
                if preserve_texture:
                    # Apply edge-preserving filter to maintain brush stroke texture
                    processed_gray = cv2.bilateralFilter(enhanced_details, 15, 50, 50)
                    
                    # Apply unsharp mask to enhance brush stroke edges
                    blurred = cv2.GaussianBlur(processed_gray, (0, 0), 1.0)
                    processed_gray = cv2.addWeighted(processed_gray, 1.8, blurred, -0.8, 0)
                else:
                    processed_gray = enhanced_details
            else:
                processed_gray = gray
            
            # Step 3: Ultra-precise thresholding
            if adaptive_threshold:
                # Use multiple adaptive thresholding methods with ultra-fine parameters
                # Method 1: Gaussian adaptive threshold with fine parameters
                binary1 = cv2.adaptiveThreshold(
                    processed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1
                )
                
                # Method 2: Mean adaptive threshold with fine parameters
                binary2 = cv2.adaptiveThreshold(
                    processed_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2
                )
                
                # Method 3: Otsu's method
                _, binary3 = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Method 4: Manual threshold with fine tuning
                _, binary4 = cv2.threshold(processed_gray, 127, 255, cv2.THRESH_BINARY)
                
                # Combine results with ultra-conservative approach (preserve all details)
                binary = np.zeros_like(binary1)
                
                # Only make pixel white if ALL methods agree it should be white
                binary[(binary1 == 255) & (binary2 == 255) & (binary3 == 255) & (binary4 == 255)] = 255
                
                # For all other cases, keep as black (preserve details)
                binary[(binary1 == 0) | (binary2 == 0) | (binary3 == 0) | (binary4 == 0)] = 0
                
            else:
                # Use Otsu's method with preprocessing
                _, binary = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Ultra-fine post-processing
            if ultra_detail_mode:
                # Apply minimal morphological operations to preserve all details
                kernel_tiny = np.ones((1, 1), np.uint8)
                
                # Only remove isolated single pixels (noise) while preserving all details
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny)
                
                # Apply ultra-sharp detail enhancement
                kernel_ultra_sharp = np.array([[-1, -1, -1, -1, -1],
                                             [-1, -1, -1, -1, -1],
                                             [-1, -1, 25, -1, -1],
                                             [-1, -1, -1, -1, -1],
                                             [-1, -1, -1, -1, -1]], dtype=np.float32)
                enhanced = cv2.filter2D(binary, -1, kernel_ultra_sharp)
                
                # Threshold the enhanced result
                _, final_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            else:
                final_binary = binary
            
            # Step 5: Ensure proper orientation
            black_pixels = np.sum(final_binary == 0)
            white_pixels = np.sum(final_binary == 255)
            
            # If more than 50% is black, invert (assuming we want black shapes on white background)
            if black_pixels > white_pixels * 0.5:
                final_binary = cv2.bitwise_not(final_binary)
            
            # Save debug information
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(debug_dir, "original_gray.png"), gray)
            if ultra_detail_mode:
                cv2.imwrite(os.path.join(debug_dir, "enhanced_details.png"), enhanced_details)
            cv2.imwrite(os.path.join(debug_dir, "processed_gray.png"), processed_gray)
            cv2.imwrite(os.path.join(debug_dir, "final_binary.png"), final_binary)
            
            # Save the final processed image
            output_path = os.path.join(output_dir, "ultra_quality_preprocessed.png")
            cv2.imwrite(output_path, final_binary)
            
            self.logger.info(f"Ultra-high-quality preprocessing completed: {output_path}")
            self.logger.info(f"Debug images saved to: {debug_dir}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ultra-high-quality preprocessing failed: {e}")
            raise
