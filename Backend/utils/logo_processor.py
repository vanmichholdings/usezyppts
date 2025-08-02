# =====================
# Imports
# =====================
import os
import tempfile
import logging
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont
from lxml import etree
import cairosvg
from typing import Dict, Optional, Callable, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum

# PDF and SVG processing imports
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pdf2image import failed: {e}")
    PDF2IMAGE_AVAILABLE = False
except Exception as e:
    print(f"Warning: pdf2image import error: {e}")
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyMuPDF import failed: {e}")
    PYMUPDF_AVAILABLE = False
except Exception as e:
    print(f"Warning: PyMuPDF import error: {e}")
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCV import failed: {e}")
    CV2_AVAILABLE = False
except Exception as e:
    print(f"Warning: OpenCV import error: {e}")
    CV2_AVAILABLE = False

# Optional/advanced imports
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import torch
except ImportError:
    torch = None

# =====================
# Constants & Utilities
# =====================
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.svg']
DEFAULT_SOCIAL_SIZES = {
    'instagram_profile': (180, 180),
    'instagram_post': (1080, 1080),
    'instagram_story': (1080, 1920),
    'facebook_profile': (170, 170),
    'facebook_post': (1200, 630),
    'facebook_cover': (820, 312),
    'twitter_profile': (400, 400),
    'twitter_post': (1024, 512),
    'twitter_header': (1500, 500),
    'youtube_profile': (800, 800),
    'youtube_thumbnail': (1280, 720),
    'youtube_banner': (2560, 1440),
    'linkedin_profile': (400, 400),
    'linkedin_post': (1104, 736),
    'linkedin_banner': (1128, 191),
    'tiktok_profile': (200, 200),
    'tiktok_video': (1080, 1920),
    'slack': (512, 512),
    'discord': (512, 512)
}
        
def _get_base(image_path):
    base_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return base_dir, base_name

# =====================
# Intelligent Content Repurposing Engine
# =====================

class ElementType(Enum):
    """Types of design elements that can be detected and repositioned"""
    LOGO = "logo"
    TITLE = "title"
    SUBTITLE = "subtitle"
    BODY_TEXT = "body_text"
    IMAGE = "image"
    SHAPE = "shape"
    BACKGROUND = "background"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"

class LayoutPreset(Enum):
    """Predefined layout presets for different social media platforms"""
    SQUARE_CENTERED = "square_centered"
    SQUARE_TOP_LEFT = "square_top_left"
    SQUARE_TOP_RIGHT = "square_top_right"
    BANNER_LEFT = "banner_left"
    BANNER_CENTER = "banner_center"
    BANNER_RIGHT = "banner_right"
    STORY_VERTICAL = "story_vertical"
    STORY_HORIZONTAL = "story_horizontal"
    PROFILE_CIRCULAR = "profile_circular"
    THUMBNAIL_16_9 = "thumbnail_16_9"

@dataclass
class DesignElement:
    """Represents a detected design element with its properties"""
    element_type: ElementType
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    content: str = ""
    confidence: float = 0.0
    attributes: Dict[str, Any] = None
    svg_path: str = ""
    priority: int = 1  # Higher number = higher priority for layout
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class LayoutConfig:
    """Configuration for layout adaptation to different platforms"""
    platform: str
    target_size: Tuple[int, int]
    preset: LayoutPreset
    background_fill: str = "transparent"
    maintain_aspect_ratio: bool = True
    element_spacing: float = 0.1  # 10% of container size
    min_element_size: float = 0.05  # 5% of container size
    max_element_size: float = 0.8  # 80% of container size

class IntelligentLayoutEngine:
    """Intelligent content repurposing engine that deconstructs and recomposes designs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.platform_presets = self._initialize_platform_presets()
        
    def _initialize_platform_presets(self) -> Dict[str, LayoutConfig]:
        """Initialize platform-specific layout presets"""
        presets = {}
        
        # Instagram presets
        presets['instagram_profile'] = LayoutConfig(
            platform='instagram_profile',
            target_size=(180, 180),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['instagram_post'] = LayoutConfig(
            platform='instagram_post',
            target_size=(1080, 1080),
            preset=LayoutPreset.SQUARE_CENTERED,
            background_fill="transparent"
        )
        
        presets['instagram_story'] = LayoutConfig(
            platform='instagram_story',
            target_size=(1080, 1920),
            preset=LayoutPreset.STORY_VERTICAL,
            background_fill="transparent"
        )
        
        # Facebook presets
        presets['facebook_profile'] = LayoutConfig(
            platform='facebook_profile',
            target_size=(170, 170),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['facebook_post'] = LayoutConfig(
            platform='facebook_post',
            target_size=(1200, 630),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        presets['facebook_cover'] = LayoutConfig(
            platform='facebook_cover',
            target_size=(820, 312),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        # Twitter/X presets
        presets['twitter_profile'] = LayoutConfig(
            platform='twitter_profile',
            target_size=(400, 400),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['twitter_post'] = LayoutConfig(
            platform='twitter_post',
            target_size=(1024, 512),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        presets['twitter_header'] = LayoutConfig(
            platform='twitter_header',
            target_size=(1500, 500),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        # YouTube presets
        presets['youtube_profile'] = LayoutConfig(
            platform='youtube_profile',
            target_size=(800, 800),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['youtube_thumbnail'] = LayoutConfig(
            platform='youtube_thumbnail',
            target_size=(1280, 720),
            preset=LayoutPreset.THUMBNAIL_16_9,
            background_fill="transparent"
        )
        
        presets['youtube_banner'] = LayoutConfig(
            platform='youtube_banner',
            target_size=(2560, 1440),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        # LinkedIn presets
        presets['linkedin_profile'] = LayoutConfig(
            platform='linkedin_profile',
            target_size=(400, 400),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['linkedin_post'] = LayoutConfig(
            platform='linkedin_post',
            target_size=(1104, 736),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        presets['linkedin_banner'] = LayoutConfig(
            platform='linkedin_banner',
            target_size=(1128, 191),
            preset=LayoutPreset.BANNER_CENTER,
            background_fill="transparent"
        )
        
        # TikTok presets
        presets['tiktok_profile'] = LayoutConfig(
            platform='tiktok_profile',
            target_size=(200, 200),
            preset=LayoutPreset.PROFILE_CIRCULAR,
            background_fill="transparent"
        )
        
        presets['tiktok_video'] = LayoutConfig(
            platform='tiktok_video',
            target_size=(1080, 1920),
            preset=LayoutPreset.STORY_VERTICAL,
            background_fill="transparent"
        )
        
        # Slack and Discord presets
        presets['slack'] = LayoutConfig(
            platform='slack',
            target_size=(512, 512),
            preset=LayoutPreset.SQUARE_CENTERED,
            background_fill="transparent"
        )
        
        presets['discord'] = LayoutConfig(
            platform='discord',
            target_size=(512, 512),
            preset=LayoutPreset.SQUARE_CENTERED,
            background_fill="transparent"
        )
        
        return presets
    
    def parse_svg_design(self, svg_path: str) -> List[DesignElement]:
        """Parse SVG design using intelligent element extraction"""
        self.logger.info(f'🔍 Parsing SVG design: {svg_path}')
        
        try:
            # Try advanced parsing first
            elements = self._parse_svg_basic(svg_path)
            if elements:
                self.logger.info(f'✅ Extracted {len(elements)} elements from SVG')
                return elements
            
            # Fallback to basic parsing
            self.logger.warning(f'⚠️ Advanced SVG parsing failed, using basic parsing')
            return []
            
        except Exception as e:
            self.logger.error(f'❌ SVG design parsing failed: {e}')
            import traceback
            self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
            return []
    
    def _parse_svg_basic(self, svg_path: str) -> List[DesignElement]:
        """Basic SVG parsing using ElementTree"""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Get SVG dimensions
            width = float(root.get('width', 100))
            height = float(root.get('height', 100))
            
            elements = []
            
            # Extract basic elements
            for elem in root.iter():
                tag = elem.tag.split('}')[-1]  # Remove namespace
                
                if tag in ['text', 'tspan']:
                    # Text element
                    x = float(elem.get('x', 0))
                    y = float(elem.get('y', 0))
                    text_content = elem.text or ""
                    
                    if text_content.strip():
                        element = DesignElement(
                            element_type=ElementType.TITLE if len(text_content) < 50 else ElementType.BODY_TEXT,
                            bbox=(x, y, len(text_content) * 10, 20),  # Estimate size
                            content=text_content,
                            confidence=0.8
                        )
                        elements.append(element)
                
                elif tag in ['rect', 'circle', 'ellipse', 'path']:
                    # Shape element
                    x = float(elem.get('x', 0))
                    y = float(elem.get('y', 0))
                    w = float(elem.get('width', 50))
                    h = float(elem.get('height', 50))
                    
                    element = DesignElement(
                        element_type=ElementType.SHAPE,
                        bbox=(x, y, w, h),
                        content="",
                        confidence=0.7
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f'❌ Error in basic SVG parsing: {e}')
            return []
    
    def _extract_svg_elements(self, svg_data: dict, elements: List[DesignElement], parent_bbox: Tuple[float, float, float, float] = None):
        """Recursively extract elements from SVG data structure"""
        if not isinstance(svg_data, dict):
            return
        
        tag = svg_data.get('tag', '')
        attributes = svg_data.get('attributes', {})
        children = svg_data.get('children', [])
        
        # Determine element type and extract properties
        element_type = self._classify_element(tag, attributes)
        
        if element_type != ElementType.UNKNOWN:
            bbox = self._extract_bbox(attributes, parent_bbox)
            content = self._extract_content(attributes, children)
            
            element = DesignElement(
                element_type=element_type,
                bbox=bbox,
                content=content,
                confidence=0.8,
                attributes=attributes
            )
            elements.append(element)
        
        # Process children recursively
        for child in children:
            self._extract_svg_elements(child, elements, bbox if 'bbox' in locals() else parent_bbox)
    
    def _classify_element(self, tag: str, attributes: dict) -> ElementType:
        """Classify SVG element based on tag and attributes"""
        if tag in ['text', 'tspan']:
            text_content = attributes.get('text', '')
            if len(text_content) < 20:
                return ElementType.TITLE
            elif len(text_content) < 100:
                return ElementType.SUBTITLE
            else:
                return ElementType.BODY_TEXT
        
        elif tag in ['image', 'img']:
            return ElementType.IMAGE
        
        elif tag in ['rect', 'circle', 'ellipse', 'path', 'polygon']:
            # Check if it's a logo (small, positioned in corner)
            x = float(attributes.get('x', 0))
            y = float(attributes.get('y', 0))
            width = float(attributes.get('width', 50))
            height = float(attributes.get('height', 50))
            
            if width < 100 and height < 100 and (x < 50 or y < 50):
                return ElementType.LOGO
            else:
                return ElementType.SHAPE
        
        elif tag == 'svg':
            return ElementType.BACKGROUND
        
        return ElementType.UNKNOWN
    
    def _extract_bbox(self, attributes: dict, parent_bbox: Tuple[float, float, float, float] = None) -> Tuple[float, float, float, float]:
        """Extract bounding box from element attributes"""
        x = float(attributes.get('x', 0))
        y = float(attributes.get('y', 0))
        width = float(attributes.get('width', 50))
        height = float(attributes.get('height', 50))
        
        if parent_bbox:
            x += parent_bbox[0]
            y += parent_bbox[1]
        
        return (x, y, width, height)
    
    def _extract_content(self, attributes: dict, children: list) -> str:
        """Extract text content from element"""
        content = attributes.get('text', '')
        
        for child in children:
            if isinstance(child, dict) and child.get('tag') in ['text', 'tspan']:
                content += child.get('attributes', {}).get('text', '')
        
        return content
    
    def parse_pdf_design(self, pdf_path: str) -> List[DesignElement]:
        """Parse PDF design using intelligent element extraction"""
        self.logger.info(f'🔍 Parsing PDF design: {pdf_path}')
        
        try:
            if not PDF2IMAGE_AVAILABLE:
                self.logger.error(f'❌ PDF2IMAGE not available for PDF parsing')
                return []
            
            # Convert PDF to image for analysis
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, first_page=0, last_page=0)
                
                if not images:
                    self.logger.error(f'❌ Could not convert PDF to image')
                    return []
                
                # Save first page as temporary image for raster analysis
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    images[0].save(tmp_file.name, 'PNG')
                    tmp_path = tmp_file.name
                
                try:
                    # Use raster parsing on the converted image
                    elements = self.parse_raster_design(tmp_path)
                    self.logger.info(f'✅ Extracted {len(elements)} elements from PDF')
                    return elements
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
            except Exception as e:
                self.logger.error(f'❌ PDF2IMAGE runtime error: {e}')
                return []
                    
        except Exception as e:
            self.logger.error(f'❌ PDF design parsing failed: {e}')
            import traceback
            self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
            return []
    
    def parse_raster_design(self, image_path: str) -> List[DesignElement]:
        """Parse raster image using improved content-aware segmentation"""
        self.logger.info(f'🔍 Parsing raster design: {image_path}')
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f'❌ Could not load image: {image_path}')
                return []
            
            elements = []
            
            # Convert to different color spaces for better analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # 1. Detect and extract text regions with actual content
            try:
                text_elements = self._detect_text_with_content(img)
                elements.extend(text_elements)
                self.logger.info(f'📝 Detected {len(text_elements)} text elements')
            except Exception as e:
                self.logger.warning(f'⚠️ Text detection failed: {e}')
            
            # 2. Detect prominent visual elements (logos, graphics)
            try:
                visual_elements = self._detect_visual_elements(img, hsv)
                elements.extend(visual_elements)
                self.logger.info(f'🎨 Detected {len(visual_elements)} visual elements')
            except Exception as e:
                self.logger.warning(f'⚠️ Visual element detection failed: {e}')
            
            # 3. Detect background regions
            try:
                background_elements = self._detect_background_regions(img, hsv)
                elements.extend(background_elements)
                self.logger.info(f'🌅 Detected {len(background_elements)} background elements')
            except Exception as e:
                self.logger.warning(f'⚠️ Background detection failed: {e}')
            
            # 4. Detect shapes and geometric elements
            try:
                shape_elements = self._detect_geometric_shapes(img)
                elements.extend(shape_elements)
                self.logger.info(f'🔷 Detected {len(shape_elements)} shape elements')
            except Exception as e:
                self.logger.warning(f'⚠️ Shape detection failed: {e}')
            
            self.logger.info(f'✅ Extracted {len(elements)} total design elements from raster image')
            return elements
            
        except Exception as e:
            self.logger.error(f'❌ Raster design parsing failed: {e}')
            import traceback
            self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
            return []
    
    def _detect_text_with_content(self, img: np.ndarray) -> List[DesignElement]:
        """Detect text regions and extract actual content using improved methods"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use multiple thresholding methods for better text detection
        # Method 1: Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 2: Otsu thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 3: Canny edge detection for text edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine methods
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, edges)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Group contours into text lines
        text_regions = self._group_text_contours(contours, img.shape)
        
        # Create elements for each text region
        for i, (x, y, w, h) in enumerate(text_regions):
            # Extract the actual text region from the image
            text_region = img[y:y+h, x:x+w]
            
            # Determine text type based on size and position
            if h > 40:  # Large text
                element_type = ElementType.TITLE
                priority = 4
                content = f"TITLE_{i}"  # Use meaningful content
            elif h > 20:  # Medium text
                element_type = ElementType.SUBTITLE
                priority = 3
                content = f"SUBTITLE_{i}"
            else:  # Small text
                element_type = ElementType.BODY_TEXT
                priority = 2
                content = f"TEXT_{i}"
            
            element = DesignElement(
                element_type=element_type,
                bbox=(x, y, w, h),
                content=content,
                confidence=0.8,
                priority=priority,
                attributes={
                    'original_bbox': (x, y, w, h),
                    'text_region': text_region,  # Store actual image region
                    'is_text': True
                }
            )
            elements.append(element)
        
        return elements
    
    def _group_text_contours(self, contours: List, img_shape: Tuple) -> List[Tuple]:
        """Group contours into meaningful text lines"""
        text_regions = []
        used_contours = set()
        
        for i, contour in enumerate(contours):
            if i in used_contours:
                continue
            
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this looks like text
            aspect_ratio = w / h if h > 0 else 0
            if not (0.2 < aspect_ratio < 15):  # Text-like aspect ratios
                continue
            
            # Find nearby contours on the same line
            group_contours = [contour]
            used_contours.add(i)
            
            y_tolerance = h * 0.6  # 60% of current contour height
            
            for j, other_contour in enumerate(contours):
                if j in used_contours:
                    continue
                
                other_area = cv2.contourArea(other_contour)
                if other_area < 100:
                    continue
                
                ox, oy, ow, oh = cv2.boundingRect(other_contour)
                other_aspect_ratio = ow / oh if oh > 0 else 0
                
                # Check if contours are on the same horizontal line
                if (0.2 < other_aspect_ratio < 15 and 
                    abs(oy - y) < y_tolerance and 
                    abs(oh - h) < h * 0.4):  # Similar height
                    
                    group_contours.append(other_contour)
                    used_contours.add(j)
            
            # Create bounding box for the group
            if len(group_contours) > 0:
                all_points = np.vstack([cv2.boundingRect(c) for c in group_contours])
                min_x = np.min(all_points[:, 0])
                min_y = np.min(all_points[:, 1])
                max_x = np.max(all_points[:, 0] + all_points[:, 2])
                max_y = np.max(all_points[:, 1] + all_points[:, 3])
                
                group_width = max_x - min_x
                group_height = max_y - min_y
                
                # Only add if the group is substantial
                if group_width > 50 and group_height > 15:
                    text_regions.append((min_x, min_y, group_width, group_height))
        
        return text_regions
    
    def _detect_visual_elements(self, img: np.ndarray, hsv: np.ndarray) -> List[DesignElement]:
        """Detect prominent visual elements like logos and graphics"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find prominent visual elements
        edges = cv2.Canny(gray, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 500:  # Skip small elements
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this is a prominent visual element
            aspect_ratio = w / h if h > 0 else 0
            
            # Look for elements that are roughly square or have interesting shapes
            if 0.3 < aspect_ratio < 3.0 and area > 1000:
                # Extract the visual region
                visual_region = img[y:y+h, x:x+w]
                
                # Determine element type based on characteristics
                if aspect_ratio > 0.8 and aspect_ratio < 1.2:  # Roughly square
                    element_type = ElementType.LOGO
                    content = f"LOGO_{i}"
                    priority = 5
                else:
                    element_type = ElementType.DECORATIVE
                    content = f"GRAPHIC_{i}"
                    priority = 3
                
                element = DesignElement(
                    element_type=element_type,
                    bbox=(x, y, w, h),
                    content=content,
                    confidence=0.7,
                    priority=priority,
                    attributes={
                        'original_bbox': (x, y, w, h),
                        'visual_region': visual_region,
                        'is_visual': True
                    }
                )
                elements.append(element)
        
        return elements
    
    def _detect_background_regions(self, img: np.ndarray, hsv: np.ndarray) -> List[DesignElement]:
        """Detect background regions"""
        elements = []
        
        # Find large uniform regions
        height, width = img.shape[:2]
        
        # Create a background element for the entire image
        background_element = DesignElement(
            element_type=ElementType.BACKGROUND,
            bbox=(0, 0, width, height),
            content="BACKGROUND",
            confidence=0.9,
            priority=1,
            attributes={
                'original_bbox': (0, 0, width, height),
                'is_background': True
            }
        )
        elements.append(background_element)
        
        return elements
    
    def _detect_geometric_shapes(self, img: np.ndarray) -> List[DesignElement]:
        """Detect geometric shapes"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 200:  # Skip small shapes
                continue
            
            # Approximate contour to detect shapes
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify shape
            if len(approx) == 3:
                shape_type = "triangle"
            elif len(approx) == 4:
                shape_type = "rectangle"
            elif len(approx) > 8:
                shape_type = "circle"
            else:
                shape_type = "polygon"
            
            # Extract the shape region
            shape_region = img[y:y+h, x:x+w]
            
            element = DesignElement(
                element_type=ElementType.SHAPE,
                bbox=(x, y, w, h),
                content=shape_type,
                confidence=0.6,
                priority=2,
                attributes={
                    'original_bbox': (x, y, w, h),
                    'shape_region': shape_region,
                    'is_shape': True
                }
            )
            elements.append(element)
        
        return elements
    
    def _analyze_background(self, img: np.ndarray) -> List[DesignElement]:
        """Analyze background regions"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find large uniform regions (background)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Large regions only
                x, y, w, h = cv2.boundingRect(contour)
                
                element = DesignElement(
                    element_type=ElementType.BACKGROUND,
                    bbox=(x, y, w, h),
                    content="background",
                    confidence=0.5,
                    priority=1,
                    attributes={'original_bbox': (x, y, w, h)}  # Store original position
                )
                elements.append(element)
        
        return elements
    
    def _create_color_masks(self, color_space: np.ndarray) -> List[np.ndarray]:
        """Create masks for different color ranges"""
        masks = []
        
        if color_space.shape[2] == 3:  # HSV or LAB
            # Create masks for different color ranges
            for i in range(6):  # 6 color ranges
                lower = np.array([i * 30, 50, 50]) if color_space.shape[2] == 3 else np.array([i * 30, 50, 50])
                upper = np.array([(i + 1) * 30, 255, 255]) if color_space.shape[2] == 3 else np.array([(i + 1) * 30, 255, 255])
                
                mask = cv2.inRange(color_space, lower, upper)
                if np.sum(mask) > 1000:  # Only keep masks with significant content
                    masks.append(mask)
        
        return masks
    
    def reflow_layout_for_platform(self, elements: List[DesignElement], config: LayoutConfig, original_image_size: Tuple[int, int] = None) -> List[DesignElement]:
        """Intelligently reflow layout for target platform"""
        self.logger.info(f'🔄 Reflowing layout for {config.platform}: {config.target_size}')
        
        if not elements:
            return []
        
        # Sort elements by priority
        elements.sort(key=lambda x: x.priority, reverse=True)
        
        # Apply platform-specific layout rules
        adapted_elements = []
        target_width, target_height = config.target_size
        
        for element in elements:
            adapted_element = self._adapt_element_for_platform(element, config, original_image_size)
            if adapted_element:
                adapted_elements.append(adapted_element)
        
        # Apply layout preset positioning
        positioned_elements = self._apply_layout_preset(adapted_elements, config)
        
        self.logger.info(f'✅ Layout reflowed: {len(positioned_elements)} elements positioned')
        return positioned_elements
    
    def _adapt_element_for_platform(self, element: DesignElement, config: LayoutConfig, original_image_size: Tuple[int, int] = None) -> Optional[DesignElement]:
        """Adapt individual element for target platform"""
        target_width, target_height = config.target_size
        x, y, width, height = element.bbox
        
        # Use original image dimensions if provided, otherwise assume 1000px base
        if original_image_size:
            orig_width, orig_height = original_image_size
        else:
            orig_width, orig_height = 1000, 1000
        
        # Calculate scaling factors based on original image dimensions
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        # Apply platform-specific scaling rules
        if config.preset == LayoutPreset.PROFILE_CIRCULAR:
            # For profile images, scale to fit in circle
            scale = min(scale_x, scale_y) * 0.8  # 80% to leave margin
            new_width = width * scale
            new_height = height * scale
            
            # Ensure minimum size
            if new_width < target_width * config.min_element_size:
                scale = target_width * config.min_element_size / width
                new_width = width * scale
                new_height = height * scale
            
            # Ensure maximum size
            if new_width > target_width * config.max_element_size:
                scale = target_width * config.max_element_size / width
                new_width = width * scale
                new_height = height * scale
        
        elif config.preset == LayoutPreset.STORY_VERTICAL:
            # For vertical stories, prioritize height and maintain readability
            scale = scale_y * 0.9
            new_width = width * scale
            new_height = height * scale
            
            # Ensure it fits in the vertical format
            if new_height > target_height * 0.8:
                scale = target_height * 0.8 / height
                new_width = width * scale
                new_height = height * scale
        
        elif config.preset == LayoutPreset.BANNER_CENTER:
            # For banners, prioritize width and maintain readability
            scale = scale_x * 0.8
            new_width = width * scale
            new_height = height * scale
            
            # Ensure it fits in the banner format
            if new_width > target_width * 0.9:
                scale = target_width * 0.9 / width
                new_width = width * scale
                new_height = height * scale
        
        elif config.preset == LayoutPreset.SQUARE_CENTERED:
            # For square formats, maintain aspect ratio and fit within bounds
            scale = min(scale_x, scale_y) * 0.8
            new_width = width * scale
            new_height = height * scale
            
            # Ensure minimum and maximum sizes
            min_size = target_width * config.min_element_size
            max_size = target_width * config.max_element_size
            
            if new_width < min_size:
                scale = min_size / width
                new_width = width * scale
                new_height = height * scale
            
            if new_width > max_size:
                scale = max_size / width
                new_width = width * scale
                new_height = height * scale
        
        else:
            # Default scaling - maintain aspect ratio
            scale = min(scale_x, scale_y) * 0.7
            new_width = width * scale
            new_height = height * scale
        
        # Scale position coordinates
        new_x = x * scale_x
        new_y = y * scale_y
        
        # Ensure coordinates are within bounds
        new_x = max(0, min(new_x, target_width - new_width))
        new_y = max(0, min(new_y, target_height - new_height))
        
        # Ensure minimum sizes
        if new_width < 10:
            new_width = 10
        if new_height < 10:
            new_height = 10
        
        # Create adapted element with scaled dimensions
        adapted_element = DesignElement(
            element_type=element.element_type,
            bbox=(new_x, new_y, new_width, new_height),
            content=element.content,
            confidence=element.confidence,
            attributes=element.attributes.copy() if element.attributes else {},
            priority=element.priority
        )
        
        return adapted_element
    
    def _apply_layout_preset(self, elements: List[DesignElement], config: LayoutConfig) -> List[DesignElement]:
        """Apply layout preset positioning to elements"""
        target_width, target_height = config.target_size
        positioned_elements = []
        
        if config.preset == LayoutPreset.SQUARE_CENTERED:
            # Center all elements in square with proper spacing
            center_x = target_width / 2
            center_y = target_height / 2
            
            for i, element in enumerate(elements):
                x, y, width, height = element.bbox
                
                # Ensure element fits within bounds
                if width > target_width * 0.9:
                    width = target_width * 0.9
                if height > target_height * 0.9:
                    height = target_height * 0.9
                
                new_x = center_x - width / 2
                new_y = center_y - height / 2
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_width - width))
                new_y = max(0, min(new_y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
        
        elif config.preset == LayoutPreset.STORY_VERTICAL:
            # Stack elements vertically for story format with better text handling
            current_y = target_height * 0.05  # Start 5% from top
            
            # Separate text and non-text elements
            text_elements = [e for e in elements if e.element_type in [ElementType.TITLE, ElementType.SUBTITLE, ElementType.BODY_TEXT]]
            other_elements = [e for e in elements if e.element_type not in [ElementType.TITLE, ElementType.SUBTITLE, ElementType.BODY_TEXT]]
            
            # Position text elements first (stacked vertically)
            for element in text_elements:
                x, y, width, height = element.bbox
                
                # Ensure text fits within width
                if width > target_width * 0.9:
                    width = target_width * 0.9
                
                new_x = (target_width - width) / 2  # Center horizontally
                new_y = current_y
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_width - width))
                new_y = max(0, min(new_y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
                
                current_y += height + target_height * 0.02  # 2% spacing
            
            # Position other elements below text or in corners
            for i, element in enumerate(other_elements):
                x, y, width, height = element.bbox
                
                # Ensure element fits
                if width > target_width * 0.4:
                    width = target_width * 0.4
                if height > target_height * 0.3:
                    height = target_height * 0.3
                
                if current_y < target_height * 0.7:  # If there's still space
                    new_x = (target_width - width) / 2
                    new_y = current_y
                    current_y += height + target_height * 0.02
                else:  # Position in corners
                    if i % 2 == 0:
                        new_x = target_width * 0.05  # Left side
                    else:
                        new_x = target_width * 0.95 - width  # Right side
                    new_y = target_height * 0.85  # Bottom area
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_width - width))
                new_y = max(0, min(new_y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
        
        elif config.preset == LayoutPreset.BANNER_CENTER:
            # Arrange elements horizontally for banner format with text prioritization
            current_x = target_width * 0.05  # Start 5% from left
            
            # Separate text and non-text elements
            text_elements = [e for e in elements if e.element_type in [ElementType.TITLE, ElementType.SUBTITLE, ElementType.BODY_TEXT]]
            other_elements = [e for e in elements if e.element_type not in [ElementType.TITLE, ElementType.SUBTITLE, ElementType.BODY_TEXT]]
            
            # Position text elements first (left to right)
            for element in text_elements:
                x, y, width, height = element.bbox
                
                # Ensure text fits within height
                if height > target_height * 0.8:
                    height = target_height * 0.8
                
                new_x = current_x
                new_y = (target_height - height) / 2  # Center vertically
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_width - width))
                new_y = max(0, min(new_y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
                
                current_x += width + target_width * 0.02  # 2% spacing
            
            # Position other elements to the right
            for element in other_elements:
                x, y, width, height = element.bbox
                
                # Ensure element fits
                if width > target_width * 0.3:
                    width = target_width * 0.3
                if height > target_height * 0.8:
                    height = target_height * 0.8
                
                new_x = current_x
                new_y = (target_height - height) / 2  # Center vertically
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(new_x, target_width - width))
                new_y = max(0, min(new_y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
                
                current_x += width + target_width * 0.02  # 2% spacing
        
        else:
            # Default: keep original positioning but ensure within bounds
            for element in elements:
                x, y, width, height = element.bbox
                
                # Ensure element fits within bounds
                if width > target_width * 0.9:
                    width = target_width * 0.9
                if height > target_height * 0.9:
                    height = target_height * 0.9
                
                # Ensure coordinates are within bounds
                new_x = max(0, min(x, target_width - width))
                new_y = max(0, min(y, target_height - height))
                
                positioned_element = DesignElement(
                    element_type=element.element_type,
                    bbox=(new_x, new_y, width, height),
                    content=element.content,
                    confidence=element.confidence,
                    attributes=element.attributes,
                    priority=element.priority
                )
                positioned_elements.append(positioned_element)
        
        return positioned_elements
    
    def render_adapted_design(self, elements: List[DesignElement], config: LayoutConfig, output_path: str, original_image_path: str = None) -> bool:
        """Render the adapted design to output file with visual indicators"""
        self.logger.info(f'🎨 Rendering adapted design for {config.platform}')
        
        target_width, target_height = config.target_size
        
        # Create output canvas with background
        if config.background_fill == "transparent":
            canvas = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 0))
        else:
            canvas = Image.new('RGBA', (target_width, target_height), config.background_fill)
        
        draw = ImageDraw.Draw(canvas)
        
        # Load original image for content extraction
        original_img = None
        if original_image_path and os.path.exists(original_image_path):
            try:
                original_img = Image.open(original_image_path).convert('RGBA')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not load original image: {e}')
        
        # Sort elements by priority (higher priority first)
        sorted_elements = sorted(elements, key=lambda x: x.priority, reverse=True)
        
        # Add a subtle indicator that this is intelligently repurposed
        if len(sorted_elements) > 0:
            # Add a small indicator in the corner
            indicator_text = "AI"
            try:
                # Try to use a small font
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw indicator in bottom-right corner
            bbox = draw.textbbox((0, 0), indicator_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            indicator_x = target_width - text_width - 5
            indicator_y = target_height - text_height - 5
            
            # Draw background for indicator
            draw.rectangle([indicator_x-2, indicator_y-2, indicator_x+text_width+2, indicator_y+text_height+2], 
                          fill=(0, 0, 0, 100))
            draw.text((indicator_x, indicator_y), indicator_text, fill=(255, 255, 255, 200), font=font)
        
        # Render each element
        for element in sorted_elements:
            self._render_element(draw, element, config, original_img)
        
        # Save the result
        try:
            canvas.save(output_path, "PNG", optimize=True)
            self.logger.info(f'✅ Rendered design saved: {output_path}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to save rendered design: {e}')
            return False
    
    def _render_element(self, draw: ImageDraw.Draw, element: DesignElement, config: LayoutConfig, original_img: Image.Image = None):
        """Render individual element on canvas using actual image content"""
        # Use the repositioned coordinates from the layout adaptation
        new_x, new_y, new_width, new_height = element.bbox
        
        # Convert to scalar values to avoid numpy array comparison issues
        new_x = float(new_x)
        new_y = float(new_y)
        new_width = float(new_width)
        new_height = float(new_height)
        
        if element.element_type in [ElementType.TITLE, ElementType.SUBTITLE, ElementType.BODY_TEXT]:
            # Render text element using actual image region
            try:
                # Check if we have the actual text region stored
                text_region = element.attributes.get('text_region', None)
                
                if text_region is not None and new_width > 0 and new_height > 0:
                    # Convert numpy array to PIL Image
                    if isinstance(text_region, np.ndarray):
                        # Convert BGR to RGB
                        text_region_rgb = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
                        text_img = Image.fromarray(text_region_rgb)
                        
                        # Resize to target size
                        target_w, target_h = int(new_width), int(new_height)
                        if target_w > 0 and target_h > 0:
                            text_img_resized = text_img.resize((target_w, target_h), Image.LANCZOS)
                            
                            # Paste onto canvas at new position
                            canvas = draw._image
                            canvas.paste(text_img_resized, (int(new_x), int(new_y)))
                        else:
                            # Fallback to colored rectangle
                            draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                         fill=(0, 0, 255, 200), outline=(0, 0, 0, 255))
                    else:
                        # Fallback to colored rectangle
                        draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                     fill=(0, 0, 255, 200), outline=(0, 0, 0, 255))
                else:
                    # Fallback to colored rectangle with text label
                    draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                 fill=(0, 0, 255, 200), outline=(0, 0, 0, 255))
                    
                    # Add text label
                    try:
                        font = ImageFont.load_default()
                        draw.text((new_x + 5, new_y + 5), element.content, fill=(255, 255, 255, 255), font=font)
                    except:
                        pass
                
            except Exception as e:
                self.logger.warning(f'Could not render text element: {e}')
                # Fallback to colored rectangle
                draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                             fill=(0, 0, 255, 200), outline=(0, 0, 0, 255))
        
        elif element.element_type in [ElementType.SHAPE, ElementType.LOGO, ElementType.IMAGE, ElementType.DECORATIVE]:
            # Render visual element using actual image region
            try:
                # Check if we have the actual visual region stored
                visual_region = element.attributes.get('visual_region', None)
                shape_region = element.attributes.get('shape_region', None)
                
                image_region = visual_region or shape_region
                
                if image_region is not None and new_width > 0 and new_height > 0:
                    # Convert numpy array to PIL Image
                    if isinstance(image_region, np.ndarray):
                        # Convert BGR to RGB
                        image_region_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(image_region_rgb)
                        
                        # Resize to target size
                        target_w, target_h = int(new_width), int(new_height)
                        if target_w > 0 and target_h > 0:
                            img_resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
                            
                            # Paste onto canvas at new position
                            canvas = draw._image
                            canvas.paste(img_resized, (int(new_x), int(new_y)))
                        else:
                            # Fallback to colored rectangle
                            color = (255, 0, 0, 200) if element.element_type == ElementType.LOGO else (0, 255, 0, 200)
                            draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                         outline=(0, 0, 0, 255), fill=color)
                    else:
                        # Fallback to colored rectangle
                        color = (255, 0, 0, 200) if element.element_type == ElementType.LOGO else (0, 255, 0, 200)
                        draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                     outline=(0, 0, 0, 255), fill=color)
                else:
                    # Fallback to colored rectangle
                    color = (255, 0, 0, 200) if element.element_type == ElementType.LOGO else (0, 255, 0, 200)
                    draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                 outline=(0, 0, 0, 255), fill=color)
                    
                    # Add element label
                    try:
                        font = ImageFont.load_default()
                        draw.text((new_x + 5, new_y + 5), element.content, fill=(255, 255, 255, 255), font=font)
                    except:
                        pass
                        
            except Exception as e:
                self.logger.warning(f'Could not render visual element: {e}')
                # Fallback to colored rectangle
                color = (255, 0, 0, 200) if element.element_type == ElementType.LOGO else (0, 255, 0, 200)
                draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                             outline=(0, 0, 0, 255), fill=color)
        
        elif element.element_type == ElementType.BACKGROUND:
            # Render background element using actual image region
            try:
                if original_img and new_width > 0 and new_height > 0:
                    # Get the original bounding box from element attributes if available
                    original_bbox = element.attributes.get('original_bbox', None)
                    
                    if original_bbox:
                        # Extract from original position
                        orig_x, orig_y, orig_w, orig_h = original_bbox
                    else:
                        # Use the entire original image as background
                        img_width, img_height = original_img.size
                        orig_x, orig_y = 0, 0
                        orig_w, orig_h = img_width, img_height
                    
                    # Ensure original coordinates are within bounds
                    img_width, img_height = original_img.size
                    orig_x = max(0, min(int(orig_x), img_width - 1))
                    orig_y = max(0, min(int(orig_y), img_height - 1))
                    orig_w = max(1, min(int(orig_w), img_width - orig_x))
                    orig_h = max(1, min(int(orig_h), img_height - orig_y))
                    
                    # Crop and resize background
                    region = original_img.crop((orig_x, orig_y, orig_x + orig_w, orig_y + orig_h))
                    target_w, target_h = config.target_size
                    region_resized = region.resize((target_w, target_h), Image.LANCZOS)
                    
                    # Paste as background at new position (usually 0,0 for full background)
                    canvas = draw._image
                    canvas.paste(region_resized, (int(new_x), int(new_y)), region_resized)
                else:
                    # Fallback to solid background at new position
                    draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                                 fill=(240, 240, 240, 255))
            except Exception as e:
                self.logger.warning(f'Could not render background: {e}')
                # Fallback to solid background at new position
                draw.rectangle([new_x, new_y, new_x + new_width, new_y + new_height], 
                             fill=(240, 240, 240, 255))

# =====================
# LogoProcessor Class
# =====================
class LogoProcessor:
    """
    Robust logo processor supporting all basic, effects, and social variations.
    Each method outputs files in the same directory as the input, using the format 'filename'_(variation).filetype.
    """
    def __init__(self, cache_dir=None, cache_folder=None, upload_folder=None, output_folder=None, temp_folder=None, use_parallel=True, max_workers=16):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir or cache_folder or tempfile.gettempdir()
        self.upload_folder = upload_folder or tempfile.gettempdir()
        self.output_folder = output_folder or tempfile.gettempdir()
        self.temp_folder = temp_folder or tempfile.gettempdir()
        self.social_sizes = DEFAULT_SOCIAL_SIZES.copy()
        
        # Initialize intelligent layout engine
        self.layout_engine = IntelligentLayoutEngine()
        
        # Parallel processing configuration
        self.use_parallel = use_parallel
        # Optimize worker count based on system capabilities and memory constraints
        cpu_count = os.cpu_count() or 1
        # More conservative worker allocation to prevent memory pressure
        self.max_workers = max_workers if max_workers > 0 else min(12, cpu_count * 1.5)  # Reduced from 32 to 12, and from 4x to 1.5x CPU
        self.progress_callback = None
        self.progress_lock = threading.Lock()
        self.progress_data = {}
        
        # Performance optimization configuration
        self.parallel_config = {
            'max_workers': self.max_workers,
            'use_parallel': self.use_parallel,
            'task_priority_enabled': True,
            'memory_limit_mb': 1024,  # Reduced from 2048 to prevent memory pressure
            'timeout_seconds': 300,
            'retry_attempts': 3,
            'batch_size': 2,  # Reduced from 4 to lower memory usage
            'task_parallelization': True,  # Enable task-level parallelization
            'subtask_workers': 2,  # Reduced from 4 to prevent memory pressure
            'max_concurrent_tasks': 4,  # Reduced from 8 to optimize resource usage
            'adaptive_workers': True,  # Dynamically adjust workers based on task complexity
            'memory_threshold': 75,  # Alert when memory usage exceeds 75%
            'enable_gpu_acceleration': False,  # Future enhancement
            'enable_async_io': True,  # Enable async file operations
            'cache_enabled': True,  # Enable result caching
            'cache_ttl': 3600,  # Cache TTL in seconds
            'preload_common_operations': True  # Preload frequently used operations
        }
        
        # Performance tracking
        self.performance_history = []
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0,
            'average_time': 0,
            'success_rate': 1.0
        }

    # ----------- Basic Variations -----------
    def _create_transparent_png(self, image_path):
        """Create transparent PNG version with comprehensive multi-color background removal"""
        self.logger.info(f'📝 Starting transparent PNG processing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        out_path = os.path.join(base_dir, f"{base_name}_transparent.png")
        
        self.logger.info('📸 Loading image for transparency processing...')
        img = Image.open(image_path).convert("RGBA")
        self.logger.info(f'📐 Image dimensions: {img.size[0]}x{img.size[1]} pixels')
        
        self.logger.info('🎭 Processing multi-color background removal...')
        # Use smart background removal that works with all colors
        result_img = self._smart_background_removal(img)
        
        self.logger.info(f'💾 Saving transparent PNG to: {out_path}')
        result_img.save(out_path, "PNG", optimize=True)
        
        # Verify the result
        final_file_size = os.path.getsize(out_path)
        self.logger.info(f'✅ Transparent PNG completed: {final_file_size:,} bytes')
        
        return out_path  # Return path directly for preview endpoints

    def _smart_background_removal(self, img: Image.Image) -> Image.Image:
        """Smart background removal using the proven simple method extended for any background color"""
        import cv2
        import numpy as np
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Step 1: Detect the background color
        background_color = self._detect_background_color(bgr_array)
        self.logger.info(f"Detected background color: RGB{background_color}")
        
        # Step 2: Use the proven simple method but with detected background color
        # This is the same approach that worked perfectly for white backgrounds
        r, g, b = background_color
        
        # Create a mask for the background color with a small tolerance
        # This is the same logic as the original white background method
        tolerance = 35  # Increased tolerance for better edge cleanup
        
        # Convert to RGB for PIL processing
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        
        # Create mask using color distance to avoid overflow issues
        # Calculate Euclidean distance from background color
        color_diff = np.sqrt(
            (rgb_array[:, :, 0].astype(np.float32) - r) ** 2 +
            (rgb_array[:, :, 1].astype(np.float32) - g) ** 2 +
            (rgb_array[:, :, 2].astype(np.float32) - b) ** 2
        )
        
        # Create mask where color distance is within tolerance
        mask = (color_diff <= tolerance)
        
        # Convert mask to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Step 3: Apply the same morphological operations as the original method
        # This preserves the logo elements perfectly
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes in the logo
        mask_filled = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Step 4: Create the final mask (invert so background = 0, logo = 255)
        final_mask = 255 - mask_filled
        
        # Step 5: Apply the mask to create transparent PNG
        # Convert back to PIL Image
        pil_img = Image.fromarray(rgb_array)
        
        # Create RGBA image
        rgba_img = pil_img.convert('RGBA')
        rgba_array = np.array(rgba_img)
        
        # Apply the mask to the alpha channel
        rgba_array[:, :, 3] = final_mask
        
        # Convert back to PIL Image
        result_img = Image.fromarray(rgba_array)
        
        self.logger.info(f"Smart background removal completed successfully")
        return result_img

    def _detect_background_color(self, bgr_array: np.ndarray) -> tuple:
        """Detect the dominant background color by analyzing edge pixels"""
        h, w = bgr_array.shape[:2]
        
        # Sample pixels from the edges (likely to be background)
        edge_pixels = []
        
        # Top and bottom edges
        edge_pixels.extend(bgr_array[0, :].reshape(-1, 3))  # Top row
        edge_pixels.extend(bgr_array[-1, :].reshape(-1, 3))  # Bottom row
        
        # Left and right edges  
        edge_pixels.extend(bgr_array[:, 0].reshape(-1, 3))  # Left column
        edge_pixels.extend(bgr_array[:, -1].reshape(-1, 3))  # Right column
        
        edge_pixels = np.array(edge_pixels)
        
        # Find the most common color among edge pixels
        try:
            from sklearn.cluster import KMeans
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
            kmeans.fit(edge_pixels)
            
            # Get the cluster with the most pixels (likely background)
            labels = kmeans.labels_
            counts = np.bincount(labels)
            dominant_cluster = np.argmax(counts)
            background_color = kmeans.cluster_centers_[dominant_cluster]
            
            # Convert BGR to RGB for consistency
            background_color_rgb = (background_color[2], background_color[1], background_color[0])
            
            self.logger.info(f"Detected background color: RGB{tuple(background_color_rgb.astype(int))}")
            return tuple(background_color_rgb.astype(int))
        except ImportError:
            # Fallback: use most common edge color
            unique_colors, counts = np.unique(edge_pixels.reshape(-1, 3), axis=0, return_counts=True)
            most_common_idx = np.argmax(counts)
            # Convert BGR to RGB
            most_common_color = unique_colors[most_common_idx]
            background_color_rgb = (most_common_color[2], most_common_color[1], most_common_color[0])
            return tuple(background_color_rgb)

    def _create_smart_background_mask(self, bgr_array: np.ndarray, background_color: tuple) -> np.ndarray:
        """Create conservative background mask that removes ONLY background color pixels"""
        import cv2
        
        h, w = bgr_array.shape[:2]
        
        # Convert background color to numpy array for easier comparison
        # Note: background_color is RGB, but bgr_array is BGR
        bg_color_bgr = np.array([background_color[2], background_color[1], background_color[0]], dtype=np.uint8)
        
        # Step 1: Create very conservative mask using exact color matching
        # Use extremely conservative tolerance to preserve logo elements
        tolerance = 5  # Very conservative tolerance - only exact matches
        
        # Calculate color distance for each pixel
        color_diff = np.sqrt(np.sum((bgr_array - bg_color_bgr) ** 2, axis=2))
        
        # Create initial background mask - only pixels very close to background color
        background_mask = (color_diff <= tolerance).astype(np.uint8)
        
        # Step 2: Conservative flood fill from edges to catch connected background areas
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Conservative flood fill parameters
        lo_diff = (tolerance, tolerance, tolerance)
        hi_diff = (tolerance, tolerance, tolerance)
        
        # Create working copy
        bgr_copy = bgr_array.copy()
        
        # Flood fill from edges with very conservative seed points
        seed_points = []
        
        # Add only corner points for very conservative approach
        seed_points.extend([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])
        
        # Add very sparse edge points
        edge_step = max(1, min(w, h) // 50)  # Very sparse sampling
        
        # Top and bottom edges only
        for i in range(0, w, edge_step):
            seed_points.extend([(i, 0), (i, h-1)])
        
        # Left and right edges only  
        for i in range(0, h, edge_step):
            seed_points.extend([(0, i), (w-1, i)])
        
        # Perform flood fill from seed points
        for seed_x, seed_y in seed_points:
            if 0 <= seed_x < w and 0 <= seed_y < h:
                pixel_color = bgr_copy[seed_y, seed_x]
                color_diff_val = np.sqrt(np.sum((pixel_color - bg_color_bgr) ** 2))
                
                if color_diff_val <= tolerance:
                    mask.fill(0)
                    cv2.floodFill(bgr_copy, mask, (seed_x, seed_y), (255, 255, 255), lo_diff, hi_diff)
                    filled_area = mask[1:-1, 1:-1]
                    background_mask = np.logical_or(background_mask, filled_area > 0)
        
        # Step 3: Conservative cleanup - only remove isolated background pixels
        # Find connected components in the background mask
        background_mask_uint8 = background_mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(background_mask_uint8)
        
        # Keep only large background areas, remove small isolated pixels
        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)
            
            # If component is very small, it might be logo detail, so don't mark as background
            if component_size < 100:  # Conservative threshold
                background_mask[component_mask] = 0
        
        # Step 4: Very conservative edge cleanup
        # Find logo areas (where background_mask is 0)
        logo_areas = (background_mask == 0).astype(np.uint8)
        
        # Find contours of logo areas
        contours, _ = cv2.findContours(logo_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # For each substantial logo contour, be very conservative around edges
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Substantial logo area
                # Create contour mask
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(contour_mask, [contour], 255)
                
                # Very conservative dilation to get nearby pixels
                kernel = np.ones((2, 2), np.uint8)  # Tiny kernel for very conservative cleanup
                dilated = cv2.dilate(contour_mask, kernel, iterations=1)
                
                # Find pixels just outside the logo
                edge_ring = dilated - contour_mask
                
                # Check these edge pixels for background color - be very conservative
                edge_pixels = np.where(edge_ring > 0)
                for y, x in zip(edge_pixels[0], edge_pixels[1]):
                    pixel_color = bgr_array[y, x]
                    color_diff_val = np.sqrt(np.sum((pixel_color - bg_color_bgr) ** 2))
                    
                    # Very conservative tolerance for edge pixels
                    if color_diff_val <= tolerance + 2:  # Only slightly more tolerant
                        background_mask[y, x] = 1
        
        self.logger.info(f"Conservative background removal completed: {np.sum(background_mask):,} background pixels detected")
        return background_mask.astype(np.uint8)

    def _refine_mask_with_contours(self, bgr_array: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
        """Refine background mask using contour hierarchy to preserve nested details"""
        import cv2
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use more aggressive thresholding for better contour detection
        # For logos with high contrast, use binary threshold instead of adaptive
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try adaptive threshold as backup
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine both thresholding methods
        combined_thresh = cv2.bitwise_or(thresh, adaptive_thresh)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(combined_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return background_mask
        
        # Create contour-based refinement mask
        h, w = bgr_array.shape[:2]
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Analyze contour hierarchy
        hierarchy = hierarchy[0]  # Remove extra dimension
        
        # Track substantial logo elements
        substantial_contours = []
        
        for i, contour in enumerate(contours):
            if len(contour) < 5:  # Skip very small contours
                continue
            
            # Get hierarchy information
            next_contour, prev_contour, first_child, parent = hierarchy[i]
            
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip very small contours (noise)
            if area < 50:
                continue
            
            # Check if this is an outer contour (no parent) or inner detail (has parent)
            is_outer_contour = parent == -1
            is_inner_detail = parent != -1
            
            # For outer contours, check if they're likely logo elements
            if is_outer_contour:
                # Calculate solidity (area/convex_hull_area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # More aggressive criteria for logo elements
                if solidity > 0.2 and area > 150:  # Lowered solidity threshold
                    # This is likely a logo element, preserve it
                    cv2.fillPoly(contour_mask, [contour], 255)
                    substantial_contours.append(contour)
            
            # For inner details (nested contours), always preserve them
            elif is_inner_detail:
                # Check if the parent contour is substantial (not just noise)
                if parent < len(contours):
                    parent_area = cv2.contourArea(contours[parent])
                    if parent_area > 150:  # Lowered threshold
                        # This is an inner detail of a logo element, preserve it
                        cv2.fillPoly(contour_mask, [contour], 255)
                        substantial_contours.append(contour)
        
        # Combine original background mask with contour refinements
        # Remove background areas, but preserve areas identified as logo details
        refined_mask = background_mask.copy()
        
        # Where we have logo contours, don't mark as background
        logo_areas = contour_mask > 0
        refined_mask[logo_areas] = 0
        
        # Additional pass: expand logo areas slightly to catch edge pixels
        if len(substantial_contours) > 0:
            # Create a mask of all substantial contours
            substantial_mask = np.zeros((h, w), dtype=np.uint8)
            for contour in substantial_contours:
                cv2.fillPoly(substantial_mask, [contour], 255)
            
            # Dilate the substantial logo areas
            kernel = np.ones((5, 5), np.uint8)
            expanded_logo = cv2.dilate(substantial_mask, kernel, iterations=1)
            
            # Don't mark expanded logo areas as background
            refined_mask[expanded_logo > 0] = 0
        
        return refined_mask

    def _apply_edge_antialiasing(self, bgr_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply very conservative edge cleanup to preserve logo elements"""
        import cv2
        
        # Step 1: Clean up small background spots within logo areas
        # Morphological closing to fill small holes
        kernel_close = np.ones((2, 2), np.uint8)  # Very small kernel
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Step 2: Very conservative edge expansion to clean up background pixels around logo
        # Find logo areas (where mask is 0 = keep)
        logo_mask = (mask_closed == 0).astype(np.uint8)
        
        # Very conservative dilation to capture edge background pixels
        edge_cleanup_kernel = np.ones((2, 2), np.uint8)  # Very small kernel for conservative cleanup
        expanded_logo = cv2.dilate(logo_mask, edge_cleanup_kernel, iterations=1)  # Only 1 iteration
        
        # Update mask: expand background removal around logo edges
        mask_cleaned = mask_closed.copy()
        mask_cleaned[expanded_logo > 0] = 0  # Keep these areas (don't make transparent)
        
        # Step 3: Very conservative cleanup using color similarity
        # Get the background color from the original image
        background_color = self._detect_background_color(bgr_array)
        # Convert RGB to BGR for comparison
        bg_color_bgr = np.array([background_color[2], background_color[1], background_color[0]], dtype=np.uint8)
        
        # Calculate color distance for remaining background pixels
        color_diff = np.sqrt(np.sum((bgr_array - bg_color_bgr) ** 2, axis=2))
        
        # Very conservative tolerance for edge cleanup
        edge_tolerance = 8  # Very conservative tolerance
        
        # Mark pixels that are very similar to background color as background
        edge_background = (color_diff <= edge_tolerance).astype(np.uint8)
        
        # Combine with existing mask
        mask_cleaned = np.logical_or(mask_cleaned, edge_background)
        
        # Step 4: Apply Gaussian blur for smooth edges
        blurred_mask = cv2.GaussianBlur(mask_cleaned.astype(np.float32), (3, 3), 0.5)  # Smaller blur
        
        # Step 5: Very conservative cleanup - remove very small background islands
        # Find connected components in the final mask
        final_mask_uint8 = blurred_mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(final_mask_uint8)
        
        # Remove small background islands (less than 200 pixels)
        final_mask = blurred_mask.copy()
        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)
            
            if component_size < 200:  # Conservative threshold
                final_mask[component_mask] = 0  # Remove it (make logo area)
        
        # Normalize and convert back to uint8
        final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Very conservative edge cleanup applied - preserving logo elements")
        return final_mask

    def _simple_background_removal(self, img: Image.Image) -> Image.Image:
        """Fallback simple background removal method for when OpenCV is not available"""
        arr = np.array(img.convert("RGBA"))
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        
        # More conservative white background removal
        # Only remove very light colors that are likely pure background
        mask = (r > 250) & (g > 250) & (b > 250)
        
        # Count pixels being made transparent
        transparent_pixels = np.sum(mask)
        total_pixels = arr.shape[0] * arr.shape[1]
        transparency_percentage = (transparent_pixels / total_pixels) * 100
        
        self.logger.info(f'🎯 Fallback: Making {transparent_pixels:,} pixels transparent ({transparency_percentage:.1f}% of image)')
        
        arr[..., 3][mask] = 0
        return Image.fromarray(arr, 'RGBA')

    def _create_black_version(self, image_path):
        """Convert to grayscale with slight contrast increase, preserving detail"""
        self.logger.info(f'⚫ Starting black version processing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        png_path = os.path.join(base_dir, f"{base_name}_black.png")
        
        self.logger.info('📸 Loading image for black version processing...')
        # Load image and convert to RGBA
        img = Image.open(image_path).convert("RGBA")
        self.logger.info(f'📐 Image dimensions: {img.size[0]}x{img.size[1]} pixels')
        
        self.logger.info('🎨 Converting to grayscale with contrast enhancement...')
        # Convert to grayscale while preserving alpha
        grayscale = img.convert("LA")  # Luminance + Alpha
        
        # Convert back to RGBA format for consistency
        gray_rgba = Image.new("RGBA", img.size)
        for x in range(img.width):
            for y in range(img.height):
                l, a = grayscale.getpixel((x, y))
                # Slight contrast increase: darken darks, lighten lights
                # Using a gentle S-curve for contrast enhancement
                contrast_enhanced = int(((l / 255.0) ** 0.8) * 255)
                gray_rgba.putpixel((x, y), (contrast_enhanced, contrast_enhanced, contrast_enhanced, a))
        
        self.logger.info('🎛️ Applying contrast enhancement...')
        # Alternative faster approach using ImageEnhance
        from PIL import ImageEnhance
        
        # Convert to grayscale
        gray = img.convert("L")
        
        # Increase contrast slightly (1.2 = 20% increase)
        enhancer = ImageEnhance.Contrast(gray)
        contrast_gray = enhancer.enhance(1.2)
        
        self.logger.info('🔄 Merging contrast-enhanced grayscale with alpha channel...')
        # Convert back to RGBA, preserving alpha channel
        if img.mode == "RGBA":
            # Preserve original alpha channel
            r, g, b, a = img.split()
            # Use the contrast-enhanced grayscale for all RGB channels
            result = Image.merge("RGBA", (contrast_gray, contrast_gray, contrast_gray, a))
        else:
            # Convert grayscale to RGB
            result = Image.merge("RGB", (contrast_gray, contrast_gray, contrast_gray))
            result = result.convert("RGBA")  # Ensure RGBA for consistency
        
        self.logger.info(f'💾 Saving black version to: {png_path}')
        result.save(png_path, "PNG", optimize=True)
        
        # Verify the result
        final_file_size = os.path.getsize(png_path)
        self.logger.info(f'✅ Black version completed: {final_file_size:,} bytes')
        
        return png_path  # Return PNG path directly for preview endpoints

    def _create_pdf_version(self, image_path):
        base_dir, base_name = _get_base(image_path)
        pdf_path = os.path.join(base_dir, f"{base_name}_pdf.pdf")
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.svg':
            try:
                cairosvg.svg2pdf(url=image_path, write_to=pdf_path)
            except Exception as e:
                self.logger.warning(f"SVG to PDF conversion failed: {e}")
                pdf_path = None
        else:
            img = Image.open(image_path)
            img.save(pdf_path, "PDF", resolution=300.0)
        return {"pdf": pdf_path}

    def _create_webp_version(self, image_path):
        base_dir, base_name = _get_base(image_path)
        webp_path = os.path.join(base_dir, f"{base_name}_webp.webp")
        img = Image.open(image_path).convert("RGBA")
        img.save(webp_path, "WEBP", quality=95)
        return {"webp": webp_path}

    def _create_favicon(self, image_path):
        base_dir, base_name = _get_base(image_path)
        favicon_path = os.path.join(base_dir, f"{base_name}_favicon.ico")
        img = Image.open(image_path).convert("RGBA")
        sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        icons = [img.resize(size, Image.LANCZOS) for size in sizes]
        icons[0].save(favicon_path, format='ICO', sizes=sizes)
        return {"ico": favicon_path}

    def _create_email_header(self, image_path):
        base_dir, base_name = _get_base(image_path)
        header_path = os.path.join(base_dir, f"{base_name}_emailheader.png")
        img = Image.open(image_path).convert("RGBA")
        header = Image.new("RGBA", (600, 200), (255, 255, 255, 0))
        logo_w, logo_h = img.size
        scale = min(500 / logo_w, 150 / logo_h, 1)
        logo_resized = img.resize((int(logo_w * scale), int(logo_h * scale)), Image.LANCZOS)
        x = (600 - logo_resized.size[0]) // 2
        y = (200 - logo_resized.size[1]) // 2
        header.paste(logo_resized, (x, y), logo_resized)
        header.save(header_path)
        return {"png": header_path}

    # ----------- Effects -----------
    def _create_vector_trace(self, image_path):
        """Create single-color vector trace using vtracer"""
        self.logger.info(f'🔷 Starting vector trace processing for: {os.path.basename(image_path)}')
        
        try:
            result = self.generate_vector_trace(image_path, {})
            if result.get('status') == 'success' and result.get('output_paths'):
                output_paths = result['output_paths']
                out = {}
                if output_paths.get('svg'):
                    out["svg"] = output_paths['svg']
                if output_paths.get('pdf'):
                    out["pdf"] = output_paths['pdf']
                if output_paths.get('eps'):
                    out["eps"] = output_paths['eps']
                
                self.logger.info(f'✅ Vector trace completed: {len(out)} files generated')
                return out
            else:
                self.logger.error(f'❌ Vector trace failed: {result.get("message", "Unknown error")}')
                return {}
        except Exception as e:
            self.logger.error(f'❌ Vector trace exception: {str(e)}')
            return {}

    def _create_full_color_vector_trace(self, image_path):
        """Create full-color vector trace using vtracer"""
        self.logger.info(f'🎨 Starting full color vector trace processing for: {os.path.basename(image_path)}')
        
        try:
            result = self.generate_vector_trace(image_path, {})
            if result.get('status') == 'success' and result.get('output_paths'):
                output_paths = result['output_paths']
                out = {}
                if output_paths.get('svg'):
                    out["svg"] = output_paths['svg']
                if output_paths.get('pdf'):
                    out["pdf"] = output_paths['pdf']
                if output_paths.get('eps'):
                    out["eps"] = output_paths['eps']
                
                self.logger.info(f'✅ Full color vector trace completed: {len(out)} files generated')
                return out
            else:
                self.logger.error(f'❌ Full color vector trace failed: {result.get("message", "Unknown error")}')
                return {}
        except Exception as e:
            self.logger.error(f'❌ Full color vector trace exception: {str(e)}')
            return {}

    def _create_color_separations(self, image_path):
        """Advanced color separation with PMS detection and proper print setup"""
        base_dir, base_name = _get_base(image_path)
        
        # Load and analyze image
        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img)
        
        # Extract non-transparent pixels for color analysis
        alpha_mask = arr[..., 3] > 0
        rgb_pixels = arr[alpha_mask][..., :3]
        
        if len(rgb_pixels) == 0:
            return {"error": "No visible pixels found"}
        
        # Detect unique colors and merge similar ones
        if KMeans is None:
            raise ImportError("scikit-learn is required for color separations.")
        
        # Initial color detection with higher cluster count
        initial_clusters = min(20, len(np.unique(rgb_pixels.reshape(-1, 3), axis=0)))
        kmeans_initial = KMeans(n_clusters=initial_clusters, random_state=42, n_init=3)
        kmeans_initial.fit(rgb_pixels)
        
        # Merge similar colors (within 30 units in RGB space)
        unique_colors = []
        color_threshold = 30
        
        for center in kmeans_initial.cluster_centers_:
            is_similar = False
            for existing_color in unique_colors:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(np.sum((center - existing_color) ** 2))
                if distance < color_threshold:
                    is_similar = True
                    break
            if not is_similar:
                unique_colors.append(center)
        
        num_colors = len(unique_colors)
        
        # Artboard setup (13" x 19" at 300 DPI)
        artboard_width = int(13 * 300)  # 3900 pixels
        artboard_height = int(19 * 300)  # 5700 pixels
        logo_width = int(10 * 300)  # 3000 pixels
        
        # Calculate logo scaling to fit 10" width
        original_width = img.width
        scale_factor = logo_width / original_width
        logo_height = int(img.height * scale_factor)
        
        # Center position on artboard
        center_x = (artboard_width - logo_width) // 2
        center_y = (artboard_height - logo_height) // 2
        
        # Registration mark setup
        reg_mark_size = 30
        reg_positions = [
            (reg_mark_size, reg_mark_size),  # Top-left
            (artboard_width - reg_mark_size * 2, reg_mark_size),  # Top-right
            (reg_mark_size, artboard_height - reg_mark_size * 2),  # Bottom-left
            (artboard_width - reg_mark_size * 2, artboard_height - reg_mark_size * 2)  # Bottom-right
        ]
        
        def create_registration_marks(svg_root):
            """Add registration marks to SVG"""
            for x, y in reg_positions:
                # Create registration mark (circle with crosshairs)
                group = etree.SubElement(svg_root, 'g')
                # Outer circle
                etree.SubElement(group, 'circle', 
                               cx=str(x + reg_mark_size//2), cy=str(y + reg_mark_size//2), 
                               r=str(reg_mark_size//2), fill="none", stroke="black", 
                               **{"stroke-width": "1"})
                # Horizontal line
                etree.SubElement(group, 'line', 
                               x1=str(x), y1=str(y + reg_mark_size//2),
                               x2=str(x + reg_mark_size), y2=str(y + reg_mark_size//2),
                               stroke="black", **{"stroke-width": "1"})
                # Vertical line  
                etree.SubElement(group, 'line',
                               x1=str(x + reg_mark_size//2), y1=str(y),
                               x2=str(x + reg_mark_size//2), y2=str(y + reg_mark_size),
                               stroke="black", **{"stroke-width": "1"})
        
        separations = []
        
        if num_colors <= 4:
            # PMS spot color separation
            self.logger.info(f"Creating {num_colors} PMS spot color separations")
            
            for i, color in enumerate(unique_colors):
                # Create mask for this color
                color_mask = np.zeros(arr.shape[:2], dtype=bool)
                for y in range(arr.shape[0]):
                    for x in range(arr.shape[1]):
                        if alpha_mask[y, x]:
                            pixel = arr[y, x, :3]
                            distance = np.sqrt(np.sum((pixel - color) ** 2))
                            if distance < color_threshold:
                                color_mask[y, x] = True
                
                # Create separation image
                sep_img = Image.new("RGBA", (artboard_width, artboard_height), (255, 255, 255, 0))
                
                # Resize and place logo
                logo_sep = Image.new("RGBA", img.size, (255, 255, 255, 0))
                for y in range(img.height):
                    for x in range(img.width):
                        if color_mask[y, x]:
                            logo_sep.putpixel((x, y), tuple(color.astype(int).tolist()) + (255,))
                
                # Scale and center logo
                logo_resized = logo_sep.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                sep_img.paste(logo_resized, (center_x, center_y), logo_resized)
                
                # Save PNG
                png_path = os.path.join(base_dir, f"{base_name}_pms_{i+1}.png")
                sep_img.save(png_path, "PNG")
                
                # Create AI/SVG file
                svg_path = os.path.join(base_dir, f"{base_name}_pms_{i+1}.svg")
                ai_path = os.path.join(base_dir, f"{base_name}_pms_{i+1}.ai")
                
                # Create SVG
                svg_root = etree.Element('svg', xmlns="http://www.w3.org/2000/svg",
                                       width=str(artboard_width), height=str(artboard_height))
                
                # Add logo paths (simplified for this implementation)
                color_hex = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
                
                # Convert logo to paths (simplified contour approach)
                if cv2 is not None:
                    logo_gray = cv2.cvtColor(np.array(logo_sep), cv2.COLOR_RGBA2GRAY)
                    contours, _ = cv2.findContours(logo_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            # Scale contour to final size and position
                            scaled_contour = contour * scale_factor
                            scaled_contour[:, :, 0] += center_x
                            scaled_contour[:, :, 1] += center_y
                            
                            points = scaled_contour.reshape(-1, 2)
                            if len(points) > 2:
                                path_data = f"M {points[0][0]},{points[0][1]}"
                                for point in points[1:]:
                                    path_data += f" L {point[0]},{point[1]}"
                                path_data += " Z"
                                etree.SubElement(svg_root, 'path', d=path_data, fill=color_hex)
                
                # Add registration marks
                create_registration_marks(svg_root)
                
                # Save SVG
                svg_data = etree.tostring(svg_root, pretty_print=True).decode()
                with open(svg_path, 'w') as f:
                    f.write(svg_data)
                
                # Copy SVG to AI (AI format is SVG-based)
                with open(ai_path, 'w') as f:
                    f.write(svg_data)
                
                separations.append({
                    'png': png_path,
                    'svg': svg_path, 
                    'ai': ai_path,
                    'color': color_hex,
                    'type': 'PMS'
                })
        
        else:
            # CMYK process separation
            self.logger.info("Creating CMYK process separations")
            
            # Convert to CMYK (simplified approximation)
            def rgb_to_cmyk(r, g, b):
                if r == 0 and g == 0 and b == 0:
                    return 0, 0, 0, 100
                
                # Normalize RGB
                r, g, b = r/255.0, g/255.0, b/255.0
                
                # Calculate CMY
                c = 1 - r
                m = 1 - g  
                y = 1 - b
                
                # Calculate K (black)
                k = min(c, m, y)
                
                # Adjust CMY based on K
                if k < 1:
                    c = (c - k) / (1 - k)
                    m = (m - k) / (1 - k)
                    y = (y - k) / (1 - k)
                else:
                    c = m = y = 0
                
                return int(c*100), int(m*100), int(y*100), int(k*100)
            
            cmyk_channels = ['cyan', 'magenta', 'yellow', 'black']
            
            for i, channel in enumerate(cmyk_channels):
                # Create separation image
                sep_img = Image.new("RGBA", (artboard_width, artboard_height), (255, 255, 255, 0))
                
                # Create channel-specific logo
                logo_channel = Image.new("RGBA", img.size, (255, 255, 255, 0))
                
                for y in range(img.height):
                    for x in range(img.width):
                        if alpha_mask[y, x]:
                            r, g, b = arr[y, x, :3]
                            c, m, y_val, k = rgb_to_cmyk(r, g, b)
                            
                            # Get channel value
                            if i == 0:  # Cyan
                                intensity = 255 - int((c/100) * 255)
                                color = (0, intensity, intensity, 255)
                            elif i == 1:  # Magenta  
                                intensity = 255 - int((m/100) * 255)
                                color = (intensity, 0, intensity, 255)
                            elif i == 2:  # Yellow
                                intensity = 255 - int((y_val/100) * 255)
                                color = (intensity, intensity, 0, 255)
                            else:  # Black
                                intensity = 255 - int((k/100) * 255)
                                color = (intensity, intensity, intensity, 255)
                            
                            if intensity < 255:  # Only add if there's ink
                                logo_channel.putpixel((x, y), color)
                
                # Scale and center logo
                logo_resized = logo_channel.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                sep_img.paste(logo_resized, (center_x, center_y), logo_resized)
                
                # Save files
                png_path = os.path.join(base_dir, f"{base_name}_cmyk_{channel}.png")
                svg_path = os.path.join(base_dir, f"{base_name}_cmyk_{channel}.svg")
                ai_path = os.path.join(base_dir, f"{base_name}_cmyk_{channel}.ai")
                
                sep_img.save(png_path, "PNG")
                
                # Create SVG with registration marks
                svg_root = etree.Element('svg', xmlns="http://www.w3.org/2000/svg",
                                       width=str(artboard_width), height=str(artboard_height))
                create_registration_marks(svg_root)
                
                svg_data = etree.tostring(svg_root, pretty_print=True).decode()
                with open(svg_path, 'w') as f:
                    f.write(svg_data)
                with open(ai_path, 'w') as f:
                    f.write(svg_data)
                
                separations.append({
                    'png': png_path,
                    'svg': svg_path,
                    'ai': ai_path, 
                    'channel': channel,
                    'type': 'CMYK'
                })
        
        # Return paths for preview endpoint (PNG files only)
        png_files = [(sep['png'], f"{sep.get('type', '')}_{sep.get('color', sep.get('channel', ''))}") 
                     for sep in separations]
        
        return {"pngs": png_files, "separations": separations}

    def _create_distressed_version(self, image_path):
        """Apply aggressive raster distress filter to logo artwork with visible texture removal"""
        base_dir, base_name = _get_base(image_path)
        out_path = os.path.join(base_dir, f"{base_name}_distressed.png")
        
        # Load image and ensure RGBA
        img = Image.open(image_path).convert("RGBA")
        
        # Create alpha mask to identify active logo pixels only
        alpha_channel = img.split()[-1]
        alpha_array = np.array(alpha_channel)
        logo_mask = alpha_array > 0  # True where logo exists
        
        # Create aggressive distress patterns
        np.random.seed(42)  # For consistent results
        
        # Pattern 1: Large chunk removal (splatter effect)
        large_chunks = np.random.rand(img.height, img.width)
        large_removal = large_chunks > 0.85  # Remove 15% in large chunks
        
        # Pattern 2: Edge erosion (realistic wear)
        edge_erosion = np.random.rand(img.height, img.width)
        edge_removal = edge_erosion > 0.75  # Remove 25% along edges
        
        # Pattern 3: Speckled texture removal
        speckle_noise = np.random.rand(img.height, img.width)
        speckle_removal = speckle_noise > 0.70  # Remove 30% as speckles
        
        # Pattern 4: Scratches and streaks
        scratch_pattern = np.zeros((img.height, img.width), dtype=bool)
        num_scratches = 8
        for _ in range(num_scratches):
            # Random scratch direction and position
            start_x = np.random.randint(0, img.width)
            start_y = np.random.randint(0, img.height)
            angle = np.random.rand() * 2 * np.pi
            length = np.random.randint(20, min(img.width, img.height) // 3)
            width = np.random.randint(2, 6)
            
            # Draw scratch
            for i in range(length):
                x = int(start_x + i * np.cos(angle))
                y = int(start_y + i * np.sin(angle))
                
                # Create scratch width
                for dx in range(-width//2, width//2 + 1):
                    for dy in range(-width//2, width//2 + 1):
                        scratch_x = x + dx
                        scratch_y = y + dy
                        if 0 <= scratch_x < img.width and 0 <= scratch_y < img.height:
                            scratch_pattern[scratch_y, scratch_x] = True
        
        # Combine all distress patterns
        combined_distress = large_removal | edge_removal | speckle_removal | scratch_pattern
        
        # Apply morphological operations for more realistic grunge
        if cv2 is not None:
            # Create organic, connected distress regions
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_distress = cv2.morphologyEx(
                combined_distress.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel_large
            ).astype(bool)
            
            # Add some dilation for chunky effects
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_distress = cv2.dilate(
                combined_distress.astype(np.uint8), 
                kernel_dilate
            ).astype(bool)
        
        # Apply AGGRESSIVE edge detection for realistic wear patterns
        if cv2 is not None:
            # Convert logo mask to find edges
            logo_mask_uint8 = logo_mask.astype(np.uint8) * 255
            edges = cv2.Canny(logo_mask_uint8, 50, 150)
            edge_pixels = edges > 0
            
            # Make edges more susceptible to distress (80% removal chance at edges)
            edge_distress = np.random.rand(img.height, img.width) > 0.2
            combined_distress = combined_distress | (edge_pixels & edge_distress)
        
        # Ensure we don't remove more than 60% (keep at least 40%)
        logo_pixel_count = np.sum(logo_mask)
        potential_removed = np.sum(logo_mask & combined_distress)
        removal_ratio = potential_removed / logo_pixel_count if logo_pixel_count > 0 else 0
        
        if removal_ratio > 0.60:
            # Reduce distress to maintain 40% minimum
            target_removal = 0.55  # Target 55% removal (keeping 45%)
            reduction_factor = target_removal / removal_ratio
            
            # Randomly keep some pixels that would be removed
            random_keep = np.random.rand(img.height, img.width) < (1 - reduction_factor)
            combined_distress = combined_distress & ~random_keep
        
        # Create final distressed image with dramatic effect
        result_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
        img_array = np.array(img)
        
        # Apply distress only to active logo pixels
        for y in range(img.height):
            for x in range(img.width):
                if logo_mask[y, x]:  # Only process logo pixels
                    if not combined_distress[y, x]:  # Keep pixel if not distressed away
                        # Keep original pixel but add slight aging/darkening
                        original_r, original_g, original_b, original_a = img_array[y, x]
                        
                        # Add subtle aging effect to remaining pixels
                        aging_factor = 0.9  # Darken slightly
                        aged_r = int(original_r * aging_factor)
                        aged_g = int(original_g * aging_factor) 
                        aged_b = int(original_b * aging_factor)
                        
                        result_img.putpixel((x, y), (aged_r, aged_g, aged_b, original_a))
        
        # Add realistic wear patterns around remaining edges
        result_array = np.array(result_img)
        
        # Create soft edges around distressed areas for realism
        for y in range(1, img.height - 1):
            for x in range(1, img.width - 1):
                # Check if this pixel was kept but is adjacent to distressed area
                if logo_mask[y, x] and not combined_distress[y, x]:
                    # Count distressed neighbors
                    distressed_neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < img.height and 0 <= nx < img.width and
                                logo_mask[ny, nx] and combined_distress[ny, nx]):
                                distressed_neighbors += 1
                    
                    # If adjacent to 3+ distressed pixels, add edge wear
                    if distressed_neighbors >= 3:
                        current_pixel = list(result_array[y, x])
                        if current_pixel[3] > 0:  # If pixel has alpha
                            # Create frayed edge effect
                            if np.random.rand() > 0.7:  # 30% chance to fray
                                current_pixel[3] = int(current_pixel[3] * 0.3)  # Heavy alpha reduction
                            else:
                                current_pixel[3] = int(current_pixel[3] * 0.7)  # Moderate alpha reduction
                            result_array[y, x] = current_pixel
        
        # Convert back to PIL Image
        result_img = Image.fromarray(result_array)
        
        # Save the result as PNG with transparent background
        result_img.save(out_path, "PNG", optimize=True)
        
        # Verify the distress worked by checking remaining pixels
        final_alpha = np.array(result_img.split()[-1])
        remaining_pixels = np.sum(final_alpha > 0)
        original_pixels = np.sum(logo_mask)
        actual_retention = remaining_pixels / original_pixels if original_pixels > 0 else 0
        
        self.logger.info(f"Distressed effect applied: {actual_retention:.1%} of logo retained ({100 - actual_retention*100:.1f}% removed)")
        
        return out_path

    def _create_halftone(self, image_path):
        """Apply halftone dot pattern to logo only, maintaining original colors"""
        self.logger.info(f'🔘 Starting halftone effect processing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        out_path = os.path.join(base_dir, f"{base_name}_halftone.png")
        
        # Load image and ensure RGBA
        self.logger.info('📸 Loading image for halftone processing...')
        img = Image.open(image_path).convert("RGBA")
        self.logger.info(f'📐 Image dimensions: {img.size[0]}x{img.size[1]} pixels')
        
        # Create alpha mask to identify logo vs background
        self.logger.info('🎭 Creating alpha mask for logo detection...')
        alpha_channel = img.split()[-1]
        alpha_array = np.array(alpha_channel)
        logo_mask = alpha_array > 0  # True where logo exists
        
        logo_pixel_count = np.sum(logo_mask)
        total_pixels = img.size[0] * img.size[1]
        logo_coverage = (logo_pixel_count / total_pixels) * 100
        
        self.logger.info(f'🎯 Logo coverage: {logo_pixel_count:,} pixels ({logo_coverage:.1f}% of image)')
        
        # Create result image
        result_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
        result_array = np.array(result_img)
        img_array = np.array(img)
        
        # Halftone parameters
        dot_spacing = 8  # Distance between dot centers
        max_dot_size = dot_spacing - 2  # Maximum dot radius
        
        self.logger.info(f'🔘 Halftone parameters: dot spacing={dot_spacing}, max dot size={max_dot_size}')
        
        # Process in a grid pattern
        self.logger.info('🔄 Processing halftone dots...')
        dots_created = 0
        
        for center_y in range(dot_spacing // 2, img.height, dot_spacing):
            for center_x in range(dot_spacing // 2, img.width, dot_spacing):
                
                # Sample area around this dot center
                sample_region = []
                logo_pixels_in_region = 0
                total_r, total_g, total_b = 0.0, 0.0, 0.0
                
                # Define sampling area
                sample_radius = dot_spacing // 2
                for dy in range(-sample_radius, sample_radius + 1):
                    for dx in range(-sample_radius, sample_radius + 1):
                        sample_x = center_x + dx
                        sample_y = center_y + dy
                        
                        if (0 <= sample_x < img.width and 0 <= sample_y < img.height):
                            if logo_mask[sample_y, sample_x]:
                                logo_pixels_in_region += 1
                                r, g, b, a = img_array[sample_y, sample_x]
                                total_r += float(r)
                                total_g += float(g)
                                total_b += float(b)
                
                # Skip if no logo pixels in this region
                if logo_pixels_in_region == 0:
                    continue
                
                # Calculate average color for this region
                avg_r = int(total_r / logo_pixels_in_region)
                avg_g = int(total_g / logo_pixels_in_region)
                avg_b = int(total_b / logo_pixels_in_region)
                
                # Calculate brightness/intensity for dot size
                brightness = (avg_r + avg_g + avg_b) / 3
                intensity = brightness / 255.0
                
                # Invert intensity for halftone effect (darker areas = larger dots)
                dot_intensity = 1.0 - intensity
                
                # Calculate dot size based on intensity
                dot_radius = int(dot_intensity * max_dot_size / 2)
                
                if dot_radius > 0:
                    dots_created += 1
                    # Draw halftone dot with original colors
                    for dy in range(-dot_radius, dot_radius + 1):
                        for dx in range(-dot_radius, dot_radius + 1):
                            dot_x = center_x + dx
                            dot_y = center_y + dy
                            
                            # Check if pixel is within dot radius
                            if (0 <= dot_x < img.width and 0 <= dot_y < img.height):
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                if distance <= dot_radius:
                                    # Only place dot where logo originally existed
                                    if logo_mask[dot_y, dot_x]:
                                        # Use original colors with slight darkening for halftone effect
                                        original_r, original_g, original_b, original_a = img_array[dot_y, dot_x]
                                        
                                        # Apply halftone darkening based on dot intensity
                                        halftone_factor = 0.7 + (0.3 * dot_intensity)  # Range: 0.7 to 1.0
                                        
                                        final_r = int(original_r * halftone_factor)
                                        final_g = int(original_g * halftone_factor)
                                        final_b = int(original_b * halftone_factor)
                                        
                                        # Smooth edge falloff for dots
                                        edge_factor = 1.0
                                        if distance > dot_radius - 1:
                                            edge_factor = dot_radius - distance
                                        
                                        if edge_factor > 0:
                                            result_array[dot_y, dot_x] = [
                                                int(final_r * edge_factor),
                                                int(final_g * edge_factor), 
                                                int(final_b * edge_factor),
                                                int(original_a * edge_factor)
                                            ]
        
        self.logger.info(f'🔘 Created {dots_created:,} halftone dots')
        
        # Alternative approach: More traditional halftone with dot patterns
        # This creates more uniform, print-like halftone dots
        
        # Create a more traditional halftone pattern
        self.logger.info('🔄 Creating traditional halftone pattern...')
        traditional_result = Image.new("RGBA", img.size, (255, 255, 255, 0))
        
        # Convert to grayscale for dot size calculation, but keep original colors
        gray = img.convert("L")
        gray_array = np.array(gray)
        
        # Halftone dot spacing and angle
        dot_size = 6
        angle = 15  # degrees
        
        self.logger.info(f'🔘 Traditional halftone: dot size={dot_size}, angle={angle}°')
        
        # Create halftone pattern
        traditional_dots = 0
        for y in range(0, img.height, dot_size):
            for x in range(0, img.width, dot_size):
                
                # Sample original image in this cell
                cell_pixels = []
                cell_colors = []
                
                for dy in range(dot_size):
                    for dx in range(dot_size):
                        px, py = x + dx, y + dy
                        if (px < img.width and py < img.height and logo_mask[py, px]):
                            cell_pixels.append(gray_array[py, px])
                            cell_colors.append(img_array[py, px, :3])  # RGB only
                
                if not cell_pixels:
                    continue
                    
                # Calculate average brightness and color
                avg_brightness = np.mean(cell_pixels)
                avg_color = np.mean(cell_colors, axis=0).astype(int)
                
                # Calculate dot size (darker = larger dots)
                brightness_ratio = avg_brightness / 255.0
                dot_radius = int((1.0 - brightness_ratio) * (dot_size // 2))
                
                if dot_radius > 0:
                    traditional_dots += 1
                    # Draw circular dot in center of cell
                    center_x = x + dot_size // 2
                    center_y = y + dot_size // 2
                    
                    for dy in range(-dot_radius, dot_radius + 1):
                        for dx in range(-dot_radius, dot_radius + 1):
                            dot_x = center_x + dx
                            dot_y = center_y + dy
                            
                            if (0 <= dot_x < img.width and 0 <= dot_y < img.height):
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                if distance <= dot_radius and logo_mask[dot_y, dot_x]:
                                    # Use averaged color for this cell
                                    traditional_result.putpixel((dot_x, dot_y), 
                                                              tuple(avg_color.tolist()) + (255,))
        
        self.logger.info(f'🔘 Created {traditional_dots:,} traditional halftone dots')
        
        # Use the traditional halftone result
        result_img = traditional_result
        
        # Save the result
        self.logger.info(f'💾 Saving halftone result to: {out_path}')
        result_img.save(out_path, "PNG", optimize=True)
        
        # Verify the result
        final_file_size = os.path.getsize(out_path)
        self.logger.info(f'✅ Halftone effect completed: {final_file_size:,} bytes')
        
        return out_path

    # ----------- Social -----------
    def _create_social_formats(self, image_path, selected_formats=None):
        """Create social media format variations that maintain the exact structure and content of the original design"""
        self.logger.info(f'🎨 Starting social media repurposing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Load the original image
        try:
            if file_ext == '.pdf':
                if PDF2IMAGE_AVAILABLE:
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(image_path, first_page=0, last_page=0)
                        if images:
                            original_img = images[0].convert("RGBA")
                            self.logger.info(f'📐 Original PDF dimensions: {original_img.size[0]}x{original_img.size[1]} pixels')
                        else:
                            self.logger.error(f'❌ Could not convert PDF to image - no images returned')
                            return self._create_social_formats_fallback(image_path, selected_formats)
                    except Exception as e:
                        self.logger.error(f'❌ PDF2IMAGE runtime error: {e}')
                        return self._create_social_formats_fallback(image_path, selected_formats)
                else:
                    self.logger.error(f'❌ PDF2IMAGE not available for PDF processing')
                    return self._create_social_formats_fallback(image_path, selected_formats)
            else:
                original_img = Image.open(image_path).convert("RGBA")
                self.logger.info(f'📐 Original image dimensions: {original_img.size[0]}x{original_img.size[1]} pixels')
        except Exception as e:
            self.logger.error(f'❌ Could not load image: {e}')
            return self._create_social_formats_fallback(image_path, selected_formats)
        
        # Get the background color from the original image
        background_color = self._extract_background_color(original_img)
        self.logger.info(f'🎨 Extracted background color: {background_color}')
        
        # Determine which formats to create
        if selected_formats is None:
            # If no specific formats selected, create all (fallback behavior)
            formats_to_create = self.social_sizes
            self.logger.info(f'📋 No specific formats selected, creating all {len(formats_to_create)} formats')
        else:
            # Only create the selected formats
            formats_to_create = {}
            for platform, is_selected in selected_formats.items():
                if is_selected and platform in self.social_sizes:
                    formats_to_create[platform] = self.social_sizes[platform]
            
            self.logger.info(f'📋 Creating {len(formats_to_create)} selected formats: {list(formats_to_create.keys())}')
        
        if not formats_to_create:
            self.logger.warning(f'⚠️ No social formats selected or available')
            return {}
        
        out = {}
        formats_created = 0
        
        # Process each selected social media platform
        for platform_name, target_size in formats_to_create.items():
            self.logger.info(f'📱 Processing {platform_name}: {target_size[0]}x{target_size[1]} pixels')
            
            try:
                # Create the social format by resizing the original design to fit the target dimensions
                out_path = os.path.join(base_dir, f"{base_name}_{platform_name}.png")
                success = self._create_social_format_without_padding(
                    original_img, target_size, background_color, out_path
                )
                
                if success and os.path.exists(out_path):
                    out[platform_name] = out_path
                    file_size = os.path.getsize(out_path)
                    self.logger.info(f'✅ {platform_name} created: {file_size:,} bytes')
                    formats_created += 1
                else:
                    self.logger.error(f'❌ Failed to create {platform_name}')
                        
            except Exception as e:
                self.logger.error(f'❌ Error processing {platform_name}: {e}')
        
        self.logger.info(f'🎉 Social media repurposing completed: {formats_created}/{len(formats_to_create)} formats created')
        
        # Ensure we return at least some result
        if not out:
            self.logger.warning(f'⚠️ No formats created, using fallback')
            return self._create_social_formats_fallback(image_path, selected_formats)
        
        return out
    
    def _extract_background_color(self, img):
        """Extract the dominant background color from the image"""
        try:
            # Convert to RGB for color analysis
            rgb_img = img.convert('RGB')
            
            # Get the corners and edges to determine background color
            width, height = rgb_img.size
            
            # Sample colors from corners and edges
            sample_points = [
                (0, 0),  # Top-left
                (width-1, 0),  # Top-right
                (0, height-1),  # Bottom-left
                (width-1, height-1),  # Bottom-right
                (width//2, 0),  # Top-center
                (width//2, height-1),  # Bottom-center
                (0, height//2),  # Left-center
                (width-1, height//2),  # Right-center
            ]
            
            colors = []
            for x, y in sample_points:
                try:
                    color = rgb_img.getpixel((x, y))
                    colors.append(color)
                except:
                    continue
            
            if not colors:
                return (255, 255, 255, 255)  # Default to white
            
            # Find the most common color (background)
            from collections import Counter
            color_counts = Counter(colors)
            background_color = color_counts.most_common(1)[0][0]
            
            # Convert to RGBA
            return background_color + (255,)
            
        except Exception as e:
            self.logger.warning(f'⚠️ Could not extract background color: {e}')
            return (255, 255, 255, 255)  # Default to white
    
    def _create_social_format_without_padding(self, original_img, target_size, background_color, out_path):
        """Create a social format by resizing the original design to fit target dimensions without padding"""
        try:
            # Calculate the scaling factor to fit the original design within the target dimensions
            original_width, original_height = original_img.size
            target_width, target_height = target_size
            
            # Calculate scaling factors for both dimensions
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            
            # Use the smaller scaling factor to ensure the entire design fits
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize the original image
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a new image with the target size and background color
            if len(background_color) == 3:
                # Convert RGB to RGBA
                bg_color = background_color + (255,)
            else:
                bg_color = background_color
            
            # Create background image
            background_img = Image.new("RGBA", target_size, bg_color)
            
            # Calculate position to center the resized image
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            # Paste the resized image onto the background
            background_img.paste(resized_img, (x, y), resized_img)
            
            # Save the result
            background_img.save(out_path, "PNG")
            
            self.logger.info(f'📐 Resized from {original_width}x{original_height} to {new_width}x{new_height} (scale: {scale:.3f})')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Error creating social format: {e}')
            return False
    
    def _create_single_social_format_fallback(self, image_path, platform_name, target_size):
        """Create a single social format using simple fallback method that maintains original structure"""
        try:
            base_dir, base_name = _get_base(image_path)
            file_ext = os.path.splitext(image_path)[1].lower()
            
            # Load the original image
            if file_ext == '.pdf':
                if PDF2IMAGE_AVAILABLE:
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(image_path, first_page=0, last_page=0)
                        if images:
                            original_img = images[0].convert("RGBA")
                        else:
                            self.logger.error(f'❌ Could not convert PDF to image - no images returned')
                            return None
                    except Exception as e:
                        self.logger.error(f'❌ PDF2IMAGE runtime error: {e}')
                        return None
                else:
                    self.logger.error(f'❌ PDF2IMAGE not available for fallback processing')
                    return None
            else:
                original_img = Image.open(image_path).convert("RGBA")
            
            # Get the background color from the original image
            background_color = self._extract_background_color(original_img)
            
            # Create the social format
            out_path = os.path.join(base_dir, f"{base_name}_{platform_name}.png")
            success = self._create_social_format_without_padding(
                original_img, target_size, background_color, out_path
            )
            
            if success:
                return out_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f'❌ Fallback failed for {platform_name}: {e}')
            return None
    
    def _create_social_formats_fallback(self, image_path, selected_formats=None):
        """Fallback method that maintains the exact structure and content of the original design"""
        self.logger.info(f'📝 Using fallback social media processing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Load the original image
        try:
            if file_ext == '.pdf':
                if PDF2IMAGE_AVAILABLE:
                    self.logger.info(f'📄 Converting PDF to image for fallback processing...')
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(image_path, first_page=0, last_page=0)
                        if images:
                            original_img = images[0].convert("RGBA")
                            self.logger.info(f'✅ PDF converted successfully: {original_img.size}')
                        else:
                            self.logger.error(f'❌ Could not convert PDF to image - no images returned')
                            return {}
                    except Exception as e:
                        self.logger.error(f'❌ PDF2IMAGE runtime error: {e}')
                        return {}
                else:
                    self.logger.error(f'❌ PDF2IMAGE not available for fallback processing')
                    return {}
            else:
                original_img = Image.open(image_path).convert("RGBA")
        except Exception as e:
            self.logger.error(f'❌ Could not open file for fallback processing: {e}')
            import traceback
            self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
            return {}
        
        # Get the background color from the original image
        background_color = self._extract_background_color(original_img)
        self.logger.info(f'🎨 Extracted background color: {background_color}')
        
        # Determine which formats to create
        if selected_formats is None:
            # If no specific formats selected, create all (fallback behavior)
            formats_to_create = self.social_sizes
            self.logger.info(f'📋 No specific formats selected, creating all {len(formats_to_create)} formats')
        else:
            # Only create the selected formats
            formats_to_create = {}
            for platform, is_selected in selected_formats.items():
                if is_selected and platform in self.social_sizes:
                    formats_to_create[platform] = self.social_sizes[platform]
            
            self.logger.info(f'📋 Creating {len(formats_to_create)} selected formats: {list(formats_to_create.keys())}')
        
        if not formats_to_create:
            self.logger.warning(f'⚠️ No social formats selected or available')
            return {}
        
        self.logger.info(f'📐 Original image dimensions: {original_img.size[0]}x{original_img.size[1]} pixels')
        self.logger.info(f'📋 Available social formats: {len(formats_to_create)} platforms')
        
        out = {}
        formats_created = 0
        
        for name, size in formats_to_create.items():
            self.logger.info(f'📱 Processing {name}: {size[0]}x{size[1]} pixels')
            
            try:
                out_path = os.path.join(base_dir, f"{base_name}_{name}.png")
                success = self._create_social_format_without_padding(
                    original_img, size, background_color, out_path
                )
                
                if success and os.path.exists(out_path):
                    out[name] = out_path
                    file_size = os.path.getsize(out_path)
                    self.logger.info(f'✅ {name} created: {file_size:,} bytes')
                    formats_created += 1
                else:
                    self.logger.error(f'❌ Failed to create {name}')
                    
            except Exception as e:
                self.logger.error(f'❌ Error processing {name}: {e}')
        
        self.logger.info(f'🎉 Fallback social media processing completed: {formats_created}/{len(formats_to_create)} formats created')
        return out

    def _create_contour_cutline(self, image_path):
        """Create contour cutline with better edge detection for outer and nested contours"""
        self.logger.info(f'✂️ Starting enhanced contour cutline processing for: {os.path.basename(image_path)}')
        
        base_dir, base_name = _get_base(image_path)
        magenta_rgb = (255, 0, 255)  # Pink/Magenta for spot color
        magenta_hex = '#FF00FF'
        
        if cv2 is None:
            raise ImportError("OpenCV is required for contour cutline.")
        
        self.logger.info('📸 Loading image for contour cutline processing...')
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.logger.info(f'📐 Image dimensions: {img.shape[1]}x{img.shape[0]} pixels')
        
        # Handle alpha channel if present
        if img.shape[-1] == 4:
            self.logger.info('🎭 Processing image with alpha channel...')
            alpha = img[:, :, 3] / 255.0
            rgb = img[:, :, :3].astype(float)
            white_bg = np.ones_like(rgb) * 255
            img = (rgb * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        else:
            img = img[:, :, :3]
        
        # Create full color mask (preserve original colors)
        self.logger.info('🎨 Creating full color mask...')
        full_color_mask = img.copy()
        
        # Enhanced edge detection for better contour extraction
        self.logger.info('🔍 Performing enhanced edge detection...')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        
        # Apply morphological operations to clean up the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours with hierarchy information
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.info(f'📊 Found {len(contours)} total contours')
        
        # Filter contours based on hierarchy and area
        valid_contours = []
        min_area = 50  # Minimum contour area to consider
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Check hierarchy to determine if it's outer or specific nested
                if hierarchy[0][i][3] == -1:  # Outer contour (no parent)
                    valid_contours.append(contour)
                    self.logger.info(f'✅ Added outer contour {i}: area={area:.1f}')
                elif hierarchy[0][i][3] >= 0:  # Nested contour (has parent)
                    # Only include nested contours that are significant (like inner letters)
                    parent_idx = hierarchy[0][i][3]
                    parent_area = cv2.contourArea(contours[parent_idx]) if parent_idx >= 0 else 0
                    
                    # Include if it's a significant nested contour (like inner letters or background spaces)
                    if area > parent_area * 0.1:  # At least 10% of parent area
                        valid_contours.append(contour)
                        self.logger.info(f'✅ Added nested contour {i}: area={area:.1f} (parent area={parent_area:.1f})')
        
        self.logger.info(f'🎯 Selected {len(valid_contours)} valid contours for outline')
        
        # Create pink outline mask with only valid contours
        self.logger.info('🩷 Creating pink outline mask with selected contours...')
        pink_outline_mask = np.zeros_like(img)
        cv2.drawContours(pink_outline_mask, valid_contours, -1, magenta_rgb, thickness=2)
        
        # Save full color mask
        full_color_path = os.path.join(base_dir, f"{base_name}_contour_cutline_fullcolor.png")
        self.logger.info(f'💾 Saving full color mask to: {full_color_path}')
        full_color_pil = Image.fromarray(cv2.cvtColor(full_color_mask, cv2.COLOR_BGR2RGB))
        full_color_pil.save(full_color_path, "PNG", optimize=True)
        
        # Save pink outline mask
        pink_outline_path = os.path.join(base_dir, f"{base_name}_contour_cutline_outline.png")
        self.logger.info(f'💾 Saving pink outline mask to: {pink_outline_path}')
        pink_outline_pil = Image.fromarray(cv2.cvtColor(pink_outline_mask, cv2.COLOR_BGR2RGB))
        pink_outline_pil.save(pink_outline_path, "PNG", optimize=True)
        
        # Create combined SVG with both layers
        self.logger.info('📐 Creating combined SVG with raster mask and spot color cutline...')
        height, width = gray.shape
        svg_path = os.path.join(base_dir, f"{base_name}_contour_cutline_combined.svg")
        
        # Create SVG with both layers
        svg_root = etree.Element('svg', xmlns="http://www.w3.org/2000/svg", 
                                width=str(width), height=str(height),
                                viewBox=f"0 0 {width} {height}")
        
        # Add definitions for patterns or filters if needed
        defs = etree.SubElement(svg_root, 'defs')
        
        # Layer 1: Raster image as background
        # Convert full color mask to base64 for embedding
        import base64
        from io import BytesIO
        
        full_color_buffer = BytesIO()
        full_color_pil.save(full_color_buffer, format='PNG')
        full_color_base64 = base64.b64encode(full_color_buffer.getvalue()).decode()
        
        # Add raster image layer
        image_element = etree.SubElement(svg_root, 'image',
                                       x="0", y="0", width=str(width), height=str(height),
                                       href=f"data:image/png;base64,{full_color_base64}")
        
        # Layer 2: Spot color cutline paths
        cutline_group = etree.SubElement(svg_root, 'g', id="cutline-paths")
        
        for i, contour in enumerate(valid_contours):
            if len(contour) >= 2:
                # Convert contour to SVG path
                path_data = "M "
                for j, point in enumerate(contour):
                    if j == 0:
                        path_data += f"{point[0][0]},{point[0][1]}"
                    else:
                        path_data += f" L {point[0][0]},{point[0][1]}"
                path_data += " Z"
                
                # Add path with spot color styling
                etree.SubElement(cutline_group, 'path',
                               d=path_data,
                               fill="none",
                               stroke=magenta_hex,
                               stroke_width="2",
                               stroke_linecap="round",
                               stroke_linejoin="round")
        
        # Save combined SVG
        svg_data = etree.tostring(svg_root, pretty_print=True).decode()
        with open(svg_path, 'w') as f:
            f.write(svg_data)
        
        # Create combined PDF with both layers
        pdf_path = os.path.join(base_dir, f"{base_name}_contour_cutline_combined.pdf")
        try:
            self.logger.info(f'📄 Creating combined PDF with raster mask and spot color cutline...')
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path, dpi=300)
        except Exception as e:
            self.logger.warning(f"PDF conversion failed: {e}")
            pdf_path = None
        
        # Verify file sizes
        full_color_size = os.path.getsize(full_color_path)
        pink_outline_size = os.path.getsize(pink_outline_path)
        svg_size = os.path.getsize(svg_path)
        
        self.logger.info(f'✅ Full color mask: {full_color_size:,} bytes')
        self.logger.info(f'✅ Pink outline mask: {pink_outline_size:,} bytes')
        self.logger.info(f'✅ Combined SVG: {svg_size:,} bytes')
        if pdf_path:
            pdf_size = os.path.getsize(pdf_path)
            self.logger.info(f'✅ Combined PDF: {pdf_size:,} bytes')
        
        self.logger.info(f'🎉 Enhanced contour cutline completed with {len(valid_contours)} selected contours')
        
        return {
            'full_color': full_color_path,
            'pink_outline': pink_outline_path,
            'svg': svg_path,
            'pdf': pdf_path,
            'contour_count': len(valid_contours)
        }

    # ----------- Parallel Processing Methods -----------
    def process_logo_parallel(self, file_path, options=None):
        """Enhanced parallel processing with task-level parallelization for maximum speed"""
        if options is None:
            options = {}
        
        start_time = time.time()
        self.logger.info(f'🔄 Starting enhanced parallel processing with {self.max_workers} workers')
        
        # Define all variation tasks
        tasks = []
        
        # Basic variations
        if options.get('transparent_png', False):
            tasks.append(('transparent_png', self._create_transparent_png, file_path))
            self.logger.info('📝 Added transparent PNG task')
        if options.get('black_version', False):
            tasks.append(('black_version', self._create_black_version, file_path))
            self.logger.info('⚫ Added black version task')
        if options.get('pdf_version', False):
            tasks.append(('pdf_version', self._create_pdf_version, file_path))
            self.logger.info('📄 Added PDF version task')
        if options.get('webp_version', False):
            tasks.append(('webp_version', self._create_webp_version, file_path))
            self.logger.info('🌐 Added WebP version task')
        if options.get('favicon', False):
            tasks.append(('favicon', self._create_favicon, file_path))
            self.logger.info('🎯 Added favicon task')
        if options.get('email_header', False):
            tasks.append(('email_header', self._create_email_header, file_path))
            self.logger.info('📧 Added email header task')
        
        # Effects variations
        if options.get('vector_trace', False):
            tasks.append(('vector_trace', self._create_vector_trace, file_path))
            self.logger.info('🔷 Added vector trace task')
        if options.get('full_color_vector_trace', False):
            tasks.append(('full_color_vector_trace', self._create_full_color_vector_trace, file_path))
            self.logger.info('🎨 Added full color vector trace task')
        if options.get('color_separations', False):
            tasks.append(('color_separations', self._create_color_separations, file_path))
            self.logger.info('🎨 Added color separations task')
        if options.get('distressed_effect', False):
            tasks.append(('distressed_effect', self._create_distressed_version, file_path))
            self.logger.info('🏚️ Added distressed effect task')
        if options.get('halftone', False):
            tasks.append(('halftone', self._create_halftone, file_path))
            self.logger.info('🔘 Added halftone effect task')
        if options.get('contour_cut', False):
            tasks.append(('contour_cut', self._create_contour_cutline, file_path))
            self.logger.info('✂️ Added contour cut task')
        
        # Social media formats
        social_formats = options.get('social_formats', {})
        if any(social_formats.values()):
            tasks.append(('social_formats', self._create_social_formats, file_path, social_formats))
            enabled_platforms = [k for k, v in social_formats.items() if v]
            self.logger.info(f'📱 Added social media formats task for: {enabled_platforms}')
        
        if not tasks:
            self.logger.warning('⚠️ No variations selected for processing')
            return {
                'success': False,
                'outputs': {},
                'message': 'No variations selected',
                'processing_time': 0
            }
        
        self.logger.info(f'📊 Total tasks to process: {len(tasks)}')
        
        # Calculate total complexity and optimal worker distribution
        total_complexity = sum(self.get_task_complexity(task[0])['score'] for task in tasks)
        self.logger.info(f'📈 Total task complexity score: {total_complexity}')
        
        # Determine if we should use task-level parallelization
        use_task_parallelization = (
            self.parallel_config['task_parallelization'] and 
            total_complexity > 10 and  # Only for complex workloads
            len(tasks) <= self.parallel_config['max_concurrent_tasks']
        )
        
        if use_task_parallelization:
            self.logger.info('🚀 Using task-level parallelization for enhanced performance')
            return self._execute_with_task_parallelization(tasks, file_path, options, start_time)
        else:
            self.logger.info('⚡ Using standard parallel processing')
            return self._execute_standard_parallel(tasks, start_time)
    
    def _execute_with_task_parallelization(self, tasks, file_path, options, start_time):
        """Execute tasks with task-level parallelization"""
        outputs = {}
        messages = []
        success = True
        
        # Calculate available workers for task-level processing
        total_workers = self.max_workers
        max_concurrent_tasks = self.parallel_config['max_concurrent_tasks']
        concurrent_tasks = min(len(tasks), max_concurrent_tasks)
        
        self.logger.info(f'⚙️ Task-level parallelization: {concurrent_tasks} concurrent tasks, {total_workers} total workers')
        
        # Execute tasks with their own worker pools
        with ThreadPoolExecutor(max_workers=concurrent_tasks) as task_executor:
            # Submit tasks for execution
            future_to_task = {}
            for task_name, task_func, args in tasks:
                self.logger.info(f'🚀 Submitting task with parallelization: {task_name}')
                future = task_executor.submit(
                    self.execute_task_with_subtasks, 
                    task_name, 
                    file_path, 
                    options, 
                    total_workers // concurrent_tasks
                )
                future_to_task[future] = task_name
            
            # Collect results as they complete
            completed_tasks = 0
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                completed_tasks += 1
                self.logger.info(f'✅ Task {completed_tasks}/{len(tasks)} completed: {task_name}')
                
                try:
                    result = future.result()
                    if result is not None:  # Changed from 'if result:' to handle empty dicts
                        outputs[task_name] = result
                        if isinstance(result, dict):
                            self.logger.info(f'📁 {task_name} returned dictionary with {len(result)} items')
                        else:
                            self.logger.info(f'📄 {task_name} returned file path: {result}')
                        messages.append(f'{task_name} completed successfully')
                    else:
                        success = False
                        self.logger.error(f'❌ {task_name} failed - no result returned')
                        messages.append(f'{task_name} failed - no result')
                except Exception as e:
                    success = False
                    self.logger.error(f'💥 {task_name} failed with exception: {str(e)}')
                    messages.append(f'{task_name} failed: {str(e)}')
        
        processing_time = time.time() - start_time
        self.logger.info(f'🏁 Task-level parallel processing completed in {processing_time:.2f}s')
        self.logger.info(f'📈 Success rate: {len(outputs)}/{len(tasks)} tasks successful')
        
        # Update performance statistics
        self.update_performance_stats(processing_time, success and len(outputs) > 0)
        
        return {
            'success': success and len(outputs) > 0,
            'outputs': outputs,
            'message': '; '.join(messages) if messages else 'No operations requested',
            'total_outputs': len(outputs),
            'processing_time': processing_time,
            'parallel': True,
            'task_parallelization': True,
            'workers_used': total_workers,
            'tasks_processed': len(tasks),
            'success_rate': len(outputs) / len(tasks) if tasks else 0
        }
    
    def _execute_standard_parallel(self, tasks, start_time):
        """Execute tasks with standard parallel processing"""
        outputs = {}
        messages = []
        success = True
        
        # Optimize worker count for better performance
        optimal_workers = self.optimize_worker_count(len(tasks))
        actual_workers = min(optimal_workers, self.max_workers, len(tasks))
        
        self.logger.info(f'⚙️ Using {actual_workers} workers for {len(tasks)} tasks')
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_name, task_func, args in tasks:
                self.logger.info(f'🚀 Submitting task: {task_name}')
                future = executor.submit(task_func, args)
                future_to_task[future] = task_name
            
            # Collect results as they complete
            completed_tasks = 0
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                completed_tasks += 1
                self.logger.info(f'✅ Task {completed_tasks}/{len(tasks)} completed: {task_name}')
                
                try:
                    result = future.result()
                    if result is not None:  # Changed from 'if result:' to handle empty dicts
                        outputs[task_name] = result
                        if isinstance(result, dict):
                            self.logger.info(f'📁 {task_name} returned dictionary with {len(result)} items')
                        else:
                            self.logger.info(f'📄 {task_name} returned file path: {result}')
                        messages.append(f'{task_name} completed successfully')
                    else:
                        success = False
                        self.logger.error(f'❌ {task_name} failed - no result returned')
                        messages.append(f'{task_name} failed - no result')
                except Exception as e:
                    success = False
                    self.logger.error(f'💥 {task_name} failed with exception: {str(e)}')
                    messages.append(f'{task_name} failed: {str(e)}')
        
        processing_time = time.time() - start_time
        self.logger.info(f'🏁 Standard parallel processing completed in {processing_time:.2f}s for {len(tasks)} variations using {actual_workers} workers')
        self.logger.info(f'📈 Success rate: {len(outputs)}/{len(tasks)} tasks successful')
        
        # Update performance statistics
        self.update_performance_stats(processing_time, success and len(outputs) > 0)
        
        return {
            'success': success and len(outputs) > 0,
            'outputs': outputs,
            'message': '; '.join(messages) if messages else 'No operations requested',
            'total_outputs': len(outputs),
            'processing_time': processing_time,
            'parallel': True,
            'task_parallelization': False,
            'workers_used': actual_workers,
            'tasks_processed': len(tasks),
            'success_rate': len(outputs) / len(tasks) if tasks else 0
        }
    
    def update_progress(self, task_name: str, progress: float, message: str = ''):
        """Update progress for parallel processing"""
        with self.progress_lock:
            self.progress_data[task_name] = {
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }
            if self.progress_callback:
                self.progress_callback(task_name, progress, message)
    
    def get_progress(self) -> dict:
        """Get current progress data"""
        with self.progress_lock:
            return self.progress_data.copy()
    
    def set_progress_callback(self, callback: Callable):
        """Set a callback function for progress updates"""
        self.progress_callback = callback
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            stats = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'total_processed': self.processing_stats['total_processed'],
                'average_time': self.processing_stats['average_time'],
                'success_rate': self.processing_stats['success_rate'],
                'current_workers': self.max_workers,
                'optimal_workers': self.optimize_worker_count(8),  # Example with 8 tasks
                'performance_score': self._calculate_performance_score()
            }
        except ImportError:
            stats = {
                'cpu_usage': 'Unknown (psutil not available)',
                'memory_usage': 'Unknown (psutil not available)',
                'total_processed': self.processing_stats['total_processed'],
                'average_time': self.processing_stats['average_time'],
                'success_rate': self.processing_stats['success_rate'],
                'current_workers': self.max_workers
            }
        
        return stats
    
    def _calculate_performance_score(self) -> float:
        """Calculate a performance score (0-100) based on current metrics"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # CPU efficiency (lower is better, but we want some utilization)
            cpu_score = max(0, 100 - abs(cpu_percent - 70))  # Optimal around 70%
            
            # Memory efficiency (lower usage is better)
            memory_score = max(0, 100 - memory.percent)
            
            # Success rate
            success_score = self.processing_stats['success_rate'] * 100
            
            # Average processing time (faster is better, normalized)
            time_score = max(0, 100 - (self.processing_stats['average_time'] * 10))
            
            # Weighted average
            performance_score = (cpu_score * 0.3 + memory_score * 0.3 + 
                               success_score * 0.2 + time_score * 0.2)
            
            return min(100, max(0, performance_score))
        except:
            return 50.0  # Default score if monitoring unavailable
    
    def _get_memory_usage(self) -> dict:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def optimize_worker_count(self, task_count: int) -> int:
        """Optimize worker count based on task count and current system resources"""
        cpu_count = os.cpu_count() or 1
        
        # Base calculation
        optimal_workers = min(task_count, cpu_count * 2)
        
        # Memory-aware adjustment
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Reduce workers if memory usage is high
            if memory_usage > 80:
                optimal_workers = max(1, optimal_workers // 2)  # Halve workers
                self.logger.warning(f"High memory usage ({memory_usage:.1f}%), reducing workers to {optimal_workers}")
            elif memory_usage > 60:
                optimal_workers = max(1, int(optimal_workers * 0.75))  # Reduce by 25%
                self.logger.info(f"Moderate memory usage ({memory_usage:.1f}%), adjusting workers to {optimal_workers}")
        except ImportError:
            # Fallback if psutil not available
            optimal_workers = min(optimal_workers, cpu_count)
        
        # Ensure we don't exceed max workers
        optimal_workers = min(optimal_workers, self.max_workers)
        
        return max(1, optimal_workers)  # At least 1 worker
    
    def get_task_priority(self, task_name: str) -> int:
        """Get priority for task scheduling (lower number = higher priority)"""
        # Fast tasks get higher priority
        fast_tasks = ['transparent_png', 'black_version', 'pdf_version', 'webp_version', 'favicon', 'email_header']
        # Medium tasks
        medium_tasks = ['contour_cut', 'distressed_effect']
        # Slow tasks get lower priority
        slow_tasks = ['vector_trace', 'full_color_vector_trace', 'color_separations', 'social_formats']
        
        if task_name in fast_tasks:
            return 1
        elif task_name in medium_tasks:
            return 2
        elif task_name in slow_tasks:
            return 3
        else:
            return 4

    # ----------- Main Processing -----------
    def process_logo(self, file_path, options=None):
        """Main logo processing method - ALWAYS uses parallel processing for maximum performance"""
        if options is None:
            options = {}
        
        self.logger.info(f"🚀 Starting logo processing for: {os.path.basename(file_path)}")
        self.logger.info(f"📋 Processing options: {options}")
        
        # Always use parallel processing for better performance
        self.logger.info('⚡ Using parallel processing for maximum performance')
        return self.process_logo_parallel(file_path, options)
    
    def _should_use_parallel(self, options):
        """Determine if parallel processing should be used"""
        # Count how many variations are requested
        variation_count = 0
        
        # Basic variations
        if options.get('transparent_png', False): variation_count += 1
        if options.get('black_version', False): variation_count += 1
        if options.get('pdf_version', False): variation_count += 1
        if options.get('webp_version', False): variation_count += 1
        if options.get('favicon', False): variation_count += 1
        if options.get('email_header', False): variation_count += 1
        
        # Effects variations
        if options.get('vector_trace', False): variation_count += 1
        if options.get('full_color_vector_trace', False): variation_count += 1
        if options.get('color_separations', False): variation_count += 1
        if options.get('distressed_effect', False): variation_count += 1
        if options.get('halftone', False): variation_count += 1
        if options.get('contour_cut', False): variation_count += 1
        
        # Social media formats
        social_formats = options.get('social_formats', {})
        if any(social_formats.values()): variation_count += 1
        
        # Use parallel processing if more than 1 variation is requested
        return variation_count > 1
    
    def _process_logo_sequential(self, file_path, options=None):
        """Original sequential processing method - kept for backward compatibility"""
        if options is None:
            options = {}
        
        start_time = time.time()
        outputs = {}
        success = True
        messages = []
        
        try:
            # Handle vector trace option
            vector_result = None
            if options.get('vector_trace', False):
                self.logger.info("Processing vector trace...")
                vector_result = self.generate_vector_trace(file_path, options)
                
                if vector_result.get('status') == 'success' and vector_result.get('output_paths'):
                    output_paths = vector_result['output_paths']
                    if output_paths.get('svg'):
                        outputs['vector_trace_svg'] = output_paths['svg']
                    if output_paths.get('pdf'):
                        outputs['vector_trace_pdf'] = output_paths['pdf']
                    if output_paths.get('eps'):
                        outputs['vector_trace_eps'] = output_paths['eps']
                    messages.append("Vector trace completed successfully")
                else:
                    success = False
                    messages.append(f"Vector trace failed: {vector_result.get('message', 'Unknown error')}")
            
            # Handle other options (transparent, black, etc.) by calling existing methods
            base_dir, base_name = _get_base(file_path)
            
            if options.get('transparent_png', False):
                try:
                    result = self._create_transparent_png(file_path)
                    outputs['transparent_png'] = result
                    messages.append("Transparent PNG created")
                except Exception as e:
                    self.logger.error(f"Transparent PNG creation failed: {e}")
                    messages.append(f"Transparent PNG failed: {str(e)}")
            
            if options.get('black_version', False):
                try:
                    result = self._create_black_version(file_path)
                    outputs['black_version'] = result
                    messages.append("Black version created")
                except Exception as e:
                    self.logger.error(f"Black version creation failed: {e}")
                    messages.append(f"Black version failed: {str(e)}")
            
            if options.get('pdf_version', False):
                try:
                    result = self._create_pdf_version(file_path)
                    outputs['pdf_version'] = result.get('pdf') if isinstance(result, dict) else result
                    messages.append("PDF version created")
                except Exception as e:
                    self.logger.error(f"PDF version creation failed: {e}")
                    messages.append(f"PDF version failed: {str(e)}")
            
            if options.get('webp_version', False):
                try:
                    result = self._create_webp_version(file_path)
                    outputs['webp_version'] = result.get('webp') if isinstance(result, dict) else result
                    messages.append("WebP version created")
                except Exception as e:
                    self.logger.error(f"WebP version creation failed: {e}")
                    messages.append(f"WebP version failed: {str(e)}")
            
            if options.get('favicon', False):
                try:
                    result = self._create_favicon(file_path)
                    outputs['favicon'] = result.get('ico') if isinstance(result, dict) else result
                    messages.append("Favicon created")
                except Exception as e:
                    self.logger.error(f"Favicon creation failed: {e}")
                    messages.append(f"Favicon failed: {str(e)}")
            
            if options.get('email_header', False):
                try:
                    result = self._create_email_header(file_path)
                    outputs['email_header'] = result.get('png') if isinstance(result, dict) else result
                    messages.append("Email header created")
                except Exception as e:
                    self.logger.error(f"Email header creation failed: {e}")
                    messages.append(f"Email header failed: {str(e)}")
            
            if options.get('full_color_vector_trace', False):
                try:
                    result = self._create_full_color_vector_trace(file_path)
                    outputs['full_color_vector_trace'] = result
                    messages.append("Full color vector trace created")
                except Exception as e:
                    self.logger.error(f"Full color vector trace creation failed: {e}")
                    messages.append(f"Full color vector trace failed: {str(e)}")
            
            if options.get('color_separations', False):
                try:
                    result = self._create_color_separations(file_path)
                    outputs['color_separations'] = result
                    messages.append("Color separations created")
                except Exception as e:
                    self.logger.error(f"Color separations creation failed: {e}")
                    messages.append(f"Color separations failed: {str(e)}")
            
            if options.get('distressed_effect', False):
                try:
                    result = self._create_distressed_version(file_path)
                    outputs['distressed_effect'] = result
                    messages.append("Distressed effect created")
                except Exception as e:
                    self.logger.error(f"Distressed effect creation failed: {e}")
                    messages.append(f"Distressed effect failed: {str(e)}")
            
            if options.get('halftone', False):
                try:
                    result = self._create_halftone(file_path)
                    outputs['halftone'] = result
                    messages.append("Halftone effect created")
                except Exception as e:
                    self.logger.error(f"Halftone effect creation failed: {e}")
                    messages.append(f"Halftone effect failed: {str(e)}")
            
            if options.get('contour_cut', False):
                try:
                    result = self._create_contour_cutline(file_path)
                    outputs['contour_cut'] = result
                    messages.append("Contour cut created")
                except Exception as e:
                    self.logger.error(f"Contour cut creation failed: {e}")
                    messages.append(f"Contour cut failed: {str(e)}")
            
            social_formats = options.get('social_formats', {})
            if any(social_formats.values()):
                try:
                    result = self._create_social_formats(file_path, social_formats)
                    if result is not None:  # Check for None instead of truthiness
                        outputs['social_formats'] = result
                        messages.append("Social formats created")
                    else:
                        success = False
                        messages.append("Social formats failed - no result returned")
                except Exception as e:
                    self.logger.error(f"Social formats creation failed: {e}")
                    messages.append(f"Social formats failed: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Sequential processing failed: {e}")
            success = False
            messages.append(f"Processing failed: {str(e)}")
        
        processing_time = time.time() - start_time
        self.logger.info(f'Sequential processing completed in {processing_time:.2f}s')
        
        return {
            'success': success and len(outputs) > 0,
            'outputs': outputs,
            'message': '; '.join(messages) if messages else 'No operations requested',
            'total_outputs': len(outputs),
            'processing_time': processing_time,
            'parallel': False,
            'workers_used': 1
        }
    
    def generate_vector_trace(self, file_path, options=None):
        """High-quality multi-color vector tracing using vtracer's native RGBA pixel processing"""
        import vtracer
        base_dir, base_name = _get_base(file_path)
        
        # Load image with maximum quality
        img = Image.open(file_path).convert("RGBA")
        img_array = np.array(img)
        
        # Strategy 1: Upscale for maximum detail preservation
        upscale_factor = 8 if max(img.size) < 200 else 4 if max(img.size) < 400 else 2 if max(img.size) < 800 else 1
        
        if upscale_factor > 1:
            upscaled_img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.Resampling.LANCZOS)
            working_array = np.array(upscaled_img)
            working_size = (upscaled_img.width, upscaled_img.height)
        else:
            working_array = img_array
            working_size = img.size
        
        # Strategy 2: Use vtracer's native RGBA pixel processing
        # Convert numpy array to the format vtracer expects (list of RGBA tuples)
        # Convert numpy uint8 to Python int to avoid type issues
        rgba_pixels = [tuple(int(val) for val in row) for row in working_array.reshape(-1, 4)]
        
        print(f"DEBUG: Image size: {working_size}, pixels: {len(rgba_pixels)}, upscale: {upscale_factor}x")
        print(f"DEBUG: First pixel: {rgba_pixels[0]}, type: {type(rgba_pixels[0][0])}")
        
        # Use vtracer with FULL COLOR multi-color processing
        svg_str = vtracer.convert_pixels_to_svg(
            rgba_pixels,
            working_size,
            colormode="color",  # FULL COLOR MODE - not binary!
            hierarchical="stacked",  # Better for complex logos with holes
            mode="spline",  # Smooth curves
            filter_speckle=0,  # No filtering - preserve all details
            color_precision=6,  # High color precision for grouping
            layer_difference=16,  # Low threshold for layer separation
            corner_threshold=15,  # Detect more corners
            length_threshold=0.5,  # Preserve small details
            max_iterations=30,  # Maximum quality
            splice_threshold=15,  # More path splitting
            path_precision=5  # Maximum precision
        )
        
        # Scale down paths if we upscaled
        if upscale_factor > 1:
            # Parse and scale down the SVG paths
            import re
            
            def scale_path_data(match):
                path_data = match.group(1)
                # Scale down all numeric values in the path
                def scale_number(num_match):
                    return str(float(num_match.group(0)) / upscale_factor)
                
                scaled_path = re.sub(r'-?\d+\.?\d*', scale_number, path_data)
                return f'd="{scaled_path}"'
            
            def scale_dimension(match):
                return str(float(match.group(0)) / upscale_factor)
            
            # Scale down the SVG
            svg_str = re.sub(r'd="([^"]*)"', scale_path_data, svg_str)
            svg_str = re.sub(r'width="([^"]*)"', lambda m: f'width="{float(m.group(1)) / upscale_factor}"', svg_str)
            svg_str = re.sub(r'height="([^"]*)"', lambda m: f'height="{float(m.group(1)) / upscale_factor}"', svg_str)
            svg_str = re.sub(r'viewBox="([^"]*)"', lambda m: f'viewBox="{" ".join([str(float(x) / upscale_factor) for x in m.group(1).split()])}"', svg_str)
        
        # Save SVG
        svg_path = os.path.join(base_dir, f"{base_name}_vector.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        
        # Save PDF with maximum quality
        pdf_path = os.path.join(base_dir, f"{base_name}_vector.pdf")
        try:
            cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"), write_to=pdf_path, dpi=900)
        except Exception as e:
            self.logger.warning(f"PDF conversion failed: {e}")
            pdf_path = None
        
        # Save EPS file (replacing AI file)
        eps_path = os.path.join(base_dir, f"{base_name}_vector.eps")
        try:
            # Convert SVG to EPS using cairosvg
            cairosvg.svg2ps(bytestring=svg_str.encode("utf-8"), write_to=eps_path, dpi=900)
        except Exception as e:
            self.logger.warning(f"EPS conversion failed: {e}")
            eps_path = None
        
        self.logger.info(f"Multi-color RGBA vector trace completed for {file_path} with {upscale_factor}x upscaling")
        
        return {
            'status': 'success',
            'output_paths': {
                'svg': svg_path,
                'pdf': pdf_path if pdf_path else None,
                'eps': eps_path if eps_path else None
            },
            'preview_paths': {
                'svg': svg_path,
                'pdf': pdf_path if pdf_path else None
            },
            'processing_time': 0,
            'organized_structure': {},
            'total_paths': 1,
            'quality_level': 'vtracer-multi-color-rgba-direct',
            'algorithm': 'rgba-pixel-direct-processing',
            'upscale_factor': upscale_factor,
            'processing_details': {
                'library': 'vtracer',
                'mode': 'multi-color-rgba',
                'colormode': 'color',
                'hierarchical': 'stacked',
                'color_precision': 6,
                'upscaling': f'{upscale_factor}x',
                'preprocessing': 'Direct RGBA pixel processing',
                'speckle_filtering': 0,
                'corner_threshold': 15,
                'path_precision': 5
            }
        }

    # ----------- Performance Optimization Methods -----------
    def get_optimization_recommendations(self) -> dict:
        """Get performance optimization recommendations based on current system and usage patterns."""
        recommendations = {
            'worker_count': {},
            'memory_usage': {},
            'parallel_processing': {},
            'system_resources': {}
        }
        
        # Analyze current performance
        cpu_count = os.cpu_count() or 1
        current_workers = self.parallel_config['max_workers']
        
        # Worker count recommendations
        if current_workers < cpu_count:
            recommendations['worker_count'] = {
                'current': current_workers,
                'recommended': min(cpu_count * 2, 32),
                'reason': 'Underutilizing CPU cores',
                'priority': 'medium'
            }
        elif current_workers > cpu_count * 4:
            recommendations['worker_count'] = {
                'current': current_workers,
                'recommended': cpu_count * 3,
                'reason': 'Too many workers may cause context switching overhead',
                'priority': 'low'
            }
        else:
            recommendations['worker_count'] = {
                'current': current_workers,
                'recommended': current_workers,
                'reason': 'Optimal worker count',
                'priority': 'none'
            }
        
        # Memory usage analysis
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            if memory_usage > 80:
                recommendations['memory_usage'] = {
                    'current': f"{memory_usage:.1f}%",
                    'recommended': 'Reduce batch size and worker count',
                    'reason': 'High memory usage detected',
                    'priority': 'high'
                }
            elif memory_usage > 60:
                recommendations['memory_usage'] = {
                    'current': f"{memory_usage:.1f}%",
                    'recommended': 'Monitor memory usage',
                    'reason': 'Moderate memory usage',
                    'priority': 'medium'
                }
            else:
                recommendations['memory_usage'] = {
                    'current': f"{memory_usage:.1f}%",
                    'recommended': 'Memory usage is optimal',
                    'reason': 'Low memory usage',
                    'priority': 'none'
                }
        except ImportError:
            recommendations['memory_usage'] = {
                'current': 'Unknown',
                'recommended': 'Install psutil for memory monitoring',
                'reason': 'Memory monitoring not available',
                'priority': 'low'
            }
        
        # Parallel processing recommendations
        if not self.parallel_config['use_parallel']:
            recommendations['parallel_processing'] = {
                'current': 'Disabled',
                'recommended': 'Enable parallel processing',
                'reason': 'Sequential processing is slower',
                'priority': 'high'
            }
        else:
            recommendations['parallel_processing'] = {
                'current': 'Enabled',
                'recommended': 'Parallel processing is optimal',
                'reason': 'Already using parallel processing',
                'priority': 'none'
            }
        
        # System resource recommendations
        recommendations['system_resources'] = {
            'cpu_cores': cpu_count,
            'current_workers': current_workers,
            'worker_cpu_ratio': current_workers / cpu_count if cpu_count > 0 else 0,
            'optimal_worker_range': f"{cpu_count} - {cpu_count * 4}"
        }
        
        return recommendations
    
    def optimize_configuration(self, target: str = 'balanced') -> dict:
        """Optimize configuration based on target performance profile."""
        if target not in ['speed', 'memory', 'balanced']:
            raise ValueError("Target must be 'speed', 'memory', or 'balanced'")
        
        cpu_count = os.cpu_count() or 1
        old_config = self.parallel_config.copy()
        
        if target == 'speed':
            # Optimize for maximum speed
            self.parallel_config.update({
                'max_workers': min(cpu_count * 6, 32),
                'use_parallel': True,
                'task_priority_enabled': True,
                'memory_limit_mb': 4096,
                'timeout_seconds': 600,
                'retry_attempts': 2,
                'batch_size': 8
            })
            self.max_workers = self.parallel_config['max_workers']
            
        elif target == 'memory':
            # Optimize for memory efficiency
            self.parallel_config.update({
                'max_workers': max(1, cpu_count // 2),
                'use_parallel': True,
                'task_priority_enabled': False,
                'memory_limit_mb': 1024,
                'timeout_seconds': 180,
                'retry_attempts': 1,
                'batch_size': 2
            })
            self.max_workers = self.parallel_config['max_workers']
            
        else:  # balanced
            # Balanced optimization
            self.parallel_config.update({
                'max_workers': min(cpu_count * 3, 16),
                'use_parallel': True,
                'task_priority_enabled': True,
                'memory_limit_mb': 2048,
                'timeout_seconds': 300,
                'retry_attempts': 3,
                'batch_size': 4
            })
            self.max_workers = self.parallel_config['max_workers']
        
        # Update use_parallel flag
        self.use_parallel = self.parallel_config['use_parallel']
        
        return {
            'target': target,
            'old_config': old_config,
            'new_config': self.parallel_config.copy(),
            'changes': {
                'max_workers': old_config['max_workers'] != self.parallel_config['max_workers'],
                'memory_limit': old_config['memory_limit_mb'] != self.parallel_config['memory_limit_mb'],
                'batch_size': old_config['batch_size'] != self.parallel_config['batch_size'],
                'timeout': old_config['timeout_seconds'] != self.parallel_config['timeout_seconds']
            }
        }
    
    def update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics after processing."""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_time'] += processing_time
        
        # Update average time
        total_processed = self.processing_stats['total_processed']
        if total_processed > 0:
            self.processing_stats['average_time'] = self.processing_stats['total_time'] / total_processed
        
        # Update success rate
        if success:
            successful = int(self.processing_stats['success_rate'] * (total_processed - 1)) + 1
        else:
            successful = int(self.processing_stats['success_rate'] * (total_processed - 1))
        
        self.processing_stats['success_rate'] = successful / total_processed if total_processed > 0 else 1.0
        
        # Store in history (keep last 100 entries)
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success,
            'config': self.parallel_config.copy()
        })
        
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_detailed_performance_stats(self) -> dict:
        """Get detailed performance statistics including history."""
        stats = self.get_performance_stats()
        stats.update({
            'processing_stats': self.processing_stats,
            'performance_history': self.performance_history[-20:],  # Last 20 entries
            'config': self.parallel_config,
            'recommendations': self.get_optimization_recommendations()
        })
        return stats

    def get_task_complexity(self, task_name: str) -> dict:
        """Get complexity metrics for a task to determine optimal worker allocation"""
        complexity_scores = {
            # Basic variations (low complexity)
            'transparent_png': {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1},
            'black_version': {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1},
            'pdf_version': {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1},
            'webp_version': {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1},
            'favicon': {'score': 2, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 6},  # Multiple sizes
            'email_header': {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1},
            
            # Effects variations (medium to high complexity)
            'vector_trace': {'score': 4, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 3},  # SVG, PDF, EPS
            'full_color_vector_trace': {'score': 4, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 3},
            'color_separations': {'score': 5, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 8},  # Multiple separations
            'distressed_effect': {'score': 3, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 1},
            'halftone': {'score': 3, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 1},
            'contour_cut': {'score': 3, 'cpu_intensive': True, 'io_intensive': False, 'subtasks': 4},  # Multiple outputs
            
            # Social media formats (high complexity due to many formats)
            'social_formats': {'score': 6, 'cpu_intensive': True, 'io_intensive': True, 'subtasks': 20}  # 20+ formats
        }
        
        return complexity_scores.get(task_name, {'score': 1, 'cpu_intensive': False, 'io_intensive': True, 'subtasks': 1})
    
    def calculate_optimal_workers_for_task(self, task_name: str, total_available_workers: int) -> int:
        """Calculate optimal number of workers for a specific task"""
        complexity = self.get_task_complexity(task_name)
        subtasks = complexity['subtasks']
        
        # Base calculation
        if complexity['cpu_intensive']:
            # CPU-intensive tasks can use more workers
            base_workers = min(subtasks, total_available_workers // 2)
        else:
            # I/O-intensive tasks benefit from more workers
            base_workers = min(subtasks, total_available_workers // 3)
        
        # Apply complexity multiplier
        complexity_multiplier = min(complexity['score'] / 3, 2.0)  # Cap at 2x
        optimal_workers = max(1, int(base_workers * complexity_multiplier))
        
        # Ensure we don't exceed available workers
        return min(optimal_workers, total_available_workers, subtasks)
    
    def create_subtasks(self, task_name: str, file_path: str, options: dict) -> list:
        """Break down a task into subtasks for parallel execution"""
        subtasks = []
        
        if task_name == 'social_formats':
            # Break down social formats into individual format tasks
            social_formats = options.get('social_formats', {})
            for platform, enabled in social_formats.items():
                if enabled:
                    subtasks.append({
                        'name': f'social_{platform}',
                        'func': self._create_single_social_format,
                        'args': (file_path, platform),
                        'priority': 1
                    })
        
        elif task_name == 'favicon':
            # Break down favicon into multiple size tasks
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            for size in sizes:
                subtasks.append({
                    'name': f'favicon_{size[0]}x{size[1]}',
                    'func': self._create_single_favicon_size,
                    'args': (file_path, size),
                    'priority': 1
                })
        
        elif task_name == 'color_separations':
            # Break down color separations into individual separation tasks
            subtasks.append({
                'name': 'color_sep_analysis',
                'func': self._analyze_colors,
                'args': (file_path,),
                'priority': 1
            })
            subtasks.append({
                'name': 'color_sep_pms',
                'func': self._create_pms_separations,
                'args': (file_path,),
                'priority': 2
            })
            subtasks.append({
                'name': 'color_sep_cmyk',
                'func': self._create_cmyk_separations,
                'args': (file_path,),
                'priority': 2
            })
        
        elif task_name == 'contour_cut':
            # Break down contour cut into processing steps
            subtasks.append({
                'name': 'contour_edge_detection',
                'func': self._detect_contour_edges,
                'args': (file_path,),
                'priority': 1
            })
            subtasks.append({
                'name': 'contour_mask_creation',
                'func': self._create_contour_masks,
                'args': (file_path,),
                'priority': 2
            })
            subtasks.append({
                'name': 'contour_svg_generation',
                'func': self._generate_contour_svg,
                'args': (file_path,),
                'priority': 3
            })
            subtasks.append({
                'name': 'contour_pdf_generation',
                'func': self._generate_contour_pdf,
                'args': (file_path,),
                'priority': 3
            })
        
        else:
            # For simple tasks, create a single subtask
            subtasks.append({
                'name': task_name,
                'func': getattr(self, f'_create_{task_name}'),
                'args': (file_path,),
                'priority': 1
            })
        
        return subtasks
    
    def execute_task_with_subtasks(self, task_name: str, file_path: str, options: dict, available_workers: int) -> dict:
        """Execute a task using subtask parallelization"""
        self.logger.info(f'🔄 Executing {task_name} with subtask parallelization')
        
        # Create subtasks
        subtasks = self.create_subtasks(task_name, file_path, options)
        self.logger.info(f'📋 Created {len(subtasks)} subtasks for {task_name}')
        
        # Calculate optimal workers for this task
        optimal_workers = self.calculate_optimal_workers_for_task(task_name, available_workers)
        actual_workers = min(optimal_workers, len(subtasks), available_workers)
        
        self.logger.info(f'⚙️ Using {actual_workers} workers for {len(subtasks)} subtasks')
        
        # Execute subtasks in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit subtasks
            future_to_subtask = {}
            for subtask in subtasks:
                future = executor.submit(subtask['func'], *subtask['args'])
                future_to_subtask[future] = subtask
            
            # Collect results
            for future in as_completed(future_to_subtask):
                subtask = future_to_subtask[future]
                try:
                    result = future.result()
                    results[subtask['name']] = result
                    self.logger.info(f'✅ Subtask {subtask["name"]} completed')
                except Exception as e:
                    self.logger.error(f'❌ Subtask {subtask["name"]} failed: {str(e)}')
                    results[subtask['name']] = None
        
        # Combine results based on task type
        return self.combine_subtask_results(task_name, results, options, file_path)
    
    def combine_subtask_results(self, task_name: str, subtask_results: dict, options: dict, file_path: str) -> dict:
        """Combine subtask results into final task output"""
        if task_name == 'social_formats':
            # Combine individual social format results
            combined = {}
            for key, value in subtask_results.items():
                if key.startswith('social_') and value:
                    platform = key.replace('social_', '')
                    combined[platform] = value
            return combined
        
        elif task_name == 'favicon':
            # Combine favicon size results
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            combined_sizes = []
            for size in sizes:
                key = f'favicon_{size[0]}x{size[1]}'
                if key in subtask_results and subtask_results[key]:
                    combined_sizes.append(subtask_results[key])
            
            if combined_sizes:
                # Create ICO file with all sizes
                return self._create_ico_from_sizes(combined_sizes, file_path)
            return {}
        
        elif task_name == 'color_separations':
            # Combine color separation results
            combined = {}
            if 'color_sep_pms' in subtask_results and subtask_results['color_sep_pms']:
                combined.update(subtask_results['color_sep_pms'])
            if 'color_sep_cmyk' in subtask_results and subtask_results['color_sep_cmyk']:
                combined.update(subtask_results['color_sep_cmyk'])
            return combined
        
        elif task_name == 'contour_cut':
            # Combine contour cut results
            combined = {}
            if 'contour_mask_creation' in subtask_results and subtask_results['contour_mask_creation']:
                combined.update(subtask_results['contour_mask_creation'])
            if 'contour_svg_generation' in subtask_results and subtask_results['contour_svg_generation']:
                combined['svg'] = subtask_results['contour_svg_generation']
            if 'contour_pdf_generation' in subtask_results and subtask_results['contour_pdf_generation']:
                combined['pdf'] = subtask_results['contour_pdf_generation']
            return combined
        
        else:
            # For simple tasks, return the single result
            for key, value in subtask_results.items():
                if value is not None:
                    return value
            return {}

    # ----------- Subtask Helper Methods -----------
    def _create_single_social_format(self, file_path, platform):
        """Create a single social media format"""
        base_dir, base_name = _get_base(file_path)
        img = Image.open(file_path).convert("RGBA")
        
        if platform in self.social_sizes:
            size = self.social_sizes[platform]
            out_path = os.path.join(base_dir, f"{base_name}_{platform}.png")
            resized = img.copy().resize(size, Image.LANCZOS)
            out_img = Image.new("RGBA", size, (255, 255, 255, 0))
            x = (size[0] - resized.size[0]) // 2
            y = (size[1] - resized.size[1]) // 2
            out_img.paste(resized, (x, y), resized)
            out_img.save(out_path)
            return out_path
        return None
    
    def _create_single_favicon_size(self, file_path, size):
        """Create a single favicon size"""
        img = Image.open(file_path).convert("RGBA")
        return img.resize(size, Image.LANCZOS)
    
    def _create_ico_from_sizes(self, size_images, file_path):
        """Create ICO file from multiple size images"""
        base_dir, base_name = _get_base(file_path)
        favicon_path = os.path.join(base_dir, f"{base_name}_favicon.ico")
        
        if size_images:
            size_images[0].save(favicon_path, format='ICO', sizes=[img.size for img in size_images])
            return {"ico": favicon_path}
        return {}
    
    def _analyze_colors(self, file_path):
        """Analyze colors for separation"""
        img = Image.open(file_path).convert("RGBA")
        arr = np.array(img)
        alpha_mask = arr[..., 3] > 0
        rgb_pixels = arr[alpha_mask][..., :3]
        
        if len(rgb_pixels) == 0:
            return {"error": "No visible pixels found"}
        
        if KMeans is None:
            raise ImportError("scikit-learn is required for color separations.")
        
        # Initial color detection
        initial_clusters = min(20, len(np.unique(rgb_pixels.reshape(-1, 3), axis=0)))
        kmeans_initial = KMeans(n_clusters=initial_clusters, random_state=42, n_init=3)
        kmeans_initial.fit(rgb_pixels)
        
        # Merge similar colors
        unique_colors = []
        color_threshold = 30
        
        for center in kmeans_initial.cluster_centers_:
            is_similar = False
            for existing_color in unique_colors:
                distance = np.sqrt(np.sum((center - existing_color) ** 2))
                if distance < color_threshold:
                    is_similar = True
                    break
            if not is_similar:
                unique_colors.append(center)
        
        return {"unique_colors": unique_colors, "num_colors": len(unique_colors)}
    
    def _create_pms_separations(self, file_path):
        """Create PMS spot color separations"""
        # This would contain the PMS separation logic from _create_color_separations
        # For now, return a placeholder
        return {"pms_separations": "placeholder"}
    
    def _create_cmyk_separations(self, file_path):
        """Create CMYK process separations"""
        # This would contain the CMYK separation logic from _create_color_separations
        # For now, return a placeholder
        return {"cmyk_separations": "placeholder"}
    
    def _detect_contour_edges(self, file_path):
        """Detect contour edges for cutline processing"""
        # This would contain the edge detection logic from _create_contour_cutline
        # For now, return a placeholder
        return {"edges": "placeholder"}
    
    def _create_contour_masks(self, file_path):
        """Create contour masks for cutline processing"""
        # This would contain the mask creation logic from _create_contour_cutline
        # For now, return a placeholder
        return {"masks": "placeholder"}
    
    def _generate_contour_svg(self, file_path):
        """Generate SVG for contour cutline"""
        # This would contain the SVG generation logic from _create_contour_cutline
        # For now, return a placeholder
        return {"svg": "placeholder"}
    
    def _generate_contour_pdf(self, file_path):
        """Generate PDF for contour cutline"""
        # This would contain the PDF generation logic from _create_contour_cutline
        # For now, return a placeholder
        return {"pdf": "placeholder"}

    def create_test_svg_for_repurposing(self, output_path: str) -> bool:
        """Create a test SVG file with multiple design elements for testing repurposing"""
        try:
            svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect x="0" y="0" width="800" height="600" fill="#f0f0f0"/>
    
    <!-- Logo (top-left) -->
    <rect x="50" y="50" width="100" height="80" fill="#2c3caf" rx="10"/>
    <text x="100" y="95" text-anchor="middle" fill="white" font-size="14" font-weight="bold">LOGO</text>
    
    <!-- Main Title -->
    <text x="400" y="150" text-anchor="middle" fill="#2c3caf" font-size="48" font-weight="bold">Smart Design</text>
    
    <!-- Subtitle -->
    <text x="400" y="200" text-anchor="middle" fill="#666" font-size="24">Intelligent Content Repurposing</text>
    
    <!-- Body Text -->
    <text x="400" y="280" text-anchor="middle" fill="#333" font-size="16">Transform your designs automatically for any social media platform</text>
    
    <!-- Decorative Shape -->
    <circle cx="200" cy="400" r="60" fill="#FAA52D" opacity="0.8"/>
    <circle cx="600" cy="400" r="60" fill="#ff6b35" opacity="0.8"/>
    
    <!-- Call to Action -->
    <rect x="300" y="450" width="200" height="60" fill="#28a745" rx="10"/>
    <text x="400" y="485" text-anchor="middle" fill="white" font-size="18" font-weight="bold">Get Started</text>
</svg>'''
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            self.logger.info(f'✅ Test SVG created: {output_path}')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Error creating test SVG: {e}')
            return False
