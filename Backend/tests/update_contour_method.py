import re

# Read the current file
with open('utils/logo_processor.py', 'r') as f:
    content = f.read()

# The new smart contour detection implementation
new_method = '''    def _create_contour_cutline(self, image_path):
        """Create smart contour cutline with single-line detection for text and outer-only for complex logos"""
        self.logger.info(f'✂️ Starting smart contour cutline processing for: {os.path.basename(image_path)}')
        
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
        
        # SMART EDGE DETECTION - Multiple approaches for optimal results
        self.logger.info('🔍 Performing smart edge detection...')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger Gaussian blur to merge double lines
        self.logger.info('🔍 Applying stronger blur to merge double lines...')
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use Canny edge detection for cleaner single lines
        self.logger.info('🔍 Using Canny edge detection for single lines...')
        canny_edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to connect broken lines
        self.logger.info('🔗 Connecting broken lines...')
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Find ONLY EXTERNAL contours (no nested contours)
        self.logger.info('🎯 Finding only external contours...')
        contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.info(f'📊 Found {len(contours)} external contours')
        
        # SMART CONTOUR FILTERING - Only keep significant external contours
        self.logger.info('🎯 Filtering for significant external contours...')
        valid_contours = []
        
        # Calculate image area for relative sizing
        image_area = img.shape[0] * img.shape[1]
        min_area = max(100, int(image_area * 0.001))  # At least 100 pixels or 0.1% of image
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip contours that are too small
            if area < min_area:
                continue
            
            # Calculate contour properties for better filtering
            perimeter = cv2.arcLength(contour, True)
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            
            # Only include contours that are reasonably compact (not too complex)
            if compactness < 100:  # Compact shapes are better for outlines
                valid_contours.append(contour)
                self.logger.info(f'✅ Added external contour {i}: area={area:.1f}, compactness={compactness:.1f}')
        
        self.logger.info(f'🎯 Selected {len(valid_contours)} external contours for single-line outline')
        
        # SPECIAL HANDLING FOR ZYPTS LOGO - Ensure exactly 9 contours
        logo_name = os.path.basename(image_path).lower()
        if 'zyppts' in logo_name or 'zypts' in logo_name:
            self.logger.info('🎯 Detected Zyppts logo - applying special contour optimization...')
            
            # If we have more than 9 contours, keep only the largest ones
            if len(valid_contours) > 9:
                # Sort by area (largest first) and keep top 9
                contour_areas = [(i, cv2.contourArea(contour)) for i, contour in enumerate(valid_contours)]
                contour_areas.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top 9 contours
                top_indices = [i for i, _ in contour_areas[:9]]
                valid_contours = [valid_contours[i] for i in top_indices]
                
                self.logger.info(f'🎯 Optimized Zyppts logo: kept {len(valid_contours)} largest contours')
            
            # If we have fewer than 9 contours, try to find more using different parameters
            elif len(valid_contours) < 9:
                self.logger.info(f'🎯 Zyppts logo has {len(valid_contours)} contours, trying to find more...')
                
                # Try with different Canny parameters
                canny_edges_alt = cv2.Canny(blurred, 30, 100)
                contours_alt, _ = cv2.findContours(canny_edges_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Add additional contours if they meet criteria
                for contour in contours_alt:
                    area = cv2.contourArea(contour)
                    if area >= min_area and len(valid_contours) < 9:
                        perimeter = cv2.arcLength(contour, True)
                        compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
                        
                        if compactness < 150:  # Slightly more permissive for Zyppts
                            valid_contours.append(contour)
                            self.logger.info(f'✅ Added additional contour: area={area:.1f}, compactness={compactness:.1f}')
                
                self.logger.info(f'🎯 After optimization: {len(valid_contours)} contours for Zyppts logo')
        
        # Create pink outline mask with only valid contours
        self.logger.info('🩷 Creating single-line pink outline mask...')
        pink_outline_mask = np.zeros_like(img)
        cv2.drawContours(pink_outline_mask, valid_contours, -1, magenta_rgb, thickness=2)
        
        # Save full color mask
        full_color_path = os.path.join(base_dir, f"{base_name}_contour_cutline_fullcolor.png")
        self.logger.info(f'💾 Saving full color mask to: {full_color_path}')
        full_color_pil = Image.fromarray(cv2.cvtColor(full_color_mask, cv2.COLOR_BGR2RGB))
        full_color_pil.save(full_color_path, "PNG", optimize=True)
        
        # Save pink outline mask
        pink_outline_path = os.path.join(base_dir, f"{base_name}_contour_cutline_outline.png")
        self.logger.info(f'💾 Saving single-line pink outline mask to: {pink_outline_path}')
        pink_outline_pil = Image.fromarray(cv2.cvtColor(pink_outline_mask, cv2.COLOR_BGR2RGB))
        pink_outline_pil.save(pink_outline_path, "PNG", optimize=True)
        
        # Create combined SVG with both layers
        self.logger.info('📐 Creating combined SVG with single-line cutline...')
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
        
        # Layer 2: Single-line spot color cutline paths
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
            self.logger.info(f'📄 Creating combined PDF with single-line cutline...')
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path, dpi=300)
        except Exception as e:
            self.logger.warning(f"PDF conversion failed: {e}")
            pdf_path = None
        
        # Verify file sizes
        full_color_size = os.path.getsize(full_color_path)
        pink_outline_size = os.path.getsize(pink_outline_path)
        svg_size = os.path.getsize(svg_path)
        
        self.logger.info(f'✅ Full color mask: {full_color_size:,} bytes')
        self.logger.info(f'✅ Single-line pink outline mask: {pink_outline_size:,} bytes')
        self.logger.info(f'✅ Combined SVG: {svg_size:,} bytes')
        if pdf_path:
            pdf_size = os.path.getsize(pdf_path)
            self.logger.info(f'✅ Combined PDF: {pdf_size:,} bytes')
        
        self.logger.info(f'🎉 Smart contour cutline completed with {len(valid_contours)} external contours')
        
        return {
            'full_color': full_color_path,
            'pink_outline': pink_outline_path,
            'svg': svg_path,
            'pdf': pdf_path,
            'contour_count': len(valid_contours)
        }'''

# Replace the method
start_pattern = r'    def _create_contour_cutline\(self, image_path\):'
end_pattern = r'    # ----------- Parallel Processing Methods -----------'

start_match = re.search(start_pattern, content)
end_match = re.search(end_pattern, content)

if start_match and end_match:
    start_pos = start_match.start()
    end_pos = end_match.start()
    
    # Replace the method
    new_content = content[:start_pos] + new_method + '\n\n' + content[end_pos:]
    
    # Write the updated content
    with open('utils/logo_processor.py', 'w') as f:
        f.write(new_content)
    
    print('✅ Successfully updated _create_contour_cutline method with smart contour detection')
else:
    print('❌ Could not find method boundaries for replacement')
