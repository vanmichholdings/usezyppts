    def separate_colors(self, img: Image.Image) -> Dict[str, Image.Image]:
        """
        Enhanced color separation with improved quality and precision
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # Initialize color channels
        channels = {}
        width, height = img.size
        
        # Convert to LAB color space for better color accuracy
        lab_img = ImageCms.createProfile('LAB')
        rgb_img = ImageCms.createProfile('sRGB')
        transform = ImageCms.buildTransformFromOpenProfiles(rgb_img, lab_img, 'RGBA', 'LAB')
        lab_image = ImageCms.applyTransform(img, transform)
        
        # Extract individual channels with enhanced precision
        l_channel, a_channel, b_channel, alpha = lab_image.split()
        
        # Process CMYK separations with improved quality
        cmyk_transform = ImageCms.buildTransformFromOpenProfiles(
            rgb_img, 
            ImageCms.createProfile('CMYK'),
            'RGBA', 'CMYK'
        )
        cmyk_image = ImageCms.applyTransform(img, cmyk_transform)
        c, m, y, k = cmyk_image.split()
        
        # Apply advanced color correction and enhancement
        channels['cyan'] = ImageEnhance.Contrast(c).enhance(1.1)
        channels['magenta'] = ImageEnhance.Contrast(m).enhance(1.1)
        channels['yellow'] = ImageEnhance.Contrast(y).enhance(1.1)
        channels['black'] = ImageEnhance.Contrast(k).enhance(1.2)
        
        # Create spot color separations with improved accuracy
        for color_name, color_value in self.spot_colors.items():
            mask = self._create_precise_color_mask(img, color_value)
            channels[color_name] = mask
            
        # Add alpha channel information
        channels['alpha'] = alpha
        
        return channels
        
    def _create_precise_color_mask(self, img: Image.Image, target_color: Tuple[int, ...], 
                                 tolerance: float = 0.1) -> Image.Image:
        """
        Creates a precise color mask with improved edge handling and color matching
        """
        img_array = np.array(img)
        target_lab = rgb2lab(np.array([[target_color[:3]]], dtype=np.uint8)/255.0)[0][0]
        
        # Convert image to LAB color space for better matching
        img_lab = rgb2lab(img_array[:, :, :3]/255.0)
        
        # Calculate color differences using Delta E 2000
        diff = deltaE_ciede2000(img_lab, target_lab)
        
        # Create mask with smooth transitions
        mask = np.exp(-diff / (tolerance * 100))
        mask = np.clip(mask, 0, 1)
        
        # Apply edge enhancement
        mask = gaussian_filter(mask, sigma=0.5)
        
        # Convert to 8-bit image
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        return mask_img