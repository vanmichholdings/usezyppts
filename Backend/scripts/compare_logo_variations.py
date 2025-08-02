import os
import sys
from pathlib import Path

# Add Backend directory to Python path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from utils.logo_processor import LogoProcessor
from base64 import b64encode

# Get project root directory (one level up from Backend/)
project_root = os.path.dirname(backend_dir)

def find_logo_file():
    """Find an available logo file in the uploads directory"""
    uploads_dir = os.path.join(project_root, 'Backend/uploads')
    
    # Look for PNG files in upload directories
    for item in os.listdir(uploads_dir):
        item_path = os.path.join(uploads_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            uploads_subdir = os.path.join(item_path, 'uploads')
            if os.path.exists(uploads_subdir):
                for file in os.listdir(uploads_subdir):
                    if file.lower().endswith('.png'):
                        return os.path.join(uploads_subdir, file)
    
    # If no PNG found, look for any image file
    for item in os.listdir(uploads_dir):
        item_path = os.path.join(uploads_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            uploads_subdir = os.path.join(item_path, 'uploads')
            if os.path.exists(uploads_subdir):
                for file in os.listdir(uploads_subdir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        return os.path.join(uploads_subdir, file)
    
    return None

# Find an available logo file
logo_path = find_logo_file()
if not logo_path:
    print("No logo file found in uploads directory. Please upload a logo first.")
    sys.exit(1)

print(f"Using logo file: {logo_path}")

proc = LogoProcessor()
base_dir = os.path.dirname(logo_path)
base_name = os.path.splitext(os.path.basename(logo_path))[0]

# Run all variations
variations = {
    'Original': {'png': logo_path},
    'Transparent PNG': proc._create_transparent_png(logo_path),
    'Black Version': proc._create_black_version(logo_path),
    'PDF Version': proc._create_pdf_version(logo_path),
    'WebP Version': proc._create_webp_version(logo_path),
    'Favicon': proc._create_favicon(logo_path),
    'Email Header': proc._create_email_header(logo_path),
    'Vector Trace': proc._create_vector_trace(logo_path),
    'Full Color Vector Trace': proc._create_full_color_vector_trace(logo_path),
    'Color Separations': proc._create_color_separations(logo_path),
    'Distressed': proc._create_distressed_version(logo_path),
    'Contour Cutline': proc._create_contour_cutline(logo_path),
}

# Social formats (grouped)
social = proc._create_social_formats(logo_path)

# Helper to embed or link
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
SVG_EXTS = {'.svg'}

def file_preview_tag(path):
    ext = Path(path).suffix.lower()
    if ext in IMG_EXTS and os.path.exists(path):
        with open(path, 'rb') as f:
            data = b64encode(f.read()).decode()
        return f'<img src="data:image/{ext[1:]};base64,{data}" alt="preview">'
    elif ext in SVG_EXTS and os.path.exists(path):
        with open(path, 'r') as f:
            svg = f.read()
        return f'<div style="background:#fff;border:1px solid #ccc;max-width:280px;max-height:180px;overflow:auto">{svg}</div>'
    else:
        return '<span style="color:#888">(no preview)</span>'

def file_link_tag(path, label=None):
    if not os.path.exists(path):
        return ''
    label = label or os.path.basename(path)
    return f'<a href="{path}" download>{label}</a>'

# Build HTML - Update paths to Frontend/templates
html_template_path = os.path.join(project_root, 'Frontend/templates/compare_logo_variations.html')
html_output_path = os.path.join(project_root, 'Frontend/templates/compare_logo_variations.html')

with open(html_template_path, 'r') as f:
    html = f.read()

inserts = []
for name, files in variations.items():
    if isinstance(files, dict):
        # Special handling for color separations
        if name == 'Color Separations' and 'pngs' in files:
            # Color separations returns a list of tuples (path, label)
            preview_html = '<div style="margin-bottom:10px"><b>Color Separations</b><br>'
            for png_path, label in files['pngs']:
                if os.path.exists(png_path):
                    preview_html += f'<div style="margin-bottom:10px"><b>{label}</b><br>{file_preview_tag(png_path)}<br>{file_link_tag(png_path)}</div>'
            preview_html += '</div>'
            inserts.append(f'<div class="variation"><h2>{name}</h2>{preview_html}</div>')
        else:
            # Regular dictionary handling
            preview = None
            for ext in ['png', 'svg', 'webp', 'jpg', 'jpeg']:
                if ext in files and os.path.exists(files[ext]):
                    preview = file_preview_tag(files[ext])
                    break
            if not preview:
                # Try first file
                for v in files.values():
                    if os.path.exists(v):
                        preview = file_preview_tag(v)
                        break
            links = ' '.join([file_link_tag(v) for v in files.values() if os.path.exists(v)])
            inserts.append(f'<div class="variation"><h2>{name}</h2>{preview}<div class="links">{links}</div></div>')
    else:
        preview = file_preview_tag(files)
        links = file_link_tag(files)
        inserts.append(f'<div class="variation"><h2>{name}</h2>{preview}<div class="links">{links}</div></div>')

# Social formats (grouped)
social_html = '<div class="variation"><h2>Social Formats</h2>'
for sname, spath in social.items():
    if os.path.exists(spath):
        social_html += f'<div style="margin-bottom:10px"><b>{sname.replace("_", " ").title()}</b><br>{file_preview_tag(spath)}<br>{file_link_tag(spath)}</div>'
social_html += '</div>'
inserts.append(social_html)

html = html.replace('<!-- Variations will be inserted here by the Python script -->', '\n'.join(inserts))

with open(html_output_path, 'w') as f:
    f.write(html)

print(f"Comparison HTML generated: {html_output_path}") 