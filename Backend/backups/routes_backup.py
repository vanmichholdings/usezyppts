from flask import Blueprint, render_template, redirect, url_for, flash, request, send_file, jsonify, current_app, Response, stream_with_context
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime, timedelta
from app_config import db, mail
from models import User, Subscription
from config import Config
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops, ImageEnhance
import os
import shutil
import zipfile
import uuid
import base64
from io import BytesIO
import time
import math
from werkzeug.utils import secure_filename
from utils.logo_processor import LogoProcessor
import logging
from flask_mail import Message
from flask import after_this_request
from werkzeug.exceptions import RequestEntityTooLarge
import hashlib
import json
import threading
from collections import defaultdict
import requests
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('main', __name__)

# Store progress for each session
progress_data = defaultdict(dict)
progress_lock = threading.Lock()

def update_progress(session_id, stage, progress, message, current_file=None, total_files=0, processed_files=0):
    """Update progress for a session"""
    with progress_lock:
        progress_data[session_id] = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'current_file': current_file,
            'total_files': total_files,
            'processed_files': processed_files,
            'timestamp': time.time()
        }

def get_progress(session_id):
    """Get current progress for a session"""
    return progress_data.get(session_id, {
        'stage': 'unknown',
        'progress': 0,
        'message': 'Processing not started',
        'current_file': None,
        'total_files': 0,
        'processed_files': 0,
        'timestamp': time.time()
    })

def cleanup_old_sessions():
    """Remove old session data to prevent memory leaks"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, data in progress_data.items()
        if current_time - data.get('timestamp', 0) > 3600  # 1 hour expiration
    ]
    
    with progress_lock:
        for session_id in expired_sessions:
            progress_data.pop(session_id, None)

# Use configuration paths
UPLOAD_FOLDER = Config.UPLOAD_FOLDER
OUTPUT_FOLDER = Config.OUTPUT_FOLDER
CACHE_FOLDER = Config.CACHE_FOLDER
TEMP_FOLDER = Config.TEMP_FOLDER

# Allowed extensions (consider moving to config or utils)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'svg', 'pdf', 'webp'}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_file_upload():
    """Handle file upload with proper error handling"""
    if 'logo' not in request.files:
        raise ValueError('No file uploaded')
    
    file = request.files['logo']
    if file.filename == '':
        raise ValueError('No selected file')
    
    if not allowed_file(file.filename):
         raise ValueError('Invalid file type. Supported: PNG, JPG, GIF, SVG, PDF, WEBP')
    
    return file

def ensure_upload_dirs(upload_id):
    """Ensure all required directories exist with proper permissions"""
    base_upload_folder = current_app.config['UPLOAD_FOLDER']
    user_upload_folder = os.path.join(base_upload_folder, upload_id)
    
    # Define required subdirectories
    dirs = {
        'upload': os.path.join(user_upload_folder, 'uploads'),
        'output': os.path.join(user_upload_folder, 'outputs'),
        'cache': os.path.join(user_upload_folder, 'cache'), 
        'temp': os.path.join(user_upload_folder, 'temp')
    }
    
    # Create directories if they don't exist
    for path in dirs.values():
        try:
            os.makedirs(path, exist_ok=True)
            os.chmod(path, 0o750) # rwxr-x---
        except OSError as e:
            logger.error(f"Error creating directory {path}: {e}")
            raise  # Re-raise the error
            
    return dirs

def cleanup_dirs(dirs):
    """Clean up temporary directories"""
    if dirs and dirs.get('upload'):
        base_dir = os.path.dirname(dirs['upload']) # Get the parent upload_id folder
        if base_dir and os.path.exists(base_dir):
            try:
                shutil.rmtree(base_dir)
                logger.info(f"Cleaned up temporary directory: {base_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up directory {base_dir}: {e}")

@bp.route('/')
def home():
    return render_template('home.html', now=datetime.now())

@bp.route('/progress/<session_id>')
def progress(session_id):
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        try:
            # Send initial data
            data = get_progress(session_id)
            yield f"data: {json.dumps(data)}\n\n"
            
            # Keep connection open and send updates
            last_update = time.time()
            while True:
                data = get_progress(session_id)
                
                # If processing is complete or there's an error, close the connection
                if data['stage'] in ['complete', 'error'] or (time.time() - data['timestamp']) > 300:  # 5 minutes timeout
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                    
                # Only send updates if something changed
                if data['timestamp'] > last_update:
                    yield f"data: {json.dumps(data)}\n\n"
                    last_update = time.time()
                
                time.sleep(0.5)  # Reduce CPU usage
                
        except GeneratorExit:
            # Client disconnected
            pass
        except Exception as e:
            logger.error(f"Error in progress stream: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable buffering in nginx
        }
    )

@bp.route('/logo_processor', methods=['GET', 'POST'])
@login_required
def logo_processor():
    if request.method == 'GET':
        return render_template('logo_processor.html', now=datetime.now())
    
    # Get session ID for progress tracking
    session_id = request.form.get('session_id')
    if not session_id and 'session_id' in request.args:
        session_id = request.args.get('session_id')
    
    if not current_user.has_active_subscription():
        update_progress(session_id, 'error', 0, 'Active subscription required')
        return jsonify({'error': 'Active subscription required'}), 403
        
    if not current_user.can_generate_files():
        update_progress(session_id, 'error', 0, 'No credits remaining')
        return jsonify({'error': 'No credits remaining'}), 403
    
    dirs = None
    file_path = None
    upload_id = None
    
    try:
        # Check if this is a preview request
        if 'preview' in request.form:
            effect = request.form.get('effect')
            if not effect:
                update_progress(session_id, 'error', 0, 'No effect specified')
                return jsonify({'error': 'No effect specified'}), 400
                
            file = request.files.get('file')
            if not file:
                update_progress(session_id, 'error', 0, 'No file uploaded')
                return jsonify({'error': 'No file uploaded'}), 400

            # Process the preview based on effect type
            processor = LogoProcessor()
            preview_path = None
            
            # Update progress
            update_progress(session_id, 'processing', 50, f'Generating {effect.replace("_", " ")} preview...', 
                          current_file=file.filename, total_files=1, processed_files=0)
            
            # Call appropriate preview method based on effect
            try:
                img = Image.open(file)
                if effect == 'effect_vector':
                    preview_path = processor._apply_vector_trace(img)
                elif effect == 'effect_color_separations':
                    preview_path = processor._create_color_separations(img)
                elif effect == 'effect_distressed':
                    preview_path = processor._create_distressed_version(img)
                elif effect == 'effect_outline':
                    preview_path = processor._create_contour_cutline(img)
                
                update_progress(session_id, 'complete', 100, 'Preview generated successfully', 
                              current_file=file.filename, total_files=1, processed_files=1)
            except Exception as e:
                update_progress(session_id, 'error', 0, f'Error generating preview: {str(e)}')
                raise

            if preview_path:
                # Convert preview to base64 for response
                with open(preview_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()
                return jsonify({
                    'preview': f'data:image/png;base64,{encoded}'
                })
            return jsonify({'error': 'Failed to generate preview'}), 500

        # Handle full processing request
        try:
            # Update progress - starting
            update_progress(session_id, 'initializing', 5, 'Preparing to process files...')
            
            # Create directories for this upload
            upload_id = str(uuid.uuid4())
            dirs = ensure_upload_dirs(upload_id)
            
            # Save uploaded file
            file = request.files['logo']
            if file.filename == '':
                update_progress(session_id, 'error', 0, 'No selected file')
                return jsonify({'error': 'No selected file'}), 400
                
            # Validate file type before processing
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.pdf']:
                update_progress(session_id, 'error', 0, f'Unsupported file type: {file_ext}')
                return jsonify({'error': f'Unsupported file type: {file_ext}. Please upload a supported image or PDF file.'}), 400
                
            if not allowed_file(file.filename):
                update_progress(session_id, 'error', 0, 'File type not allowed')
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(dirs['upload'], filename)
            
            # Update progress - saving file
            update_progress(session_id, 'processing', 10, 'Saving uploaded file...', 
                          current_file=filename, total_files=1, processed_files=0)
            
            file.save(file_path)
            
            # Initialize processor with the uploaded file
            processor = LogoProcessor(
                cache_folder=dirs['cache'],
                upload_folder=dirs['upload'],
                output_folder=dirs['output'],
                temp_folder=dirs['temp']
            )
            
            # Get processing options from form - map to the expected option keys in process_logo
            # Debug: Log all form fields to see what's being sent
            logger.info(f"Form fields received: {list(request.form.keys())}")
            logger.info(f"Form data: {dict(request.form)}")
            
            options = {
                # Basic formats
                'transparent_png': 'transparent_png' in request.form,
                'black_version': 'black_version' in request.form,  # One Color Black Monochrome
                'pdf_version': 'pdf_version' in request.form,
                'webp_version': 'webp_version' in request.form,
                'favicon': 'favicon' in request.form,
                'email_header': 'email_header' in request.form,
                
                # Effects - with multiple possible field names for compatibility
                'vector_trace': ('effect_vector' in request.form or 
                               'vector_trace' in request.form or 
                               'vector' in request.form),
                'full_color_vector_trace': ('effect_full_color_vector' in request.form or 
                                          'full_color_vector_trace' in request.form or 
                                          'full_color_vector' in request.form or
                                          'fullcolor_vector' in request.form),
                'color_separations': ('effect_color_separations' in request.form or 
                                    'color_separations' in request.form),
                'distressed_effect': ('effect_distressed' in request.form or 
                                    'distressed_effect' in request.form),
                'contour_cut': ('effect_outline' in request.form or 
                              'contour_cut' in request.form or 
                              'outline' in request.form),  # Contour Cutline
                
                # Social media formats (handled separately)
                'social_formats': {
                    # Instagram
                    'instagram_profile': 'social_instagram_profile' in request.form,
                    'instagram_post': 'social_instagram_post' in request.form,
                    'instagram_story': 'social_instagram_story' in request.form,
                    
                    # Facebook
                    'facebook_profile': 'social_facebook_profile' in request.form,
                    'facebook_post': 'social_facebook_post' in request.form,
                    'facebook_cover': 'social_facebook_cover' in request.form,
                    
                    # Twitter/X
                    'twitter_profile': 'social_twitter_profile' in request.form,
                    'twitter_post': 'social_twitter_post' in request.form,
                    'twitter_header': 'social_twitter_header' in request.form,
                    
                    # YouTube
                    'youtube_profile': 'social_youtube_profile' in request.form,
                    'youtube_cover': 'social_youtube_cover' in request.form,
                    'youtube_thumbnail': 'social_youtube_thumbnail' in request.form,
                    
                    # LinkedIn
                    'linkedin_profile': 'social_linkedin_profile' in request.form,
                    'linkedin_cover': 'social_linkedin_cover' in request.form,
                    'linkedin_post': 'social_linkedin_post' in request.form,
                    
                    # Pinterest
                    'pinterest_pin': 'social_pinterest_pin' in request.form,
                    'pinterest_profile': 'social_pinterest_profile' in request.form,
                    'pinterest_board': 'social_pinterest_board' in request.form,
                    
                    # TikTok
                    'tiktok_profile': 'social_tiktok_profile' in request.form,
                    'tiktok_video': 'social_tiktok_video' in request.form
                }
            }
            
            # Debug: Log the processed options
            logger.info(f"Processed options: {options}")
            logger.info(f"Vector trace enabled: {options.get('vector_trace')}")
            logger.info(f"Full color vector trace enabled: {options.get('full_color_vector_trace')}")
            
            # Update progress - starting processing
            update_progress(session_id, 'processing', 20, 'Processing logo with selected effects...', 
                          current_file=filename, total_files=1, processed_files=0)
            
            # Process the logo with the selected options
            result = processor.process_logo(file_path=file_path, options=options)
            
            # Verify the processing was successful
            if not result.get('success'):
                error_msg = result.get('message', 'Unknown error during processing')
                logger.error(f"Logo processing failed: {error_msg}")
                update_progress(session_id, 'error', 0, f"Processing failed: {error_msg}")
                return jsonify({'error': error_msg}), 500
            
            # Get the output files from the result
            output_files = []
            outputs = result.get('outputs', {})
            
            # Get the original filename without extension
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Helper function to safely add files to the output list with proper naming
            def add_file(file_path, dest_path, use_original_name=True):
                if not file_path or not os.path.exists(file_path):
                    return False
                output_files.append((file_path, dest_path))
                return True
            
            summary_lines = []
            
            # Add the original file only if requested and it exists
            if options.get('include_original', False) and os.path.exists(file_path):
                arcname = f"Original/{original_name}{os.path.splitext(file_path)[1]}"
                add_file(file_path, arcname)
                summary_lines.append(f"Original: {arcname}")

            # Formats
            if options.get('transparent_png', False) and 'transparent_png' in outputs:
                path = outputs['transparent_png']
                arcname = f"Formats/{original_name}_Transparent.png"
                add_file(path, arcname)
                summary_lines.append(f"Transparent PNG: {arcname}")
            if options.get('black_version', False) and 'black_version' in outputs:
                path = outputs['black_version']
                arcname = f"Formats/{original_name}_BlackVersion.png"
                add_file(path, arcname)
                summary_lines.append(f"Black Version: {arcname}")
            if options.get('pdf_version', False) and 'pdf_version' in outputs:
                path = outputs['pdf_version']
                arcname = f"Formats/{original_name}.pdf"
                add_file(path, arcname)
                summary_lines.append(f"PDF (Raster): {arcname}")
            if options.get('webp_version', False) and 'webp_version' in outputs:
                path = outputs['webp_version']
                arcname = f"Formats/{original_name}.webp"
                add_file(path, arcname)
                summary_lines.append(f"WebP: {arcname}")
            if options.get('favicon', False) and 'favicon' in outputs:
                path = outputs['favicon']
                arcname = f"Formats/{original_name}_Favicon.ico"
                add_file(path, arcname)
                summary_lines.append(f"Favicon: {arcname}")
            if options.get('email_header', False) and 'email_header' in outputs:
                path = outputs['email_header']
                arcname = f"Formats/{original_name}_EmailHeader.png"
                add_file(path, arcname)
                summary_lines.append(f"Email Header: {arcname}")

            # Effects
            if options.get('distressed_effect', False) and 'distressed_effect' in outputs:
                path = outputs['distressed_effect']
                arcname = f"Effects/{original_name}_Distressed.png"
                add_file(path, arcname)
                summary_lines.append(f"Distressed Effect: {arcname}")
            if options.get('contour_cut', False) and 'contour_cut' in outputs:
                contour_files = outputs['contour_cut']
                if isinstance(contour_files, dict):
                    if contour_files.get('pdf_vector'):
                        arcname = f"Effects/{original_name}_Contour_Vector.pdf"
                        add_file(contour_files['pdf_vector'], arcname)
                        summary_lines.append(f"Contour Cutline (Vector PDF): {arcname}")
                    if contour_files.get('svg'):
                        arcname = f"Effects/{original_name}_Contour.svg"
                        add_file(contour_files['svg'], arcname)
                        summary_lines.append(f"Contour Cutline (SVG): {arcname}")
                    if contour_files.get('pdf_preview'):
                        arcname = f"Effects/{original_name}_Contour_Preview.pdf"
                        add_file(contour_files['pdf_preview'], arcname)
                        summary_lines.append(f"Contour Cutline (Preview PDF): {arcname}")
                    if contour_files.get('png'):
                        arcname = f"Effects/{original_name}_Contour_Preview.png"
                        add_file(contour_files['png'], arcname)
                        summary_lines.append(f"Contour Cutline (Preview PNG): {arcname}")
            if options.get('vector_trace', False):
                if outputs.get('vector_trace_svg'):
                    arcname = f"Effects/{original_name}_VectorTrace.svg"
                    add_file(outputs['vector_trace_svg'], arcname)
                    summary_lines.append(f"Vector Trace (SVG): {arcname}")
                if outputs.get('vector_trace_pdf'):
                    arcname = f"Effects/{original_name}_VectorTrace.pdf"
                    add_file(outputs['vector_trace_pdf'], arcname)
                    summary_lines.append(f"Vector Trace (PDF): {arcname}")
                if outputs.get('vector_trace_ai'):
                    arcname = f"Effects/{original_name}_VectorTrace.ai"
                    add_file(outputs['vector_trace_ai'], arcname)
                    summary_lines.append(f"Vector Trace (AI): {arcname}")

            # Full Color Vector Trace
            if options.get('full_color_vector_trace', False):
                if outputs.get('full_color_vector_trace_svg'):
                    arcname = f"Effects/{original_name}_FullColorVector.svg"
                    add_file(outputs['full_color_vector_trace_svg'], arcname)
                    summary_lines.append(f"Full Color Vector Trace (SVG): {arcname}")
                if outputs.get('full_color_vector_trace_pdf'):
                    arcname = f"Effects/{original_name}_FullColorVector.pdf"
                    add_file(outputs['full_color_vector_trace_pdf'], arcname)
                    summary_lines.append(f"Full Color Vector Trace (PDF): {arcname}")
                if outputs.get('full_color_vector_trace_ai'):
                    arcname = f"Effects/{original_name}_FullColorVector.ai"
                    add_file(outputs['full_color_vector_trace_ai'], arcname)
                    summary_lines.append(f"Full Color Vector Trace (AI): {arcname}")
                # Add color count info to readme
                if outputs.get('full_color_vector_trace_info'):
                    colors_used = outputs['full_color_vector_trace_info'].get('colors_used', 0)
                    summary_lines.append(f"Full Color Vector Trace (Colors Used: {colors_used})")

            # Color Separations
            if options.get('color_separations', False) and 'color_separations' in outputs:
                color_seps = outputs['color_separations']
                if isinstance(color_seps, dict):
                    # Add all PNGs with PMS/CMYK labels
                    pngs = color_seps.get('pngs') or color_seps.get('png') or []
                    if isinstance(pngs, tuple):
                        pngs = [pngs]
                    for i, png_info in enumerate(pngs):
                        if isinstance(png_info, tuple):
                            png_path, label = png_info
                        else:
                            png_path, label = png_info, f"ColorSep{i+1}"
                        if isinstance(png_path, str) and os.path.exists(png_path):
                            ext = os.path.splitext(png_path)[1].lstrip('.') or 'png'
                            safe_label = label.replace(' ', '').replace('/', '').replace('(', '').replace(')', '')
                            arcname = f"Color Separations/{original_name}_{safe_label}.{ext}"
                            add_file(png_path, arcname, use_original_name=False)
                            summary_lines.append(f"Color Separation ({label}): {arcname}")
                    # Add PSD if present
                    psd_path = color_seps.get('psd')
                    if psd_path and os.path.exists(psd_path):
                        arcname = f"Color Separations/{original_name}_ColorSeps.psd"
                        add_file(psd_path, arcname, use_original_name=False)
                        summary_lines.append(f"Color Separations (PSD): {arcname}")

            # Social Media
            social_formats = options.get('social_formats', {})
            if any(social_formats.values()) and 'social_media' in outputs:
                social_media = outputs['social_media']
                if isinstance(social_media, dict):
                    for platform, formats in social_media.items():
                        if isinstance(formats, dict):
                            for format_name, format_path in formats.items():
                                if format_path and os.path.exists(format_path):
                                    ext = os.path.splitext(format_path)[1].lstrip('.') or 'png'
                                    arcname = f"Social Media/{platform}/{original_name}_{format_name}.{ext}"
                                    add_file(format_path, arcname, use_original_name=False)
                                    summary_lines.append(f"Social Media ({platform} - {format_name}): {arcname}")
                        elif isinstance(formats, str) and os.path.exists(formats):
                            ext = os.path.splitext(formats)[1].lstrip('.') or 'png'
                            arcname = f"Social Media/{platform}/{original_name}.{ext}"
                            add_file(formats, arcname, use_original_name=False)
                            summary_lines.append(f"Social Media ({platform}): {arcname}")
            
            total_files = len(output_files)

            # Add a readme.txt summary
            readme_path = os.path.join(dirs['temp'], 'readme.txt')
            with open(readme_path, 'w') as f:
                f.write("Zyppts Logo Processor Output\n\n")
                f.write(f"Processed file: {original_name}\n\n")
                f.write("Included files:\n")
                for line in summary_lines:
                    f.write(f"- {line}\n")
                f.write("\nThank you for using Zyppts!\n")
            add_file(readme_path, 'readme.txt', use_original_name=False)
            
            # If no output files were generated, log an error
            if total_files == 0:
                error_msg = "No output files were generated during processing"
                logger.error(error_msg)
                update_progress(session_id, 'error', 0, error_msg)
                return jsonify({'error': error_msg}), 500
            
            # Update progress - creating zip file with more detailed progress
            update_progress(session_id, 'zipping', 80, 'Preparing to create download package...', 
                          current_file='Starting...', total_files=total_files, processed_files=0)
            
            # Create a zip file of the results
            zip_filename = f'processed_logos_{upload_id}.zip'
            zip_path = os.path.join(dirs['temp'], zip_filename)
            
            try:
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for i, (file_path, arcname) in enumerate(output_files, 1):
                        # Ensure the directory structure exists in the zip
                        zipf.write(file_path, arcname)
                        
                        # Update progress for each file added to zip with more frequent updates
                        update_progress(session_id, 'zipping', 
                                      min(80 + int(15 * (i / total_files)), 95), 
                                      f'Adding files to archive...',
                                      current_file=os.path.basename(file_path), 
                                      total_files=total_files, 
                                      processed_files=i)
                
                # Update progress - zip created
                update_progress(session_id, 'zipping', 95, 'Finalizing download package...', 
                              current_file='Finalizing...', total_files=total_files, processed_files=total_files)
                
                # Deduct a credit for the operation
                try:
                    if not current_user.subscription or not current_user.subscription.use_credits(1):
                        logger.error("Failed to use credits: Insufficient credits or no subscription")
                        update_progress(session_id, 'error', 0, 'Insufficient credits for this operation')
                        return jsonify({'error': 'Insufficient credits for this operation'}), 402
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"Error updating credits: {str(e)}")
                    update_progress(session_id, 'error', 0, 'Error updating user credits')
                    return jsonify({'error': 'Error processing your request'}), 500
                
                # Update progress - complete
                update_progress(session_id, 'complete', 100, 'Processing complete! Your download will start shortly...', 
                              current_file=filename, total_files=total_files, processed_files=total_files)
                
                # Clean up function to remove temp files after download
                @after_this_request
                def cleanup(response):
                    try:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        if dirs and os.path.exists(dirs['temp']):
                            shutil.rmtree(dirs['temp'], ignore_errors=True)
                        if dirs and os.path.exists(dirs['output']):
                            shutil.rmtree(dirs['output'], ignore_errors=True)
                        if dirs and os.path.exists(dirs['cache']):
                            shutil.rmtree(dirs['cache'], ignore_errors=True)
                    except Exception as e:
                        logger.error(f"Error cleaning up files: {str(e)}")
                    return response
                
                # Return the zip file for download
                return send_file(
                    zip_path,
                    as_attachment=True,
                    download_name=f'processed_logos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                    mimetype='application/zip'
                )
                
            except Exception as zip_error:
                logger.error(f"Error creating zip file: {str(zip_error)}", exc_info=True)
                update_progress(session_id, 'error', 0, f'Error creating download package: {str(zip_error)}')
                return jsonify({'error': 'Error creating download package'}), 500
                
        except Exception as e:
            logger.error(f"Error processing logo: {str(e)}", exc_info=True)
            update_progress(session_id, 'error', 0, f'Error processing logo: {str(e)}')
            return jsonify({'error': f'Error processing logo: {str(e)}'}), 500
            
            # Cache the result
            try:
                os.makedirs(cache_folder, exist_ok=True)
                os.chmod(cache_folder, 0o750)
                shutil.copyfile(zip_path, os.path.join(dirs['cache'], f'processed_logos_{upload_id}.zip'))
                os.chmod(os.path.join(dirs['cache'], f'processed_logos_{upload_id}.zip'), 0o640)
                logger.info(f"Result cached for session {session_id}")
            except Exception as cache_save_err:
                logger.error(f"Error saving result to cache: {str(cache_save_err)}")
            
            # Update user credits
            try:
                current_user.subscription.use_credits(1)
                db.session.commit()
            except Exception as e:
                logger.error(f"Error updating user credits: {str(e)}")
                db.session.rollback()
            
            # Return the zip file for download
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'processed_logos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                mimetype='application/zip'
            )
            
    except ValueError as e:
        if dirs:
            cleanup_dirs(dirs)
        error_msg = str(e)
        details = 'Please check your input and try again.'
        
        # Provide more specific messages for common errors
        if 'unsupported file type' in error_msg.lower():
            status_code = 400
            details = 'Please upload a supported file type (PNG, JPG, GIF, BMP, TIFF, WEBP, or PDF).'
        elif 'invalid or corrupted' in error_msg.lower():
            status_code = 400
            details = 'The uploaded file appears to be corrupted. Please try with a different file.'
        elif 'no variations were generated' in error_msg.lower():
            status_code = 400
            details = 'No variations could be generated from the provided file. Please try with a different file.'
        else:
            status_code = 400
            
        return jsonify({
            'error': error_msg,
            'details': details
        }), status_code
        
    except RequestEntityTooLarge:
        if dirs:
            cleanup_dirs(dirs)
        max_size_mb = current_app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        return jsonify({
            'error': f'File too large',
            'details': f'Maximum file size is {max_size_mb}MB.'
        }), 413
        
    except Exception as e:
        logger.error(f"Error processing logo: {str(e)}", exc_info=True)
        if dirs:
            cleanup_dirs(dirs)
            
        # Handle specific exceptions with more user-friendly messages
        if 'PDF' in str(e) and 'not found' in str(e):
            error_msg = 'Error processing PDF file'
            details = 'The PDF file could not be processed. Please ensure it is not password protected and try again.'
        elif 'memory' in str(e).lower():
            error_msg = 'Insufficient memory'
            details = 'The server is currently under heavy load. Please try again later or use a smaller file.'
        else:
            error_msg = 'An unexpected error occurred'
            details = 'Please try again. If the problem persists, contact support.'
            
        return jsonify({
            'error': error_msg,
            'details': details
        }), 500

@bp.route('/about')
def about():
    return render_template('about.html', now=datetime.now())

@bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('Please fill in all fields', 'error')
            return render_template('contact.html', now=datetime.now())
        
        try:
            # Send email to admin
            msg = Message(
                subject=f"Contact Form: {subject}",
                recipients=['team@zyppts.com'],
                body=f"""
                Name: {name}
                Email: {email}
                Subject: {subject}
                
                Message:
                {message}
                """,
                reply_to=email
            )
            
            mail.send(msg)
            
            # Send confirmation email to user
            confirmation_msg = Message(
                subject="Thank you for contacting Zyppts",
                recipients=[email],
                body=f"""
                Dear {name},
                
                Thank you for contacting Zyppts. We have received your message and will get back to you as soon as possible.
                
                Your message details:
                Subject: {subject}
                
                Best regards,
                The Zyppts Team
                """
            )
            
            mail.send(confirmation_msg)
            
            flash('Your message has been sent successfully! We will get back to you soon.', 'success')
            return redirect(url_for('main.contact'))
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            flash('There was an error sending your message. Please try again later.', 'error')
            return render_template('contact.html', now=datetime.now())
    
    return render_template('contact.html', now=datetime.now())

@bp.route('/subscription/plans')
def subscription_plans():
    return render_template('subscription_plans.html', plans=Config.SUBSCRIPTION_PLANS, now=datetime.now())

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
            
            if not email or not password:
                flash('Please provide both email and password', 'error')
                return render_template('login.html', now=datetime.now())
            
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                # Update last login time
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                login_user(user)
                return redirect(url_for('main.home'))
            
            flash('Invalid email or password', 'error')
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login. Please try again.', 'error')
    
    return render_template('login.html', now=datetime.now())

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            
            if not all([username, email, password]):
                flash('Please fill in all fields', 'error')
                return render_template('register.html', now=datetime.now())
            
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return render_template('register.html', now=datetime.now())
            
            user = User(
                username=username,
                email=email,
                created_at=datetime.utcnow(),
                last_login=None,
                is_active=True
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            login_user(user)
            flash('Registration successful!', 'success')
            return redirect(url_for('main.home'))
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
    
    return render_template('register.html', now=datetime.now())

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))

@bp.route('/preview_variation', methods=['POST'])
@login_required
def preview_variation():
    """Generate a preview of a specific variation"""
    try:
        if 'logo' not in request.files:
            raise ValueError('No file uploaded')
        
        file = request.files['logo']
        variation_type = request.form.get('type', 'original')
        
        img = Image.open(file).convert("RGBA")
        
        if variation_type == 'transparent':
            result = img
        elif variation_type == 'black':
            result = ImageOps.grayscale(img).point(lambda x: 0 if x < 128 else 255, '1').convert("RGBA")
        elif variation_type == 'white':
            result = ImageOps.grayscale(img).point(lambda x: 255 if x < 128 else 0, '1').convert("RGBA")
        elif variation_type.startswith('color_'):
            color_name = variation_type.split('_')[1]
            colors = {
                'primary': Config.BRAND_COLORS['primary']['rgb'],
                'secondary': Config.BRAND_COLORS['secondary']['rgb']
            }
            if color_name in colors:
                colored = Image.new("RGBA", img.size, colors[color_name] + (255,))
                colored.putalpha(img.getchannel('A'))
                result = colored
            else:
                result = img
        else:
            result = img
        
        # Convert to base64 for preview
        buffer = BytesIO()
        result.save(buffer, format="PNG", optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'preview': f'data:image/png;base64,{img_str}'
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error during preview variation: {e}", exc_info=True)
        return jsonify({'error': f'An exception occurred: {str(e)}'}), 500

@bp.route('/test_vector_trace')
def test_vector_trace():
    """
    An internal endpoint to test the vector tracing feature end-to-end.
    This should be protected or removed in a production environment.
    """
    if not current_app.config['DEBUG']:
        return jsonify({'error': 'This endpoint is only available in DEBUG mode.'}), 404

    # --- Configuration ---
    LOGO_URL = "https://mirrors.creativecommons.org/presskit/logos/cc.logo.large.png"
    LOGO_PATH = "/tmp/cc-logo-test.png"
    TEST_EMAIL = f"test-vector-trace-{uuid.uuid4()}@example.com"
    TEST_PASSWORD = "password123"
    
    temp_user = None
    dirs = None
    upload_id = str(uuid.uuid4())

    try:
        # --- Setup ---
        # 1. Download the test logo if it doesn't exist
        if not os.path.exists(LOGO_PATH):
            try:
                response = requests.get(LOGO_URL, stream=True)
                response.raise_for_status()
                with open(LOGO_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                current_app.logger.info(f"Downloaded test logo to {LOGO_PATH}")
            except Exception as e:
                return jsonify({'error': f'Failed to download test logo: {str(e)}'}), 500

        # 2. Create a temporary user with a subscription
        temp_user = User.query.filter_by(email=TEST_EMAIL).first()
        if not temp_user:
            temp_user = User(email=TEST_EMAIL, username=TEST_EMAIL.split('@')[0])
            temp_user.set_password(TEST_PASSWORD)
            db.session.add(temp_user)
            db.session.flush()  # Get user.id

            subscription = Subscription(
                user_id=temp_user.id, plan='studio', status='active',
                monthly_credits=1000, used_credits=0, start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=1)
            )
            db.session.add(subscription)
            db.session.commit()
            current_app.logger.info(f"Created and provisioned test user: {TEST_EMAIL}")

        # --- Execution ---
        dirs = ensure_upload_dirs(upload_id)
        processor = LogoProcessor(
            cache_folder=dirs['cache'], upload_folder=dirs['upload'],
            output_folder=dirs['output'], temp_folder=dirs['temp']
        )
        
        options = {'vector_trace': True}
        shutil.copy(LOGO_PATH, dirs['upload'])
        test_file_path = os.path.join(dirs['upload'], os.path.basename(LOGO_PATH))

        result = processor.process_logo(file_path=test_file_path, options=options)
        
        # --- Validation ---
        if result.get('success') and result.get('outputs'):
            outputs = result['outputs']
            svg_path = outputs.get('vector_trace_svg')
            pdf_path = outputs.get('vector_trace_pdf')
            ai_path = outputs.get('vector_trace_ai')

            if svg_path and pdf_path and ai_path and os.path.exists(svg_path) and os.path.exists(pdf_path) and os.path.exists(ai_path):
                return jsonify({
                    'success': True,
                    'message': 'Vector trace test completed successfully.',
                    'output_paths': {'svg': svg_path, 'pdf': pdf_path, 'ai': ai_path}
                })
            else:
                missing = [f for f, p in [('SVG', svg_path), ('PDF', pdf_path), ('AI', ai_path)] if not p or not os.path.exists(p)]
                return jsonify({
                    'success': False, 'message': 'Vector trace processing failed.',
                    'error_details': f'Output file(s) not found: {", ".join(missing)}'
                }), 500
        else:
            return jsonify({
                'success': False, 'message': 'Vector trace processing failed.',
                'error_details': f"Processing failed. Details: {result.get('message', 'Unknown error')}"
            }), 500

    except Exception as e:
        current_app.logger.error(f"Error during vector trace test: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

    finally:
        # --- Cleanup ---
        if dirs:
            cleanup_dirs(dirs)
        user_to_delete = User.query.filter_by(email=TEST_EMAIL).first()
        if user_to_delete:
            Subscription.query.filter_by(user_id=user_to_delete.id).delete()
            db.session.delete(user_to_delete)
            db.session.commit()
            current_app.logger.info(f"Cleaned up test user and data for {TEST_EMAIL}")

@bp.route('/account')
@login_required
def account():
    """Display account settings and subscription details"""
    return render_template('account.html', now=datetime.now())

@bp.route('/subscription/cancel', methods=['POST'])
@login_required
def cancel_subscription():
    try:
        if not current_user.subscription:
            flash('No active subscription found', 'error')
            return redirect(url_for('main.account'))
            
        if current_user.subscription.status != 'active':
            flash('Subscription is not active', 'error')
            return redirect(url_for('main.account'))
            
        # Update subscription status
        current_user.subscription.status = 'cancelled'
        current_user.subscription.auto_renew = False
        db.session.commit()
        
        # Send confirmation email
        msg = Message(
            subject="Subscription Cancellation Confirmation",
            recipients=[current_user.email],
            body=f"""
            Dear {current_user.username},
            
            Your subscription has been successfully cancelled.
            
            Details:
            - Plan: {current_user.subscription.plan}
            - Cancellation Date: {datetime.utcnow().strftime('%B %d, %Y')}
            - Access will continue until: {current_user.subscription.end_date.strftime('%B %d, %Y') if current_user.subscription.end_date else 'End of current billing period'}
            
            If you have any questions or need assistance, please don't hesitate to contact our support team.
            
            Best regards,
            The Zyppts Team
            """
        )
        mail.send(msg)
        
        flash('Your subscription has been cancelled successfully', 'success')
        return redirect(url_for('main.account'))
        
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        db.session.rollback()
        flash('An error occurred while cancelling your subscription. Please try again later.', 'error')
        return redirect(url_for('main.account'))

@bp.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html', now=datetime.now())

@bp.route('/terms-of-service')
def terms_of_service():
    return render_template('terms_of_service.html', now=datetime.now())

@bp.route('/refund-policy')
def refund_policy():
    return render_template('refund_policy.html', now=datetime.now())

@bp.route('/preview/vector', methods=['POST'])
@login_required
def preview_vector():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate vector trace and return a PNG preview
            result = processor.generate_vector_trace(temp_path, {})
            svg_path = result.get('output_paths', {}).get('svg')
            if svg_path and os.path.exists(svg_path):
                from cairosvg import svg2png
                import base64
                with open(svg_path, 'rb') as f:
                    svg_data = f.read()
                png_bytes = svg2png(bytestring=svg_data)
                img_str = base64.b64encode(png_bytes).decode()
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No SVG generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview/outline', methods=['POST'])
@login_required
def preview_outline():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate contour cutline and return a PNG preview
            result = processor._create_contour_cutline(temp_path)
            if result and isinstance(result, dict):
                png_path = result.get('png')
                if png_path and os.path.exists(png_path):
                    import base64
                    with open(png_path, 'rb') as f:
                        img_data = f.read()
                    img_str = base64.b64encode(img_data).decode()
                    return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No PNG preview generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview/color_separations', methods=['POST'])
@login_required
def preview_color_separations():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate color separations and return PNG previews
            result = processor._create_color_separations(temp_path)
            if result and isinstance(result, dict):
                pngs = result.get('pngs', [])
                previews = []
                
                for png_info in pngs:
                    if isinstance(png_info, tuple):
                        png_path, label = png_info
                    else:
                        png_path = png_info
                    if png_path and os.path.exists(png_path):
                        import base64
                        with open(png_path, 'rb') as f:
                            img_data = f.read()
                        img_str = base64.b64encode(img_data).decode()
                        previews.append(f'data:image/png;base64,{img_str}')
                
                return jsonify({'success': True, 'previews': previews})
            return jsonify({'error': 'No color separations generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview/distressed', methods=['POST'])
@login_required
def preview_distressed():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate distressed effect and return a PNG preview
            result = processor._create_distressed_version(temp_path)
            if result and os.path.exists(result):
                import base64
                with open(result, 'rb') as f:
                    img_data = f.read()
                img_str = base64.b64encode(img_data).decode()
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No PNG generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview/black', methods=['POST'])
@login_required
def preview_black():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate black version and return a PNG preview
            result = processor._create_black_version(temp_path)
            if result and os.path.exists(result):
                import base64
                with open(result, 'rb') as f:
                    img_data = f.read()
                img_str = base64.b64encode(img_data).decode()
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No PNG generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview/transparent', methods=['POST'])
@login_required
def preview_transparent():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Generate transparent PNG and return a PNG preview
            result = processor._create_transparent_png(temp_path)
            if result and os.path.exists(result):
                import base64
                with open(result, 'rb') as f:
                    img_data = f.read()
                img_str = base64.b64encode(img_data).decode()
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No PNG generated'}), 500
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/performance/stats')
@login_required
def get_performance_stats():
    """Get current performance statistics and optimization recommendations."""
    try:
        processor = LogoProcessor()
        stats = processor.get_performance_stats()
        recommendations = processor.get_optimization_recommendations()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/performance/optimize', methods=['POST'])
@login_required
def optimize_performance():
    """Optimize performance configuration based on target profile."""
    try:
        target = request.json.get('target', 'balanced')
        if target not in ['speed', 'memory', 'balanced']:
            return jsonify({'error': 'Invalid target. Must be speed, memory, or balanced'}), 400
        
        processor = LogoProcessor()
        result = processor.optimize_configuration(target)
        
        return jsonify({
            'success': True,
            'optimization_result': result,
            'message': f'Configuration optimized for {target} performance'
        })
    except Exception as e:
        logger.error(f"Error optimizing performance: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/performance/config', methods=['GET', 'POST'])
@login_required
def manage_performance_config():
    """Get or update performance configuration."""
    try:
        processor = LogoProcessor()
        
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'config': processor.parallel_config,
                'current_stats': processor.get_performance_stats()
            })
        else:
            # POST - update configuration
            new_config = request.json.get('config', {})
            if not isinstance(new_config, dict):
                return jsonify({'error': 'Invalid configuration format'}), 400
            
            # Validate and apply new configuration
            processor.parallel_config.update(new_config)
            
            # Update max_workers if specified
            if 'max_workers' in new_config:
                processor.max_workers = new_config['max_workers']
                # Recreate thread pool
                if hasattr(processor, 'thread_pool'):
                    processor.thread_pool.shutdown(wait=True)
                processor.thread_pool = ThreadPoolExecutor(
                    max_workers=processor.max_workers,
                    thread_name_prefix='logo_processor'
                )
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully',
                'new_config': processor.parallel_config
            })
    except Exception as e:
        logger.error(f"Error managing performance config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/api/high-quality-vector-trace', methods=['POST'])
def high_quality_vector_trace():
    """High-quality single-color vector tracing with VTracer."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get processing options
        options = {
            'simplify': float(request.form.get('simplify', 1.0)),
            'turdsize': int(request.form.get('turdsize', 2)),
            'noise_reduction': request.form.get('noise_reduction', 'true').lower() == 'true',
            'adaptive_threshold': request.form.get('adaptive_threshold', 'true').lower() == 'true',
            'preview': request.form.get('preview', 'true').lower() == 'true',
            'output_format': request.form.get('output_format', 'both')
        }
        
        # Process with high-quality vector tracing
        processor = LogoProcessor()
        result = processor.generate_vector_trace(file_path, options)
        
        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'message': 'High-quality vector tracing completed successfully',
                'outputs': result['output_paths'],
                'previews': result['preview_paths'],
                'processing_time': result['processing_time'],
                'organized_structure': result['organized_structure']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['message']
            }), 500
            
    except Exception as e:
        logger.error(f"High-quality vector trace error: {e}", exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500