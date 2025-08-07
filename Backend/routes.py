from flask import Blueprint, render_template, redirect, url_for, flash, request, send_file, jsonify, current_app, Response, stream_with_context
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime, timedelta

try:
    from .models import db, User, Subscription
except ImportError:
    from models import db, User, Subscription

try:
    from .app_config import mail
except ImportError:
    from app_config import mail

try:
    from .config import Config
except ImportError:
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

try:
    from .utils.logo_processor import LogoProcessor
except ImportError:
    from utils.logo_processor import LogoProcessor

try:
    from .utils.analytics_tracker import track_user_action, track_upload, track_processing_completion
except ImportError:
    from utils.analytics_tracker import track_user_action, track_upload, track_processing_completion

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
import stripe

# Initialize Stripe with your secret key from config
stripe.api_key = Config.STRIPE_SECRET_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('main', __name__)

# Store progress for each session
progress_data = defaultdict(dict)
progress_lock = threading.Lock()

# Health check endpoint
@bp.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Check database connection
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))
        db.session.commit()
        
        # Check Redis connection if configured
        redis_status = "not_configured"
        try:
            import redis
            redis_url = current_app.config.get('REDIS_URL')
            if redis_url:
                redis_client = redis.from_url(redis_url)
                redis_client.ping()
                redis_status = "connected"
        except Exception:
            redis_status = "failed"
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'redis': redis_status,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Favicon route
@bp.route('/favicon.ico')
def favicon():
    """Serve favicon.ico at root"""
    try:
        favicon_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'favicon.ico')
        if os.path.exists(favicon_path):
            response = send_file(
                favicon_path,
                mimetype='image/vnd.microsoft.icon'
            )
            # Add cache headers for favicon
            response.headers['Cache-Control'] = 'public, max-age=86400'  # Cache for 24 hours
            response.headers['Pragma'] = 'cache'
            return response
        else:
            # Return a 404 if favicon doesn't exist
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving favicon.ico: {e}")
        return '', 404

@bp.route('/favicon.png')
def favicon_png():
    """Serve favicon.png as alternative"""
    try:
        favicon_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'favicon-32x32.png')
        if os.path.exists(favicon_path):
            response = send_file(
                favicon_path,
                mimetype='image/png'
            )
            # Add cache headers for favicon
            response.headers['Cache-Control'] = 'public, max-age=86400'  # Cache for 24 hours
            response.headers['Pragma'] = 'cache'
            return response
        else:
            # Return a 404 if favicon doesn't exist
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving favicon.png: {e}")
        return '', 404

@bp.route('/apple-touch-icon.png')
def apple_touch_icon():
    """Serve Apple touch icon"""
    try:
        icon_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'apple-touch-icon.png')
        if os.path.exists(icon_path):
            response = send_file(
                icon_path,
                mimetype='image/png'
            )
            response.headers['Cache-Control'] = 'public, max-age=86400'
            return response
        else:
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving apple-touch-icon.png: {e}")
        return '', 404

@bp.route('/site.webmanifest')
def site_webmanifest():
    """Serve site webmanifest"""
    try:
        manifest_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'site.webmanifest')
        if os.path.exists(manifest_path):
            response = send_file(
                manifest_path,
                mimetype='application/manifest+json'
            )
            response.headers['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour
            return response
        else:
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving site.webmanifest: {e}")
        return '', 404

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
        # Update progress - starting
        update_progress(session_id, 'initializing', 5, 'Preparing to process files...')
        
        # Start timing for analytics
        start_time = time.time()
        
        # Create directories for this upload
        upload_id = str(uuid.uuid4())
        dirs = ensure_upload_dirs(upload_id)
        
        # Check if user has batch processing capability (Studio plan or higher)
        has_batch_processing = current_user.subscription and current_user.subscription.plan in ['studio', 'enterprise']
        
        # Handle single file or batch processing
        uploaded_files = []
        
        # Get all files from request
        files = request.files.getlist('logo')
        logger.info(f"Files received: {len(files)} files")
        logger.info(f"File names: {[f.filename for f in files if f.filename]}")
        
        # Check for batch mode - either explicit batch_mode parameter or multiple files
        batch_mode = 'batch_mode' in request.form or len([f for f in files if f.filename]) > 1
        logger.info(f"Batch mode detected: {batch_mode}")
        logger.info(f"Has batch processing capability: {has_batch_processing}")
        logger.info(f"Form fields: {list(request.form.keys())}")
        
        if has_batch_processing and batch_mode:
            # Batch processing mode - handle multiple files
            logger.info("Starting batch processing mode")
            
            if not files or all(f.filename == '' for f in files):
                update_progress(session_id, 'error', 0, 'No files selected for batch processing')
                return jsonify({'error': 'No files selected for batch processing'}), 400
            
            # Validate and save all files
            for i, file in enumerate(files):
                if file.filename == '':
                    continue
                    
                # Validate file type
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.pdf']:
                    update_progress(session_id, 'error', 0, f'Unsupported file type: {file_ext} in {file.filename}')
                    return jsonify({'error': f'Unsupported file type: {file_ext} in {file.filename}. Please upload supported image or PDF files.'}), 400
                
                if not allowed_file(file.filename):
                    update_progress(session_id, 'error', 0, f'File type not allowed: {file.filename}')
                    return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(dirs['upload'], filename)
                file.save(file_path)
                
                # Track upload for analytics
                file_size = os.path.getsize(file_path)
                file_type = file_ext.lstrip('.')
                track_upload(
                    user_id=current_user.id,
                    filename=filename,
                    original_filename=file.filename,
                    file_size=file_size,
                    file_type=file_type,
                    status='pending'
                )
                
                uploaded_files.append((filename, file_path))
                logger.info(f"Saved file {i+1}: {filename}")
            
            if not uploaded_files:
                update_progress(session_id, 'error', 0, 'No valid files uploaded')
                return jsonify({'error': 'No valid files uploaded'}), 400
            
            total_files = len(uploaded_files)
            logger.info(f"Batch processing {total_files} files for user {current_user.username}")
            
        else:
            # Single file processing (existing logic)
            logger.info("Starting single file processing mode")
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
            
            # Track upload for analytics
            file_size = os.path.getsize(file_path)
            file_type = file_ext.lstrip('.')
            track_upload(
                user_id=current_user.id,
                filename=filename,
                original_filename=file.filename,
                file_size=file_size,
                file_type=file_type,
                status='pending'
            )
            
            uploaded_files = [(filename, file_path)]
            total_files = 1
            logger.info(f"Single file processing: {filename}")

        # Initialize processor with the uploaded file
        processor = LogoProcessor(
            cache_folder=dirs['cache'],
            upload_folder=dirs['upload'],
            output_folder=dirs['output'],
            temp_folder=dirs['temp']
        )
        
        # Get processing options from form
        logger.info(f"Form fields received: {list(request.form.keys())}")
        logger.info(f"Form data: {dict(request.form)}")
        
        # Parse selected variations from form data
        options = {}
        
        # Basic variations
        basic_variations = [
            'transparent_png', 'black_version', 'pdf_version', 'webp_version', 
            'favicon', 'email_header'
        ]
        
        for variation in basic_variations:
            if variation in request.form and request.form[variation] in ['true', 'on']:
                options[variation] = True
                logger.info(f"✅ Selected variation: {variation}")
            else:
                options[variation] = False
                logger.info(f"❌ Not selected: {variation}")
        
        # Effects variations - map form field names to internal names
        effects_mapping = {
            'effect_vector': 'vector_trace',
            'effect_color_separations': 'color_separations', 
            'effect_distressed': 'distressed_effect',
            'effect_outline': 'contour_cut',
            'effect_halftone': 'halftone'
        }
        
        # Initialize all effects as False
        for internal_name in effects_mapping.values():
            options[internal_name] = False
        
        # Check form fields and map to internal names
        for form_field, internal_name in effects_mapping.items():
            if form_field in request.form and request.form[form_field] in ['true', 'on']:
                options[internal_name] = True
                logger.info(f"✅ Selected effect: {form_field} -> {internal_name}")
            else:
                logger.info(f"❌ Not selected: {form_field} -> {internal_name}")
        
        # Social media formats - map form field names to internal names
        social_mapping = {
            'social_instagram_post': 'instagram_post',
            'social_instagram_story': 'instagram_story', 
            'social_facebook_post': 'facebook_post',
            'social_twitter_post': 'twitter_post',
            'social_linkedin_post': 'linkedin_post'
        }
        
        social_formats = {}
        for form_field, internal_name in social_mapping.items():
            if form_field in request.form and request.form[form_field] in ['true', 'on']:
                social_formats[internal_name] = True
                logger.info(f"✅ Selected social format: {form_field} -> {internal_name}")
            else:
                social_formats[internal_name] = False
                logger.info(f"❌ Not selected: {form_field} -> {internal_name}")
        
        options['social_formats'] = social_formats
        
        # Log the final options
        logger.info(f"📋 Final processing options: {options}")
        
        # Check if any variations were selected
        selected_variations = sum(1 for v in basic_variations + list(effects_mapping.values()) if options.get(v, False))
        selected_social = sum(1 for v in social_formats.values() if v)
        
        if selected_variations == 0 and selected_social == 0:
            update_progress(session_id, 'error', 0, 'No variations selected')
            return jsonify({'error': 'Please select at least one variation to process'}), 400
        
        logger.info(f"🎯 Processing {selected_variations} variations and {selected_social} social formats")
        
        # Update progress - starting processing
        update_progress(session_id, 'processing', 20, f'Processing {"files" if total_files > 1 else "logo"} with selected effects...', 
                      current_file=uploaded_files[0][0] if uploaded_files else "Unknown", total_files=total_files, processed_files=0)
        
        # Process files based on plan type
        all_output_files = []
        summary_lines = []
        
        logger.info(f"Starting to process {total_files} files with options: {options}")
        
        for file_index, (filename, file_path) in enumerate(uploaded_files):
            current_file_progress = file_index + 1
            
            logger.info(f"Processing file {current_file_progress} of {total_files}: {filename}")
            
            # Update progress for current file
            update_progress(session_id, 'processing', 
                          20 + int(60 * (file_index / total_files)), 
                          f'Processing file {current_file_progress} of {total_files}: {filename}...', 
                          current_file=filename, total_files=total_files, processed_files=file_index)
            
            # Process the logo with the selected options
            logger.info(f"Calling process_logo for {filename} with options: {options}")
            result = processor.process_logo_parallel(file_path=file_path, options=options)
            
            # Log the outputs dictionary for debugging
            logger.info(f"Outputs returned from process_logo for {filename}: {result.get('outputs', {})}")
            
            # Verify the processing was successful
            if not result.get('success'):
                error_msg = result.get('message', 'Unknown error during processing')
                logger.error(f"Logo processing failed for {filename}: {error_msg}")
                update_progress(session_id, 'error', 0, f"Processing failed for {filename}: {error_msg}")
                return jsonify({'error': f"Processing failed for {filename}: {error_msg}"}), 500
            
            logger.info(f"Successfully processed {filename}")
            
            # Get the output files from the result
            outputs = result.get('outputs', {})
            
            # Get the original filename without extension
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            
            logger.info(f"Adding output files for {filename} to zip")
            
            # Helper function to safely add files to the output list with proper naming
            def add_file(file_path, dest_path, use_original_name=True):
                """Helper function to safely add files to the output list with proper naming"""
                # Handle dictionary outputs (extract the actual file path)
                if isinstance(file_path, dict):
                    # Handle contour cutline outputs
                    if 'outputs' in file_path:
                        outputs = file_path['outputs']
                        for output_type, path in outputs.items():
                            if isinstance(path, str) and os.path.exists(path):
                                # Create appropriate destination path based on output type
                                file_ext = os.path.splitext(path)[1]
                                if total_files > 1:
                                    arcname = f"{original_name}/Effects/{original_name}_contourcut_{output_type}{file_ext}"
                                else:
                                    arcname = f"Effects/{original_name}_contourcut_{output_type}{file_ext}"
                                all_output_files.append((path, arcname))
                                logger.info(f"Added contour cutline output: {path} -> {arcname}")
                        return True
                    # Try to find the actual file path in the dictionary
                    for key in ['pdf', 'png', 'svg', 'ico', 'webp', 'eps']:
                        if key in file_path:
                            file_path = file_path[key]
                            break
                    else:
                        # If we can't find a known key, try the first string value
                        for key, value in file_path.items():
                            if isinstance(value, str) and os.path.exists(value):
                                file_path = value
                                break
                        else:
                            logger.warning(f"Could not extract file path from dictionary: {file_path}")
                            return False

                if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
                    logger.warning(f"Expected output file missing: {file_path}")
                    return False

                try:
                    # For batch processing, organize files by logo name
                    if total_files > 1:
                        # Create folder structure: LogoName/FileType/filename
                        all_output_files.append((file_path, dest_path))
                        logger.info(f"Added to batch output: {file_path} -> {dest_path}")
                    else:
                        # Single file processing - use existing structure
                        all_output_files.append((file_path, dest_path))
                        logger.info(f"Added to single output: {file_path} -> {dest_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error adding file to output list: {str(e)}")
                    return False

            # Process the outputs based on selected variations
            logger.info(f"Processing outputs for {filename} with selected options: {options}")
            
            # Handle basic variations
            basic_variation_mapping = {
                'transparent_png': '_create_transparent_png',
                'black_version': '_create_black_version', 
                'pdf_version': '_create_pdf_version',
                'webp_version': '_create_webp_version',
                'favicon': '_create_favicon',
                'email_header': '_create_email_header'
            }
            
            for variation, func_name in basic_variation_mapping.items():
                if options.get(variation, False) and func_name in outputs:
                    result = outputs[func_name]
                    logger.info(f"Processing {variation}: {result}")
                    
                    if isinstance(result, dict):
                        # Handle dictionary results (extract file paths)
                        for key, path in result.items():
                            if isinstance(path, str) and os.path.exists(path):
                                ext = os.path.splitext(path)[1].lstrip('.') or 'png'
                                if total_files > 1:
                                    arcname = f"{original_name}/Basic/{original_name}_{variation}.{ext}"
                                else:
                                    arcname = f"Basic/{original_name}_{variation}.{ext}"
                                if add_file(path, arcname):
                                    logger.info(f"✅ Added {variation}: {path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: {variation.replace('_', ' ').title()}: {arcname}")
                    elif isinstance(result, str) and os.path.exists(result):
                        # Handle string results (direct file paths)
                        ext = os.path.splitext(result)[1].lstrip('.') or 'png'
                        if total_files > 1:
                            arcname = f"{original_name}/Basic/{original_name}_{variation}.{ext}"
                        else:
                            arcname = f"Basic/{original_name}_{variation}.{ext}"
                        if add_file(result, arcname):
                            logger.info(f"✅ Added {variation}: {result} -> {arcname}")
                            summary_lines.append(f"{original_name}: {variation.replace('_', ' ').title()}: {arcname}")
                    else:
                        logger.warning(f"⚠️ Invalid result for {variation}: {result}")
                else:
                    logger.info(f"📝 Skipping {variation} - not selected or not in outputs")
            
            # Handle effects variations
            effects_variation_mapping = {
                'vector_trace': '_create_vector_trace',
                'full_color_vector_trace': '_create_full_color_vector_trace',
                'color_separations': '_create_color_separations',
                'distressed_effect': '_create_distressed_version',
                'halftone': '_create_halftone',
                'contour_cut': '_create_contour_cutline'
            }
            
            for variation, func_name in effects_variation_mapping.items():
                if options.get(variation, False) and func_name in outputs:
                    result = outputs[func_name]
                    logger.info(f"Processing {variation}: {result}")
                    
                    if variation == 'contour_cut':
                        # Special handling for contour cutline (has nested outputs)
                        if isinstance(result, dict) and 'outputs' in result:
                            outputs_dict = result['outputs']
                            
                            # Handle full color mask
                            if 'full_color_mask' in outputs_dict:
                                path = outputs_dict['full_color_mask']
                                if total_files > 1:
                                    arcname = f"{original_name}/Effects/{original_name}_contourcut_fullcolor.png"
                                else:
                                    arcname = f"Effects/{original_name}_contourcut_fullcolor.png"
                                if add_file(path, arcname):
                                    logger.info(f"✅ Added contour cutline full color: {path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: Contour Cutline (Full Color): {arcname}")
                            
                            # Handle pink outline mask
                            if 'pink_outline_mask' in outputs_dict:
                                path = outputs_dict['pink_outline_mask']
                                if total_files > 1:
                                    arcname = f"{original_name}/Effects/{original_name}_contourcut_outline.png"
                                else:
                                    arcname = f"Effects/{original_name}_contourcut_outline.png"
                                if add_file(path, arcname):
                                    logger.info(f"✅ Added contour cutline outline: {path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: Contour Cutline (Pink Outline): {arcname}")
                            
                            # Handle combined SVG
                            if 'combined_svg' in outputs_dict:
                                path = outputs_dict['combined_svg']
                                if total_files > 1:
                                    arcname = f"{original_name}/Effects/{original_name}_contourcut_combined.svg"
                                else:
                                    arcname = f"Effects/{original_name}_contourcut_combined.svg"
                                if add_file(path, arcname):
                                    logger.info(f"✅ Added contour cutline SVG: {path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: Contour Cutline (Combined SVG): {arcname}")
                            
                            # Handle combined PDF (NEW)
                            if 'combined_pdf' in outputs_dict:
                                path = outputs_dict['combined_pdf']
                                if total_files > 1:
                                    arcname = f"{original_name}/Effects/{original_name}_contourcut_combined.pdf"
                                else:
                                    arcname = f"Effects/{original_name}_contourcut_combined.pdf"
                                if add_file(path, arcname):
                                    logger.info(f"✅ Added contour cutline PDF: {path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: Contour Cutline (Combined PDF): {arcname}")
                            
                            # Add contour count info if available
                            if 'message' in result:
                                summary_lines.append(f"{original_name}: {result['message']}")
                    elif variation == 'color_separations':
                        # Special handling for color separations (has separations list)
                        if isinstance(result, dict) and 'separations' in result:
                            separations = result['separations']
                            if isinstance(separations, list):
                                for sep in separations:
                                    if isinstance(sep, dict):
                                        channel = sep.get('channel', 'unknown')
                                        # Add PNG file
                                        if 'png' in sep and os.path.exists(sep['png']):
                                            if total_files > 1:
                                                arcname = f"{original_name}/Effects/{original_name}_colorsep_{channel}.png"
                                            else:
                                                arcname = f"Effects/{original_name}_colorsep_{channel}.png"
                                            if add_file(sep['png'], arcname):
                                                logger.info(f"✅ Added color separation {channel} PNG: {sep['png']} -> {arcname}")
                                                summary_lines.append(f"{original_name}: Color Separation ({channel}): {arcname}")
                                        
                                        # Add SVG file
                                        if 'svg' in sep and os.path.exists(sep['svg']):
                                            if total_files > 1:
                                                arcname = f"{original_name}/Effects/{original_name}_colorsep_{channel}.svg"
                                            else:
                                                arcname = f"Effects/{original_name}_colorsep_{channel}.svg"
                                            if add_file(sep['svg'], arcname):
                                                logger.info(f"✅ Added color separation {channel} SVG: {sep['svg']} -> {arcname}")
                                                summary_lines.append(f"{original_name}: Color Separation ({channel} SVG): {arcname}")
                                        
                                        # Add AI file
                                        if 'ai' in sep and os.path.exists(sep['ai']):
                                            if total_files > 1:
                                                arcname = f"{original_name}/Effects/{original_name}_colorsep_{channel}.ai"
                                            else:
                                                arcname = f"Effects/{original_name}_colorsep_{channel}.ai"
                                            if add_file(sep['ai'], arcname):
                                                logger.info(f"✅ Added color separation {channel} AI: {sep['ai']} -> {arcname}")
                                                summary_lines.append(f"{original_name}: Color Separation ({channel} AI): {arcname}")
                                
                                # Add color count info if available
                                if 'num_colors' in result:
                                    summary_lines.append(f"{original_name}: Color Separations - {result['num_colors']} colors detected")
                            else:
                                logger.warning(f"⚠️ Invalid color separations structure: {result}")
                        else:
                            logger.warning(f"⚠️ Invalid result for color_separations: {result}")
                    else:
                        # Handle other effects variations
                        if isinstance(result, dict):
                            # Handle dictionary results
                            for key, path in result.items():
                                if isinstance(path, str) and os.path.exists(path):
                                    ext = os.path.splitext(path)[1].lstrip('.') or 'png'
                                    if total_files > 1:
                                        arcname = f"{original_name}/Effects/{original_name}_{variation}_{key}.{ext}"
                                    else:
                                        arcname = f"Effects/{original_name}_{variation}_{key}.{ext}"
                                    if add_file(path, arcname):
                                        logger.info(f"✅ Added {variation} {key}: {path} -> {arcname}")
                                        summary_lines.append(f"{original_name}: {variation.replace('_', ' ').title()}: {arcname}")
                                else:
                                    logger.warning(f"⚠️ Invalid result for {variation}: {result}")
                        elif isinstance(result, str) and os.path.exists(result):
                            # Handle string results
                            ext = os.path.splitext(result)[1].lstrip('.') or 'png'
                            if total_files > 1:
                                arcname = f"{original_name}/Effects/{original_name}_{variation}.{ext}"
                            else:
                                arcname = f"Effects/{original_name}_{variation}.{ext}"
                            if add_file(result, arcname):
                                logger.info(f"✅ Added {variation}: {result} -> {arcname}")
                                summary_lines.append(f"{original_name}: {variation.replace('_', ' ').title()}: {arcname}")
                            else:
                                logger.warning(f"⚠️ Invalid result for {variation}: {result}")
                        else:
                            logger.warning(f"⚠️ Invalid result for {variation}: {result}")

            # Handle social media formats
            if any(social_formats.values()) and '_create_social_formats' in outputs:
                social_media = outputs['_create_social_formats']
                logger.info(f"Processing social media formats: {social_media}")
                if isinstance(social_media, dict):
                    for platform, format_path in social_media.items():
                        if platform in social_formats and social_formats[platform]:
                            if isinstance(format_path, str) and os.path.exists(format_path):
                                ext = os.path.splitext(format_path)[1].lstrip('.') or 'png'
                                if total_files > 1:
                                    arcname = f"{original_name}/Social Media/{original_name}_{platform}.{ext}"
                                else:
                                    arcname = f"Social Media/{original_name}_{platform}.{ext}"
                                if add_file(format_path, arcname):
                                    logger.info(f"✅ Added social format {platform}: {format_path} -> {arcname}")
                                    summary_lines.append(f"{original_name}: Social Media ({platform}): {arcname}")
                            else:
                                logger.warning(f"⚠️ Invalid social format path for {platform}: {format_path}")
                        else:
                            logger.info(f"📝 Skipping {platform} - not selected by user")
                else:
                    logger.warning(f"⚠️ Unexpected social_media format: {type(social_media)}")
            else:
                logger.info(f"📝 Skipping social formats - not selected or not in outputs")

        # Log the collected files before creating zip
        logger.info(f"All collected output files: {all_output_files}")
        logger.info(f"Summary lines: {summary_lines}")

        # Create zip file with all outputs
        if all_output_files:
            zip_path = os.path.join(dirs['temp'], 'processed_logos.zip')
            
            # Generate zip filename based on uploaded files
            if total_files == 1:
                # Single file: use the uploaded filename + "variations"
                uploaded_filename = os.path.splitext(uploaded_files[0][0])[0]  # Remove extension
                zip_filename = f"{uploaded_filename}variations.zip"
            else:
                # Multiple files: use a combination of filenames or generic name
                if total_files <= 3:
                    # For 2-3 files, combine their names
                    filenames = [os.path.splitext(f[0])[0] for f in uploaded_files]
                    zip_filename = f"{'_'.join(filenames)}variations.zip"
                else:
                    # For more than 3 files, use a generic name
                    zip_filename = f"batch_variations_{total_files}_files.zip"
            
            try:
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add readme first
                    readme_path = os.path.join(dirs['temp'], 'readme.txt')
                    with open(readme_path, 'w') as f:
                        f.write("Zyppts Logo Processor Output\n\n")
                        if total_files > 1:
                            f.write(f"Batch processed {total_files} logos\n\n")
                        else:
                            f.write(f"Processed file: {uploaded_files[0][0]}\n\n")
                        f.write("Included files:\n")
                        for line in summary_lines:
                            f.write(f"- {line}\n")
                        f.write("\nThank you for using Zyppts!\n")
                    
                    # Add readme to zip
                    zipf.write(readme_path, 'readme.txt')
                    logger.info(f"Added readme to zip: {readme_path}")
                    
                    # Add all output files
                    for file_path, arcname in all_output_files:
                        if os.path.exists(file_path):
                            try:
                                zipf.write(file_path, arcname)
                                logger.info(f"Successfully added to zip: {file_path} -> {arcname}")
                            except Exception as e:
                                logger.error(f"Error adding file to zip - {file_path}: {str(e)}")
                        else:
                            logger.error(f"File not found when creating zip: {file_path}")

                # Verify zip file was created and has content
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                    logger.info(f"Zip file created successfully: {zip_path} ({os.path.getsize(zip_path)} bytes)")
                    
                    # Track processing completion for analytics
                    processing_time = time.time() - start_time if 'start_time' in locals() else 0
                    variations_generated = len(all_output_files)
                    credits_used = total_files  # 1 credit per file
                    
                    track_processing_completion(
                        user_id=current_user.id,
                        processing_time=processing_time,
                        files_processed=total_files,
                        variations_generated=variations_generated,
                        credits_used=credits_used
                    )
                    
                    # Add cleanup after download
                    @after_this_request
                    def cleanup(response):
                        try:
                            if os.path.exists(zip_path):
                                os.remove(zip_path)
                            if dirs:
                                cleanup_dirs(dirs)
                            logger.info("Cleanup completed after download")
                        except Exception as e:
                            logger.error(f"Error during cleanup: {str(e)}")
                        return response

                    return send_file(
                        zip_path,
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name=zip_filename
                    )
                else:
                    logger.error(f"Zip file creation failed or file is empty: {zip_path}")
                    return jsonify({'error': 'Failed to create zip file'}), 500
                    
            except Exception as e:
                logger.error(f"Error creating zip file: {str(e)}", exc_info=True)
                return jsonify({'error': 'Failed to create zip file', 'details': str(e)}), 500
        else:
            logger.error("No output files were collected for zip creation")
            return jsonify({'error': 'No output files were generated', 'details': 'No files were collected for processing'}), 500

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
                recipients=['zyppts@gmail.com'],
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
                
                # Track login action
                track_user_action(user.id, 'login', {
                    'login_method': 'email',
                    'ip_address': request.remote_addr
                })
                
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
            
            # Create user with additional registration data
            user = User(
                username=username,
                email=email,
                created_at=datetime.utcnow(),
                last_login=None,
                is_active=True
            )
            
            # Store registration metadata for admin tracking
            user.registration_ip = request.remote_addr
            user.registration_user_agent = request.headers.get('User-Agent', 'Unknown')
            
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            # Send email notifications
            try:
                from utils.email_notifications import send_new_account_notification
                send_new_account_notification(user)
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
                # Don't fail registration if email fails
            
            login_user(user)
            flash('Registration successful! Welcome to ZYPPTS!', 'success')
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
        
        # Process with vector trace option
        options = {'vector_trace': True}
        shutil.copy(LOGO_PATH, dirs['upload'])
        test_file_path = os.path.join(dirs['upload'], os.path.basename(LOGO_PATH))

        result = processor.process_logo_parallel(file_path=test_file_path, options=options)
        
        # --- Validation ---
        if result.get('success') and result.get('outputs'):
            outputs = result['outputs']
            vector_trace_output = outputs.get('_create_vector_trace')
            
            if vector_trace_output and isinstance(vector_trace_output, dict):
                svg_path = vector_trace_output.get('svg')
                pdf_path = vector_trace_output.get('pdf')
                eps_path = vector_trace_output.get('eps')

                if svg_path and pdf_path and eps_path and os.path.exists(svg_path) and os.path.exists(pdf_path) and os.path.exists(eps_path):
                    return jsonify({
                        'success': True,
                        'message': 'Vector trace test completed successfully.',
                        'output_paths': {'svg': svg_path, 'pdf': pdf_path, 'eps': eps_path}
                    })
                else:
                    missing = [f for f, p in [('SVG', svg_path), ('PDF', pdf_path), ('EPS', eps_path)] if not p or not os.path.exists(p)]
                    return jsonify({
                        'success': False, 'message': 'Vector trace processing failed.',
                        'error_details': f'Output file(s) not found: {", ".join(missing)}'
                    }), 500
            else:
                return jsonify({
                    'success': False, 'message': 'Vector trace processing failed.',
                    'error_details': f'Vector trace output not found or invalid: {vector_trace_output}'
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
        logger.info("Vector preview: Starting")
        file = request.files['file']
        processor = LogoProcessor()
        
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        logger.info(f"Vector preview: Saving file to {temp_path}")
        file.save(temp_path)
        
        try:
            # Generate vector trace and return a PNG preview
            logger.info("Vector preview: Calling generate_vector_trace")
            result = processor.generate_vector_trace(temp_path, {})
            logger.info(f"Vector preview: Result = {result}")
            
            svg_path = result.get('output_paths', {}).get('svg')
            logger.info(f"Vector preview: SVG path = {svg_path}")
            
            if svg_path and os.path.exists(svg_path):
                import base64
                import cairosvg
                logger.info("Vector preview: Reading SVG file")
                with open(svg_path, 'r') as f:
                    svg_data = f.read()
                logger.info(f"Vector preview: SVG content length = {len(svg_data)}")
                
                logger.info("Vector preview: Converting SVG to PNG")
                png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
                logger.info(f"Vector preview: PNG conversion successful, {len(png_bytes)} bytes")
                
                img_str = base64.b64encode(png_bytes).decode()
                logger.info("Vector preview: Returning success")
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            
            logger.error(f"Vector preview: No SVG generated or file doesn't exist. SVG path: {svg_path}")
            return jsonify({'error': 'No SVG generated'}), 500
        finally:
            # Clean up temporary files
            logger.info(f"Vector preview: Cleaning up {temp_dir}")
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Vector preview error: {str(e)}", exc_info=True)
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
                # Access the pink_outline_mask from the nested outputs structure
                outputs = result.get('outputs', {})
                png_path = outputs.get('pink_outline_mask')
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

@bp.route('/preview/halftone', methods=['POST'])
@login_required
def preview_halftone():
    try:
        file = request.files['file']
        processor = LogoProcessor()
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        try:
            result = processor._create_halftone(temp_path)
            if result and os.path.exists(result):
                import base64
                with open(result, 'rb') as f:
                    img_data = f.read()
                img_str = base64.b64encode(img_data).decode()
                return jsonify({'success': True, 'preview': f'data:image/png;base64,{img_str}'})
            return jsonify({'error': 'No PNG generated'}), 500
        finally:
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
                'organized_structure': result['organized_structure'],
                'quality_level': result.get('quality_level'),
                'algorithm': result.get('algorithm'),
                'upscale_factor': result.get('upscale_factor'),
                'processing_details': result.get('processing_details')
            })
        else:
            return jsonify({
                'success': False,
                'error': result['message']
            }), 500
            
    except Exception as e:
        logger.error(f"High-quality vector trace error: {e}", exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
@bp.route('/api/create-portal-session', methods=['POST'])
@login_required
def create_portal_session():
    """Create Stripe Customer Portal session for managing subscription and payment methods."""
    try:
        if not current_user.subscription or not current_user.subscription.payment_id:
            return jsonify({'error': 'No active subscription found'}), 400
        
        # Get the Stripe subscription to find the customer ID
        try:
            stripe_subscription = stripe.Subscription.retrieve(current_user.subscription.payment_id)
            customer_id = stripe_subscription.customer
            logger.info(f"Retrieved customer ID: {customer_id} for subscription: {current_user.subscription.payment_id}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid subscription ID: {current_user.subscription.payment_id}, Error: {str(e)}")
            return jsonify({'error': 'Unable to retrieve subscription details. Please contact support.'}), 400
        
        # Create portal session with better return URL handling
        try:
            # Use a more reliable return URL
            if request.host_url.startswith('http://localhost'):
                # For local development
                return_url = "http://localhost:5000/account?portal=success"
            else:
                # For production - use your actual domain
                return_url = "https://usezyppts.com/account?portal=success"
            
            logger.info(f"Creating portal session for customer: {customer_id} with return URL: {return_url}")
            
            portal_session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            
            logger.info(f"Portal session created successfully: {portal_session.id}")
            return jsonify({'portal_url': portal_session.url})
            
        except stripe.error.InvalidRequestError as e:
            error_msg = str(e)
            logger.error(f"Stripe portal session creation failed: {error_msg}")
            
            # Provide more specific error messages
            if "No such customer" in error_msg:
                return jsonify({'error': 'Customer not found in Stripe. Please contact support.'}), 400
            elif "No such configuration" in error_msg:
                return jsonify({'error': 'Customer Portal not configured. Please contact support.'}), 400
            elif "Invalid return URL" in error_msg:
                return jsonify({'error': 'Invalid return URL configuration. Please contact support.'}), 400
            else:
                return jsonify({'error': f'Unable to create portal session: {error_msg}'}), 400
                
    except stripe.error.AuthenticationError:
        logger.error("Stripe authentication failed - check API key")
        return jsonify({'error': 'Payment system configuration error. Please contact support.'}), 500
    except Exception as e:
        logger.error(f"Unexpected portal session error: {str(e)}")
        return jsonify({'error': 'An error occurred while creating the portal session'}), 500

@bp.route('/api/stripe-config', methods=['GET'])
def get_stripe_config():
    """Get Stripe publishable key for frontend."""
    return jsonify({
        'publishable_key': Config.STRIPE_PUBLISHABLE_KEY
    })


@bp.route('/api/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create Stripe checkout session for a selected plan."""
    data = request.get_json() or {}
    plan = data.get('plan', 'pro')
    billing = data.get('billing', 'monthly')  # 'monthly' or 'annual'
    
    plan_config = Config.SUBSCRIPTION_PLANS.get(plan)
    if not plan_config:
        if plan == 'enterprise':
            return jsonify({'redirect': url_for('main.contact'), 'message': 'Please contact us for Enterprise setup'}), 200
        return jsonify({'error': 'Invalid plan'}), 400
    
    # Get the appropriate Stripe price ID based on billing cycle
    stripe_price_id = None
    if billing == 'annual':
        stripe_price_id = plan_config.get('stripe_annual_price_id') or plan_config.get('stripe_price_id')
    else:
        stripe_price_id = plan_config.get('stripe_price_id')
    
    if not stripe_price_id:
        return jsonify({'error': 'Price configuration not found for this plan'}), 400
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            line_items=[{
                'price': stripe_price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.host_url + 'account?success=true',
            cancel_url=request.host_url + 'subscription/plans?canceled=true',
            metadata={
                'user_id': current_user.id,
                'plan': plan,
                'billing': billing
            }
        )
        return jsonify({'checkout_url': checkout_session.url})
    except Exception as e:
        logger.error(f"Stripe checkout error: {str(e)}")
        return jsonify({'error': 'Payment setup failed'}), 500

@bp.route('/api/stripe-webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for subscription activation."""
    payload = request.get_data()
    sig_header = request.headers.get('stripe-signature')
    
    # Get webhook secret from config
    webhook_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET')
    if not webhook_secret:
        logger.error("Stripe webhook secret not configured")
        return jsonify({'error': 'Webhook not configured'}), 500
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        logger.error("Invalid payload in Stripe webhook")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid signature in Stripe webhook")
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle different webhook events
    if event['type'] == 'checkout.session.completed':
        session_obj = event['data']['object']
        user_id = session_obj['metadata'].get('user_id')
        plan = session_obj['metadata'].get('plan')
        billing = session_obj['metadata'].get('billing', 'monthly')  # Default to monthly
        
        user = User.query.get(user_id)
        if user:
            try:
                if not user.subscription:
                    user.subscription = Subscription()
                user.subscription.plan = plan
                user.subscription.status = 'active'
                user.subscription.monthly_credits = Config.SUBSCRIPTION_PLANS[plan].get('monthly_credits', 100)
                user.subscription.used_credits = 0
                user.subscription.start_date = datetime.utcnow()
                user.subscription.payment_id = session_obj.get('subscription')
                user.subscription.billing_cycle = billing  # Store billing cycle
                db.session.commit()
                logger.info(f"Subscription activated for user {user_id}: {plan} ({billing})")
                
                # Send payment confirmation email
                from utils.email_notifications import send_payment_confirmation
                amount = session_obj.get('amount_total', 0) / 100  # Convert from cents
                send_payment_confirmation(
                    user=user,
                    subscription=user.subscription,
                    amount=amount,
                    transaction_id=session_obj.get('id'),
                    payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
                
            except Exception as e:
                logger.error(f"Error activating subscription: {e}")
                db.session.rollback()
                return jsonify({'error': 'Failed to activate subscription'}), 500
        else:
            logger.error(f"User not found for subscription activation: {user_id}")
    
    elif event['type'] == 'invoice.payment_succeeded':
        """Handle successful recurring payments"""
        invoice_obj = event['data']['object']
        subscription_id = invoice_obj.get('subscription')
        
        # Find user by subscription ID
        subscription = Subscription.query.filter_by(payment_id=subscription_id).first()
        if subscription and subscription.user:
            try:
                # Send payment confirmation email
                from utils.email_notifications import send_payment_confirmation
                amount = invoice_obj.get('amount_paid', 0) / 100  # Convert from cents
                send_payment_confirmation(
                    user=subscription.user,
                    subscription=subscription,
                    amount=amount,
                    transaction_id=invoice_obj.get('id'),
                    payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
                logger.info(f"Payment confirmation sent for subscription: {subscription_id}")
                
            except Exception as e:
                logger.error(f"Error sending payment confirmation: {e}")
    
    elif event['type'] == 'invoice.payment_failed':
        """Handle failed recurring payments"""
        invoice_obj = event['data']['object']
        subscription_id = invoice_obj.get('subscription')
        
        # Find user by subscription ID
        subscription = Subscription.query.filter_by(payment_id=subscription_id).first()
        if subscription and subscription.user:
            try:
                # Send payment failed email
                from utils.email_notifications import send_payment_failed
                amount = invoice_obj.get('amount_due', 0) / 100  # Convert from cents
                error_message = invoice_obj.get('last_payment_error', {}).get('message', 'Payment failed')
                send_payment_failed(
                    user=subscription.user,
                    subscription=subscription,
                    amount=amount,
                    transaction_id=invoice_obj.get('id'),
                    payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    error_message=error_message
                )
                logger.info(f"Payment failed notification sent for subscription: {subscription_id}")
                
            except Exception as e:
                logger.error(f"Error sending payment failed notification: {e}")
    
    elif event['type'] == 'customer.subscription.updated':
        """Handle subscription updates (upgrades/downgrades)"""
        subscription_obj = event['data']['object']
        subscription_id = subscription_obj.get('id')
        
        # Find user by subscription ID
        subscription = Subscription.query.filter_by(payment_id=subscription_id).first()
        if subscription and subscription.user:
            try:
                # Check if this is an upgrade (you might need to store previous plan)
                # For now, we'll just log the update
                logger.info(f"Subscription updated for user: {subscription.user.username}")
                
                # You can add upgrade email logic here if needed
                # This would require tracking the previous plan
                
            except Exception as e:
                logger.error(f"Error handling subscription update: {e}")
    
    elif event['type'] == 'customer.subscription.deleted':
        """Handle subscription cancellations"""
        subscription_obj = event['data']['object']
        subscription_id = subscription_obj.get('id')
        
        # Find user by subscription ID
        subscription = Subscription.query.filter_by(payment_id=subscription_id).first()
        if subscription and subscription.user:
            try:
                # Update subscription status
                subscription.status = 'cancelled'
                subscription.end_date = datetime.utcnow()
                db.session.commit()
                
                # Send cancellation email
                from utils.email_notifications import send_account_cancellation
                access_until_date = (datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d')
                send_account_cancellation(
                    user=subscription.user,
                    subscription=subscription,
                    cancellation_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    access_until_date=access_until_date
                )
                logger.info(f"Subscription cancellation processed for user: {subscription.user.username}")
                
            except Exception as e:
                logger.error(f"Error handling subscription cancellation: {e}")
                db.session.rollback()
            
    return jsonify({'status': 'success'})

@bp.app_errorhandler(403)
def forbidden(e):
    from datetime import datetime
    return render_template('paywall_403.html', message=getattr(e, 'description', None), now=datetime.now()), 403

@bp.route('/performance/celery-stats')
@login_required
def get_celery_stats():
    """Get Celery worker performance statistics"""
    try:
        from utils.celery_worker import celery_app, monitor_performance
        
        # Get performance metrics
        stats = monitor_performance.delay()
        result = stats.get(timeout=10)
        
        if result:
            return jsonify({
                'success': True,
                'stats': result,
                'message': 'Celery performance stats retrieved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to retrieve Celery stats'
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting Celery stats: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving Celery stats: {str(e)}'
        }), 500

@bp.route('/safari-pinned-tab.svg')
def safari_pinned_tab():
    """Serve Safari pinned tab SVG"""
    try:
        svg_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'safari-pinned-tab.svg')
        if os.path.exists(svg_path):
            response = send_file(
                svg_path,
                mimetype='image/svg+xml'
            )
            response.headers['Cache-Control'] = 'public, max-age=86400'
            return response
        else:
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving safari-pinned-tab.svg: {e}")
        return '', 404

@bp.route('/favicon')
def favicon_fallback():
    """Fallback favicon route for better browser compatibility"""
    try:
        favicon_path = os.path.join(current_app.static_folder, 'images', 'favicon', 'favicon.ico')
        if os.path.exists(favicon_path):
            response = send_file(
                favicon_path,
                mimetype='image/x-icon'
            )
            response.headers['Cache-Control'] = 'public, max-age=86400'
            return response
        else:
            return '', 404
    except Exception as e:
        current_app.logger.error(f"Error serving fallback favicon: {e}")
        return '', 404