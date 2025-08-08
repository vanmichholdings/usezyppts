from celery import Celery
import logging
import os
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get CPU count for ultra-fast worker configuration
CPU_COUNT = multiprocessing.cpu_count()
ULTRA_FAST_WORKERS = min(CPU_COUNT * 2, 16)  # Reduced to 2x CPU cores, max 16 workers for stability

# Create Celery app with ultra-fast configuration
celery_app = Celery('logo_tasks')

# Configure Celery settings for ultra-fast processing (20-second target)
celery_app.conf.update(
    broker_url=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    
    # Increased timeouts for 20-second processing
    task_time_limit=30,  # 30 seconds hard limit
    task_soft_time_limit=20,  # 20 seconds soft limit
    
    # Optimized parallel processing settings
    worker_prefetch_multiplier=4,  # Reduced to 4 tasks for better stability
    task_acks_late=True,  # Acknowledge task only after completion
    worker_disable_rate_limits=True,
    task_reject_on_worker_lost=True,
    
    # Optimized concurrency settings
    worker_concurrency=ULTRA_FAST_WORKERS,  # Use 2x CPU cores for stability
    worker_max_tasks_per_child=100,  # Increased for better performance
    
    # Queue optimizations for speed
    task_default_queue='ultra_fast',
    task_routes={
        'utils.logo_worker.run_ultra_fast_task': {'queue': 'ultra_fast'},
        'utils.logo_worker.run_vector_task': {'queue': 'vector_processing'},
        'utils.logo_worker.run_social_task': {'queue': 'social_processing'},
    },
    
    # Performance optimizations
    task_compression='gzip',  # Compress task data for faster transmission
    result_compression='gzip',  # Compress results for faster transmission
    task_ignore_result=False,  # Keep results for monitoring
    
    # Redis optimizations for stability
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,  # Increased retries
    result_expires=3600,  # Keep results for 1 hour
    
    # Worker optimizations for stability
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_enable_remote_control=True,
    
    # Memory optimizations for stability
    worker_max_memory_per_child=200000,  # 200MB memory limit per worker
    
    # Processing settings
    task_always_eager=False,  # Ensure async processing
    task_eager_propagates=True,
    worker_direct=True,  # Direct worker communication
    
    # Additional stability settings
    broker_connection_retry=True,
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
        'retry_on_timeout': True,
        'max_connections': 20,
    },
)

# Create ultra-fast task for maximum speed
@celery_app.task(name='utils.logo_worker.run_ultra_fast_task', bind=True, queue='ultra_fast')
def run_ultra_fast_task(self, task_func_name, file_path, options):
    """Execute logo processing task with ultra-fast optimization for 20-second target"""
    logger.info(f'⚡ Starting ultra-fast task: {task_func_name} for {file_path}')
    
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f'❌ File not found: {file_path}')
            raise FileNotFoundError(f'File not found: {file_path}')
        
        # Import inside function to avoid circular imports
        from utils.logo_processor import LogoProcessor
        
        processor = LogoProcessor()
        func = getattr(processor, task_func_name)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': f'Executing {task_func_name}'})
        
        # Execute the task with ultra-fast optimization
        if task_func_name == '_create_social_formats':
            logger.info(f'⚡ Executing social formats with options: {options}')
            social_formats = options.get('social_formats', {})
            logger.info(f'⚡ Social formats to create: {social_formats}')
            result = func(file_path, social_formats)
        else:
            logger.info(f'⚡ Executing {task_func_name} with parallel optimization')
            result = func(file_path)
        
        # Update final state
        self.update_state(state='SUCCESS', meta={'status': f'Completed {task_func_name}'})
        
        logger.info(f'✅ Ultra-fast task completed: {task_func_name}')
        return result
        
    except Exception as e:
        logger.error(f'❌ Ultra-fast task failed: {task_func_name} - {str(e)}')
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(name='utils.logo_worker.run_logo_task', bind=True, queue='logo_processing')
def run_logo_task(self, task_func_name, file_path, options):
    """Execute logo processing task with improved error handling and performance"""
    logger.info(f'🚀 Starting task: {task_func_name} for {file_path}')
    
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f'❌ File not found: {file_path}')
            raise FileNotFoundError(f'File not found: {file_path}')
        
        # Import inside function to avoid circular imports
        from utils.logo_processor import LogoProcessor
        
        processor = LogoProcessor()
        func = getattr(processor, task_func_name)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': f'Executing {task_func_name}'})
        
        # Execute the task with better error handling
        if task_func_name == '_create_social_formats':
            logger.info(f'📋 Executing social formats with options: {options}')
            social_formats = options.get('social_formats', {})
            logger.info(f'📋 Social formats to create: {social_formats}')
            result = func(file_path, social_formats)
            logger.info(f'📊 Social formats result type: {type(result)}')
            logger.info(f'📊 Social formats result: {result}')
        else:
            result = func(file_path)
        
        logger.info(f'✅ Completed task: {task_func_name}')
        return result
        
    except Exception as exc:
        # Simplified error handling to avoid Celery serialization issues
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        
        logger.error(f'❌ Task {task_func_name} failed: {exc_type}: {exc_msg}')
        import traceback
        logger.error(f'❌ Traceback: {traceback.format_exc()}')
        
        # Update task state with error information
        self.update_state(
            state='FAILURE',
            meta={
                'error': f'{exc_type}: {exc_msg}',
                'task': task_func_name
            }
        )
        
        # Simply re-raise the original exception
        raise

@celery_app.task(name='utils.logo_worker.run_vector_task', bind=True, queue='vector_processing')
def run_vector_task(self, task_func_name, file_path, options):
    """Specialized task for vector processing (higher priority)"""
    return run_logo_task(self, task_func_name, file_path, options)

@celery_app.task(name='utils.logo_worker.run_social_task', bind=True, queue='social_processing')
def run_social_task(self, task_func_name, file_path, options):
    """Specialized task for social media processing"""
    return run_logo_task(self, task_func_name, file_path, options)

# Performance monitoring
@celery_app.task(name='utils.logo_worker.monitor_performance')
def monitor_performance():
    """Monitor Celery performance metrics"""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        reserved = inspect.reserved()
        
        if stats:
            total_workers = len(stats)
            total_processes = sum(worker.get('pool', {}).get('processes', 0) for worker in stats.values())
            
            logger.info(f'📊 Celery Performance: {total_workers} workers, {total_processes} processes')
            logger.info(f'📊 Active tasks: {len(active) if active else 0}')
            logger.info(f'📊 Reserved tasks: {len(reserved) if reserved else 0}')
            
            return {
                'workers': total_workers,
                'processes': total_processes,
                'active_tasks': len(active) if active else 0,
                'reserved_tasks': len(reserved) if reserved else 0
            }
    except Exception as e:
        logger.error(f'❌ Performance monitoring failed: {str(e)}')
        return None 