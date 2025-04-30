import multiprocessing
import os

def run_celery():
    # Set environment variables for Celery if needed
    os.environ.setdefault('PYTHONPATH', '.')
    
    # Import after setting multiprocessing method
    from celery_app import celery
    
    # Start the worker with desired concurrency
    # In case of GPU model workers, concurrency=1 is often best
    # For CPU-only workers, you can increase based on available cores
    worker_args = [
        'worker',
        '--loglevel=info',
        '--concurrency=1',  # For GPU tasks, use 1 worker per GPU
        '--pool=solo',      # Avoids issues with fork and CUDA
        '--without-gossip', # Performance optimization
        '--without-mingle', # Performance optimization
    ]
    
    celery.worker_main(worker_args)

if __name__ == "__main__":
    # Important: Set spawn method before any other imports that might use multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    run_celery()