"""
Timing utilities for performance monitoring.
"""
import time
import functools
from flask import current_app


def time_function(func_name=None):
    """
    Decorator for timing function execution.
    
    Args:
        func_name: Optional custom name for the function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            print(f"\n⏱️  [TIMER] Starting {name}...")
            current_app.logger.info(f"[TIMER] Starting {name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                print(f"✅ [TIMER] {name} completed in {duration:.2f} seconds")
                current_app.logger.info(f"[TIMER] {name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"❌ [TIMER] {name} failed after {duration:.2f} seconds: {str(e)}")
                current_app.logger.error(f"[TIMER] {name} failed after {duration:.2f} seconds: {str(e)}")
                raise
                
        return wrapper
    return decorator
