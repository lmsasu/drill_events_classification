import logging
import time
from functools import wraps


def timeit(func):
    """Log execution time for a function using the module logger."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger = logging.getLogger(func.__module__)
            logger.info("%s executed in %.3f seconds", func.__name__, elapsed)

    return wrapper
