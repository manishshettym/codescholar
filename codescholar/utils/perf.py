from functools import wraps
import time


def perftimer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = (end_time - start_time) / 60
        print(f"[{func.__name__}] took {total_time:.4f} mins")
        return result

    return timeit_wrapper
