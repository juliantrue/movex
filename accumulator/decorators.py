import time
import functools


def collect_run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(dir(func))
        print(func.accumulator)
        last = time.perf_counter()
        ret = func(*args, **kwargs)
        now = time.perf_counter()
        elapsed_run_time = now - last
        func.accumulator._watched[func.__name__].append(elapsed_run_time)
        return ret

    return wrapper
