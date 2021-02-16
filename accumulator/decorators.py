import time
import functools


def collect_run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last = time.perf_counter()
        ret = func(*args, **kwargs)
        now = time.perf_counter()
        elapsed_run_time = now - last
        wrapper.samples.append(elapsed_run_time)
        return ret

    wrapper.samples = []
    return wrapper
