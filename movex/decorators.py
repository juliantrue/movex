import cv2
import functools


def rolling_frame(func):
    @functools.wraps(func)
    def wrapper(frame):
        frame = frame.to_ndarray(format="rgb24")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret = func(frame, wrapper.last_frame)
        wrapper.last_frame = frame.copy()
        return ret

    wrapper.last_frame = None
    return wrapper
