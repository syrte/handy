from __future__ import division, print_function, absolute_import
from collections import Mapping, Iterable
from functools import wraps
import traceback
import sys
import gc
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

__all__ = ['unpack_args', 'callback_gc', 'catch_exception', 'full_traceback',
           'print_flush', 'timeout']


def unpack_args(func):
    @wraps(func)
    def wrapper(args):
        if isinstance(args, Mapping):
            return func(**args)
        elif isinstance(args, Iterable):
            return func(*args)
        else:
            return func(args)
    return wrapper


def callback_gc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        gc.collect()
        return res
    return wrapper


def catch_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "Failed:\n  {}(*{}, **{})\n{}".format(
                func.__name__, args, kwargs, traceback.format_exc())
            print(msg)
            return e
    return wrapper


def full_traceback(func):
    """
    Seems to not not necessary in Python 3
    http://stackoverflow.com/a/29442282
    http://bugs.python.org/issue13831
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)
    return wrapper


def print_flush(*args, **kwargs):
    flush = kwargs.pop('flush', True)
    print(*args, **kwargs)
    file = kwargs.get('file', sys.stdout)
    if flush and file is not None:
        file.flush()


class timeout:
    """
    Handling timeout.

    Note signal.signal can only be called from the main thread.
    If not the case, should use the decorator mode with `thread=True`.

    Examples
    --------
        import time

        # context mode
        with timeout(seconds=3):
            time.sleep(4)

        # decorator mode
        @timeout()
        def func():
            time.sleep(4)
        func()

    Reference
    ---------
    https://stackoverflow.com/a/22348885/2144720
    https://stackoverflow.com/a/2282656/2144720
    """

    def __init__(self, seconds=1, exception=TimeoutError('Timeout.'),
                 thread=False):
        """
        thread :
            True: use concurrent.futures.ThreadPoolExecutor
            False : use signal.signal
        """
        self.seconds = seconds
        if isinstance(exception, Exception):
            self.exception = exception
        else:
            self.exception = TimeoutError(exception)
        self.thread = thread

    def handler(self, signum, frame):
        raise self.exception

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.thread:
                with self:
                    return func(*args, **kwargs)
            else:
                with ThreadPoolExecutor(1) as pool:
                    res = pool.submit(func, *args, **kwargs)
                    res.set_exception(self.exception)
                    return res.result(self.seconds)
        return wrapper
