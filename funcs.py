from __future__ import division, print_function, absolute_import
from collections.abc import Mapping, Iterable
from functools import wraps
import traceback
import sys
import gc
import signal
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError

__all__ = ['unpack_args', 'callback_gc', 'catch_exception', 'full_traceback',
           'print_flush', 'timeout']


def get_default_args(func):
    """
    cf. https://stackoverflow.com/a/12627202/
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


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

    Note that `signal.signal` can only be called from the main thread
    in Unix-like system, and can not be nested,
    i.e. when timeout is called under another timeout, the first one will overridden.

    To avoid above issues, one should use the decorator mode with `thread=True`.

    Examples
    --------
        import time

        # context mode
        with timeout(seconds=1):
            time.sleep(4)

        # decorator mode
        @timeout(1)
        def func():
            time.sleep(4)
        func()

    Reference
    ---------
    https://stackoverflow.com/a/22348885/ for decorator
    https://stackoverflow.com/a/2282656/ for context
    https://stackoverflow.com/a/11901541/ for `signal.setitimer`
    """

    def __init__(self, seconds=1, exception=TimeoutError('Timeout.'),
                 thread=True):
        """
        seconds :
            Note `signal.alarm`
        thread :
            If True, `concurrent.futures.ThreadPoolExecutor` is used,
            otherwise `signal.signal` is used.
            Only takes effect in decorator mode.
        """
        self.seconds = seconds
        self.thread = thread

        if isinstance(exception, type) and issubclass(exception, Exception):
            self.exception = exception('Timeout.')
        elif isinstance(exception, Exception):
            self.exception = exception
        else:
            self.exception = TimeoutError(exception)

    def handler(self, signum, frame):
        raise self.exception

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handler)
        # signal.alarm(self.seconds)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.thread:
                with ThreadPoolExecutor(1) as pool:
                    res = pool.submit(func, *args, **kwargs)
                    res.set_exception(self.exception)
                    return res.result(self.seconds)
            else:
                with self:
                    return func(*args, **kwargs)

        return wrapper
