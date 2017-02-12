from __future__ import division, print_function, absolute_import
from collections import Mapping, Iterable
from functools import wraps
import sys
import gc

__all__ = ['unpack_args', 'callback_gc', 'catch_exception', 'print_flush']


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
        except KeyboardInterrupt:
            raise
        except Exception as msg:
            print("failed:  %s(*%s, **%s)\nmessage: %s" %
                  (func.__name__, args, kwargs, msg))
    return wrapper


def print_flush(*args, **kwargs):
    flush = kwargs.pop('flush', True)
    print(*args, **kwargs)
    file = kwargs.get('file', sys.stdout)
    if flush and file is not None:
        file.flush()
