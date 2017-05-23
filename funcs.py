from __future__ import division, print_function, absolute_import
from collections import Mapping, Iterable
from functools import wraps
import traceback
import sys
import gc

__all__ = ['unpack_args', 'callback_gc', 'catch_exception', 'full_traceback',
           'print_flush']


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
