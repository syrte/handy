from __future__ import absolute_import, print_function

import imp
import re
import io
import os
import sys
import time
import hashlib
import inspect
import contextlib
from distutils.core import Extension

import Cython
from Cython.Utils import get_cython_cache_dir
from Cython.Build import cythonize
from Cython.Build.Inline import to_unicode, strip_common_indent
from Cython.Build.Inline import _get_build_extension


__all__ = ['cythonmagic']


def _so_ext():
    """get extension for the compiled library.
    """
    if not hasattr(_so_ext, 'ext'):
        _so_ext.ext = _get_build_extension().get_ext_filename('')
    return _so_ext.ext


def _append_args(dic, key, value):
    dic[key] = [value] + dic.get(key, [])


def _export_all(source, target):
    """import all variables from one namespace to another.
    source, target must be dict objects.
    import will skip names stats with underscore.
    """
    if '__all__' in source:
        keys = source['__all__']
    else:
        keys = [k for k in source if not k.startswith('_')]

    for k in keys:
        try:
            target[k] = source[k]
        except KeyError:
            msg = "'module' object has no attribute '%s'" % k
            raise AttributeError(msg)


def _update_flag(code, args, smart=True):
    """Update compiler options for numpy and openmp.
    Helper function for cythonmagic.
    """
    numpy = args.pop('numpy', None)
    openmp = args.pop('openmp', None)

    if numpy is None and smart:
        reg_numpy = re.compile("""
            ^\s* cimport \s+ numpy |
            ^\s* from \s+ numpy \s+ cimport
            """, re.M | re.X)
        numpy = reg_numpy.match(code)

    if openmp is None and smart:
        reg_openmp = re.compile("""
            ^\s* c?import \s+cython\.parallel |
            ^\s* from \s+ cython\.parallel \s+ c?import |
            ^\s* from \s+ cython \s+ c?import \s+ parallel
            """, re.M | re.X)
        openmp = reg_openmp.match(code)

    if numpy:
        import numpy
        _append_args(args, 'include_dirs', numpy.get_include())

    if openmp:
        if hasattr(openmp, 'startswith'):
            openmp_flag = openmp  # openmp is string
        else:
            openmp_flag = '-fopenmp'
        _append_args(args, 'extra_compile_args', openmp_flag)
        _append_args(args, 'extra_link_args', openmp_flag)


def get_frame_dir(depth=0):
    """Return the source file directory of a frame in the call stack.
    """
    frame = inspect.currentframe(depth + 1)  # +1 for this function itself
    file = inspect.getabsfile(frame)
    return os.path.dirname(file)


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    source: http://stackoverflow.com/a/34333710/

    >>> with set_env(PLUGINS_DIR=u'plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True
    >>> "PLUGINS_DIR" in os.environ
    False
    """
    try:
        if environ:
            old_environ = dict(os.environ)
            os.environ.update(environ)
        yield
    finally:
        if environ:
            os.environ.clear()
            os.environ.update(old_environ)


def cython_build(name, file=None, force=False, cythonize_args={},
                 lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
                 temp_dir=os.path.join(get_cython_cache_dir(), 'inline/temp'),
                 **extension_args):
    """Build a cython extension.
    """
    if file is not None:
        _append_args(extension_args, 'sources', file)

    extension = Extension(name, **extension_args)
    extensions = cythonize([extension], force=force,
                           **cythonize_args
                           )

    build_extension = _get_build_extension()
    build_extension.extensions = extensions
    build_extension.build_lib = lib_dir
    build_extension.build_temp = temp_dir
    build_extension.run()

    #ext_file = os.path.join(lib_dir, name + _so_ext())
    #module = imp.load_dynamic(name, ext_file)
    # return module


def cythonmagic(code, export=None, name=None,
                force=False, smart=True, fast_indexing=False,
                directives={}, cimport_dirs=[], cythonize_args={},
                lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
                temp_dir=os.path.join(get_cython_cache_dir(), 'inline/temp'),
                environ={}, **args):
    """Compile a code snippet in string.
    The contents of the code are written to a `.pyx` file in the
    cython cache directory using a filename with the hash of the
    code. This file is then cythonized and compiled.

    Raw string is recommended to avoid breaking escape character
    when defining the`code`.

    Parameters
    ----------
    code : str
        The code to compile.
        It can be a file path, but must start with "./" or "/".
    export : dict
        Export the variables from the compiled module to a dict.
        Set `export=globals()` to export into the current module.
    name : str, optional
        Name of compiled module. If not given, it will be generated
        automatically by hash of the code and options.
    force : bool
        Force the compilation of a new module,
        even if the source has been previously compiled.
    smart : bool
        If True, numpy and openmp will be auto-detected.
    fast_indexing : bool
        If True, `boundscheck` and `wraparound` are turned off
        for better arrays indexing performance (at cost of safety).
        This setting will be overridden by `directives`.
    directives : dict
        Cython compiler directives, e.g.
        `directives={'nonecheck':True, 'language_level':2}`
        http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives
    cimport_dirs : list
        Directories for finding cimported modules.
    cythonize_args : dict
        Arguments for `Cython.Build.cythonize`, including
            quiet, language, build_dir, output_file, language_level,
            include_path, compiler_directives, etc.
        Can override `directives` and `cimport_dirs` above.
    environ : dict
        Temporary environment variables for compilation.
    lib_dir : str
        Directory to put the compiled module.
    temp_dir : str
        Directory to put the temporary files.
    **args :
        Arguments for `distutils.core.Extension`, including
            name, sources, define_macros, undef_macros,
            include_dirs, library_dirs, runtime_library_dirs,
            libraries, extra_compile_args, extra_link_args,
            extra_objects, export_symbols, depends, language
        https://docs.python.org/2/distutils/apiref.html#distutils.core.Extension

    Examples
    --------
    Basic usage:
        code = r'''
        def f(x):
            return 2.0*x
        '''
        m = cythonmagic(code)
        m.f(1)

    Export the names from compiled module:
        cythonmagic(code, globals())

    Get better performance (with risk) with arrays:
        cythonmagic(code, fast_indexing=True)

    Compile OpenMP codes with gcc:
        cythonmagic(openmpcode,
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    )
        # use '-openmp' or '-qopenmp' (>15.0) for Intel
        # use '/openmp' for Microsoft Visual C++ Compiler

    Use icc to compile:
        cythonmagic(code, environ={'CC':'icc'})

    Suppress prompts in compiling:
        cythonmagic(code, extra_compile_args=['-w'],
            cythonize_args={'quiet':True})

    Set directory for searching cimports (*.pxd):
        cythonmagic(code, cimport_dirs=[custum_path]})
        # or
        cythonmagic(code, cythonize_args={'include_path': [custum_path]})

    The cython `directives` and distutils `args` can also be
    set in a directive comment at the top of the code, e.g.:
        # cython: boundscheck=False, wraparound=False
        # distutils: extra_compile_args = -fopenmp
        # distutils: extra_link_args = -fopenmp
        ...code...

    References
    ----------
    https://github.com/cython/cython/blob/master/Cython/Build/IpythonMagic.py
    https://github.com/cython/cython/blob/master/Cython/Build/Inline.py
    """
    # assume all paths are relative to cur_dir
    cur_dir = get_frame_dir(depth=1)  # the caller frame's directory
    lib_dir = os.path.join(cur_dir, lib_dir)
    temp_dir = os.path.join(cur_dir, temp_dir)

    # check if `code` is a file or a string
    if code.endswith('.pyx') and (code.startswith('/') or
                                  code.startswith('./')):
        pyx_file = os.path.join(cur_dir, code)
        code = io.open(pyx_file, 'r', encoding='utf-8').read()
    else:
        pyx_file = None
        code = strip_common_indent(to_unicode(code))

    # setting the arguments
    if fast_indexing:
        directives = directives.copy()
        directives.setdefault('boundscheck', False)
        directives.setdefault('boundscheck', False)

    if pyx_file is None:
        cimport_dirs = cimport_dirs + [cur_dir]
    else:
        cimport_dirs = cimport_dirs + [os.path.dirname(pyx_file)]

    cythonize_args = cythonize_args.copy()
    cythonize_args.setdefault('compiler_directives', directives)
    cythonize_args.setdefault('include_path', cimport_dirs)

    # module name
    if name is None:
        key = (code, cythonize_args, args, environ, os.environ,
               sys.executable, sys.version_info, Cython.__version__)
        if force:
            # force a new module name by adding the current time into hash
            key += (time.time(),)
        hashed = hashlib.md5(str(key)).hexdigest()
        ext_name = "_cython_magic_{}".format(hashed)
    else:
        ext_name = name

    # module path
    ext_file = os.path.join(lib_dir, ext_name + _so_ext())

    # build
    if force or not os.path.isfile(ext_file):
        if pyx_file is None:
            pyx_file = os.path.join(lib_dir, ext_name + '.pyx')
            if not os.path.isdir(lib_dir):
                os.makedirs(lib_dir)
            with io.open(pyx_file, 'w', encoding='utf-8') as f:
                f.write(code)

        with set_env(**environ):
            _update_flag(code, args, smart)
            cython_build(ext_name, file=pyx_file, force=force,
                         cythonize_args=cythonize_args,
                         lib_dir=lib_dir, temp_dir=temp_dir,
                         **args)

    # import
    module = imp.load_dynamic(ext_name, ext_file)
    if export is not None:
        _export_all(module.__dict__, export)
    return module
