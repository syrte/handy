from __future__ import absolute_import, print_function

import imp
import re
import io
import os
import sys
import hashlib
import inspect
import contextlib
from distutils.core import Extension

import Cython
from Cython.Utils import captured_fd
from Cython.Utils import get_cython_cache_dir
from Cython.Build import cythonize
from Cython.Build.Inline import to_unicode, strip_common_indent
from Cython.Build.Inline import _get_build_extension


__all__ = ['cythonmagic']


def _so_ext():
    """Get extension for the compiled library.
    """
    if not hasattr(_so_ext, 'ext'):
        _so_ext.ext = _get_build_extension().get_ext_filename('')
    return _so_ext.ext


def _append_args(dic, key, value):
    dic[key] = [value] + dic.get(key, [])


def _extend_args(dic, key, value):
    dic[key] = value + dic.get(key, [])


def _export_all(source, target):
    """Import all variables from one namespace to another.
    Arguments must be dict-like objects.
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


def join_path(a, b):
    """Join and normalize two paths.
    """
    return os.path.normpath(os.path.join(a, os.path.expanduser(b)))


def get_frame_dir(depth=0):
    """Return the source file directory of a frame in the call stack.
    """
    if hasattr(sys, "_getframe"):
        frame = sys._getframe(depth + 1)  # +1 for this function itself
    else:
        raise NotImplementedError("Support CPython only.")
    file = inspect.getabsfile(frame)
    return os.path.dirname(file)


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the environment variables.
    source: http://stackoverflow.com/a/34333710/

    Examples
    --------
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


@contextlib.contextmanager
def _suppress_output(quiet=True):
    """Suppress any output/error/warning in compiling
    if quiet is True and no exception raised.
    """
    try:
        # redirect the streams to defaults (for jupyter notebook)
        old_stream = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

        get_outs = get_errs = lambda: None
        with captured_fd(1) as get_outs:
            with captured_fd(2) as get_errs:
                yield

    except Exception:
        quiet = False
        raise

    finally:
        sys.stdout, sys.stderr = old_stream

        if not quiet:
            outs, errs = get_outs(), get_errs()
            if outs:
                print("Compiler Output\n===============",
                      outs.decode('utf8'), sep='\n', file=sys.stdout)
            if errs:
                print("Compiler Error/Warning\n======================",
                      errs.decode('utf8'), sep='\n', file=sys.stderr)


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
        numpy = reg_numpy.search(code)

    if openmp is None and smart:
        reg_openmp = re.compile("""
            ^\s* c?import \s+cython\.parallel |
            ^\s* from \s+ cython\.parallel \s+ c?import |
            ^\s* from \s+ cython \s+ c?import \s+ parallel
            """, re.M | re.X)
        openmp = reg_openmp.search(code)

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


def cython_build(name, file=None, force=False, quiet=True, cythonize_args={},
                 lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
                 temp_dir=os.path.join(get_cython_cache_dir(), 'inline/temp'),
                 **extension_args):
    """Build a cython extension.
    """
    if file is not None:
        _append_args(extension_args, 'sources', file)

    with _suppress_output(quiet=quiet):
        extension = Extension(name, **extension_args)
        extensions = cythonize([extension], force=force, **cythonize_args)

        build_extension = _get_build_extension()
        build_extension.extensions = extensions
        build_extension.build_lib = lib_dir
        build_extension.build_temp = temp_dir
        build_extension.run()

        # ext_file = os.path.join(lib_dir, name + _so_ext())
        # module = imp.load_dynamic(name, ext_file)
        # return module


def cythonmagic(code, export=None, name=None, force=False,
                quiet=True, smart=True, fast_indexing=False,
                directives={}, cimport_dirs=[], cythonize_args={},
                lib_dir=os.path.join(get_cython_cache_dir(), 'inline/lib'),
                temp_dir=os.path.join(get_cython_cache_dir(), 'inline/temp'),
                environ={}, **extension_args):
    """Compile a code snippet in string.
    The contents of the code are written to a `.pyx` file in the
    cython cache directory using a filename with the hash of the
    code. This file is then cythonized and compiled.

    Parameters
    ----------
    code : str
        The code to compile.
        It can be a file path, but must start with "./", "/" or "~".
        String like "import abc.pyx" or "a=1;b=a.pyx" are assumed to
        be code snippet.
    export : dict
        Export the variables from the compiled module to a dict.
        `export=globals()` is equivalent to `from module import *`.
    name : str, optional
        Name of compiled module. If not given, it will be generated
        automatically by hash of the code and options.
    force : bool
        Force the compilation of a new module, even if the source
        has been previously compiled.
        Important: cythonmagic will not check the modification time
        of .pyx file or other dependences. If the source files are 
        updated, you should manually set `force=True`.
        No worry if you only compile an anonymous code snippet.
    quiet : bool
        Suppress compiler's outputs/warnings.
    smart : bool
        If True, numpy and openmp will be auto-detected from the code.
    fast_indexing : bool
        If True, `boundscheck` and `wraparound` are turned off
        for better array indexing performance (at cost of safety).
        This setting can be overridden by `directives`.
    directives : dict
        Cython compiler directives, including
            binding, boundscheck, wraparound, initializedcheck, nonecheck,
            overflowcheck, overflowcheck.fold, embedsignature, cdivision, cdivision_warnings,
            always_allow_keywords, profile, linetrace, infer_types, language_level, etc.
        Ref http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
        This setting can be overridden by `cythonize_args['compiler_directives']`.
    cimport_dirs : list
        Directories for finding cimport modules (.pxd files).
        This setting can be overridden by `cythonize_args['include_path']`.
    cythonize_args : dict
        Arguments for `Cython.Build.cythonize`, including
            aliases, quiet, force, language, annotate, build_dir, output_file,
            include_path, compiler_directives, etc.
        Ref http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
    environ : dict
        Temporary environment variables for compilation.
    lib_dir : str
        Directory to put the compiled module.
    temp_dir : str
        Directory to put the temporary files.
    **extension_args :
        Arguments for `distutils.core.Extension`, including
            name, sources, define_macros, undef_macros,
            include_dirs, library_dirs, runtime_library_dirs,
            libraries, extra_compile_args, extra_link_args,
            extra_objects, export_symbols, depends, language
        Ref https://docs.python.org/2/distutils/apiref.html#distutils.core.Extension

    Examples
    --------
    Basic usage:
        code = r'''
        def f(x):
            return 2.0 * x
        '''
        m = cythonmagic(code)
        m.f(1)
    Raw string is recommended to avoid breaking escape character.

    Export the names from compiled module:
        cythonmagic(code, globals())

    Get better performance (with risk) with arrays:
        cythonmagic(code, fast_indexing=True)

    Compile OpenMP codes with gcc:
        cythonmagic(openmpcode, openmp='-fopenmp')
        # or equivalently
        cythonmagic(openmpcode,
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    )
        # use '-openmp' or '-qopenmp' (>=15.0) for Intel
        # use '/openmp' for Microsoft Visual C++ Compiler
        # use '-fopenmp=libomp' for Clang

    Use icc to compile:
        cythonmagic(code, environ={'CC':'icc', 'LDSHARED':'icc -shared'})
    Ref https://software.intel.com/en-us/articles/thread-parallelism-in-cython

    Set directory for searching cimport (.pxd file):
        cythonmagic(code, cimport_dirs=[custum_path]})
        # or equivalently
        cythonmagic(code, cythonize_args={'include_path': [custum_path]})
    Try setting `cimport_dirs=sys.path` if Cython can not find installed
    cimport module.

    The cython `directives` and distutils `extension_args` can also be
    set in a directive comment at the top of the code, e.g.:
        # cython: boundscheck=False, wraparound=False, cdivision=True
        # distutils: extra_compile_args = -fopenmp
        # distutils: extra_link_args = -fopenmp
        ...code...

    References
    ----------
    https://github.com/cython/cython/blob/master/Cython/Build/IpythonMagic.py
    https://github.com/cython/cython/blob/master/Cython/Build/Inline.py
    """
    # assume all paths are relative to cur_dir
    cur_dir = get_frame_dir(depth=1)  # the caller frame's directory!!
    lib_dir = join_path(cur_dir, lib_dir)
    temp_dir = join_path(cur_dir, temp_dir)

    # check if `code` presents .pyx or .py file
    reg_pyx = re.compile(r"^ ( ~ | [\.]? [/\\] | [a-zA-Z]:) .* \.pyx? $ | "
                         r"^ [^\s=;]+ \.pyx? $", re.X | re.S)
    if reg_pyx.match(code):
        pyx_file = join_path(cur_dir, code)
        code = io.open(pyx_file, 'r', encoding='utf-8').read()
        name = os.path.splitext(os.path.basename(pyx_file))[0]
    else:
        pyx_file = None
        code = strip_common_indent(to_unicode(code))

    # setting the arguments
    directives = directives.copy()
    if fast_indexing:
        directives.setdefault('boundscheck', False)
        directives.setdefault('wraparound', False)
    directives.setdefault('embedsignature', True)  # recommended setting

    if pyx_file is None:
        cimport_dirs = cimport_dirs + [cur_dir]
    else:
        cimport_dirs = cimport_dirs + [os.path.dirname(pyx_file)]

    cythonize_args = cythonize_args.copy()
    cythonize_args.setdefault('compiler_directives', directives)
    cythonize_args.setdefault('include_path', cimport_dirs)

    # module name
    if name is None:
        key = (code, cythonize_args, extension_args, environ, os.environ,
               sys.executable, sys.version_info, Cython.__version__)
        key = u"{}".format(key).encode('utf-8')   # for 2, 3 compatibility
        hashed = hashlib.md5(key).hexdigest()
        ext_name = "_cython_magic_{}".format(hashed)
    else:
        ext_name = name

    # module path
    ext_file = os.path.join(lib_dir, ext_name + _so_ext())

    if force and os.path.isfile(ext_file):
        os.remove(ext_file)  # dangerous?

    # build
    if force or not os.path.isfile(ext_file):
        if not os.path.isdir(lib_dir):
            os.makedirs(lib_dir)
        if pyx_file is None:
            pyx_file = os.path.join(lib_dir, ext_name + '.pyx')
            with io.open(pyx_file, 'w', encoding='utf-8') as f:
                f.write(code)

        with set_env(**environ):
            _update_flag(code, extension_args, smart)
            cython_build(ext_name, file=pyx_file, force=force,
                         quiet=quiet, cythonize_args=cythonize_args,
                         lib_dir=lib_dir, temp_dir=temp_dir,
                         **extension_args)

    # import
    module = imp.load_dynamic(ext_name, ext_file)
    if export is not None:
        _export_all(module.__dict__, export)
    return module
