from __future__ import absolute_import, print_function

import imp
import io
import os
import sys
import time
import hashlib

import Cython
from Cython.Utils import get_cython_cache_dir
from Cython.Build import cythonize
from Cython.Build.Inline import to_unicode, strip_common_indent
from Cython.Build.Inline import _get_build_extension

from distutils.core import Extension


__all__ = ['cythonmagic']


def _so_ext():
    """get extension for the compiled library.
    """
    if not hasattr(_so_ext, 'ext'):
        _so_ext.ext = _get_build_extension().get_ext_filename('')
    return _so_ext.ext


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


def cythonmagic(code, export=None, force=False, quiet=False,
                fast_indexing=False, directives={},
                lib_dir=os.path.join(get_cython_cache_dir(), 'inline'),
                **args):
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
    export : dict
        Export the names from the compiled module to a dict.
        Set `export=globals()` to change the current module.
    force : bool
        Force the compilation of a new module,
        even if the source has been previously compiled.
    fast_indexing : bool
        If True, `boundscheck` and `wraparound` are turned off
        for better arrays indexing performance (at cost of safety).
        This setting will be overrided by `directives`.
    directives : dict
        Cython compiler directives, e.g.
        `directives={'nonecheck':True, 'language_level':2}`
        http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives
    lib_dir : str
        Directory to put the temporary files and the compiled module.
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
        cythonmagic(code, boundscheck=False, wraparound=False,
                    )

    Compile OpenMP codes with gcc:
        cythonmagic(openmpcode,
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    )
        # use '-openmp' or '-qopenmp' (>15.0) for Intel
        # use '/openmp' for Microsoft Visual C++ Compiler

    Suppress all warnings (not recommended) with gcc:
        cythonmagic(code,
                    quiet=True, extra_compile_args=['-w'],
                   )

    Use icc to compile:
        import os
        os.environ['CC'] = 'icc'
        cythonmagic(code)

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
    if os.path.isfile(code):
        code = io.open(code, 'r', encoding='utf-8').read()
    else:
        code = strip_common_indent(to_unicode(code))

    if fast_indexing:
        directives = directives.copy()
        directives.setdefault('boundscheck', False)
        directives.setdefault('boundscheck', False)

    # generate module name
    key = (code, directives, args, sys.version_info, sys.executable,
           Cython.__version__)
    if force:
        # Force a new module name by adding the current time into hash
        key += time.time(),
    hashed = hashlib.md5(str(key).encode('utf-8')).hexdigest()
    module_name = "_cython_magic_" + hashed

    # module path
    # lib_dir = os.path.join(get_cython_cache_dir(), 'snippet')
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    module_path = os.path.join(lib_dir, module_name + _so_ext())
    pyx_file = os.path.join(lib_dir, module_name + '.pyx')

    # build
    if force or not os.path.isfile(module_path):
        with io.open(pyx_file, 'w', encoding='utf-8') as f:
            f.write(code)

        args['sources'] = [pyx_file] + args.get('sources', [])

        if 'numpy' in code:
            import numpy
            args['include_dirs'] = ([numpy.get_include()] +
                                    args.get('include_dirs', []))
        if 'prange' in code:
            openmp_flag = args.pop('omp_flag', '-fopenmp')
            args.setdefault('extra_compile_args', [openmp_flag])
            args.setdefault('extra_link_args', [openmp_flag])
        if quiet:
            compile_args = ['-w'] + args.get('extra_compile_args', [])
            args['extra_compile_args'] = compile_args

        extension = Extension(name=module_name, **args)
        extensions = cythonize([extension],
                               force=force,
                               quiet=quiet,
                               compiler_directives=directives,
                               )

        # to make the build_dir to be the same as lib_dir, set
        # temp_dir = '/' if os.path.isabs(lib_dir) else ''
        # however this may go wrong when extra `sources` is given.

        # note build_dir = os.path.join(temp_dir, source.strip('/'))
        temp_dir = os.path.join(lib_dir, 'build')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        build_extension = _get_build_extension()
        build_extension.extensions = extensions
        build_extension.build_temp = temp_dir
        build_extension.build_lib = lib_dir
        build_extension.run()

    module = imp.load_dynamic(module_name, module_path)
    module.__code__ = code
    if export is not None:
        _export_all(module.__dict__, export)
    return module
