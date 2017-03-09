from __future__ import absolute_import, print_function

import imp
import io
import os
import sys
import time
import hashlib

from distutils import dir_util
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext

import Cython
from Cython.Utils import get_cython_cache_dir
from Cython.Build import cythonize
from Cython.Build.Inline import to_unicode, strip_common_indent

__all__ = ['cythomagic']


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


def _get_build_extension():
    # prevents distutils from skipping re-creation of dirs
    # that have been removed
    dir_util._path_created.clear()

    dist = Distribution()
    config_files = dist.find_config_files()
    try:
        config_files.remove('setup.cfg')
    except ValueError:
        pass
    dist.parse_config_files(config_files)

    build_extension = build_ext(dist)
    build_extension.finalize_options()
    return build_extension


def cythomagic(code, export=None, force=False,
               boundscheck=True, wraparound=True,
               **args):
    """Compile a code snippet in string.
    The contents of the code are written to a `.pyx` file in the
    cython cache directory using a filename with the hash of the
    code. This file is then cythonized and compiled. 

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
    boundscheck, wraparound : bool
        Cython compiler directives.
        http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives        
    **args :
        Arguments for `distutils.core.Extension`, including
            name, sources, define_macros, undef_macros, 
            include_dirs, library_dirs, runtime_library_dirs, 
            libraries, extra_compile_args, extra_link_args, 
            extra_objects, export_symbols, depends, language
        https://docs.python.org/2/distutils/apiref.html#distutils.core.Extension

    Examples
    --------
    Usage:
        cythonmagic('''
            def f(x):
                return 2.0*x
        ''')

    To compile OpenMP codes, pass the required  `extra_compile_args`
    and `extra_link_args`. For example with gcc:
        cythonmagic(openmpcode, 
                    extra_compile_args=['-fopenmp'], 
                    extra_link_args=['-fopenmp'])

    References
    ----------
    https://github.com/cython/cython/blob/master/Cython/Build/IpythonMagic.py
    https://github.com/cython/cython/blob/master/Cython/Build/Inline.py
    """
    code = strip_common_indent(to_unicode(code))

    key = code, args, sys.version_info, sys.executable, Cython.__version__
    if force:
        # Force a new module name by adding the current time into hash
        key += time.time(),
    hashed = hashlib.md5(str(key).encode('utf-8')).hexdigest()
    module_name = "_cython_magic_" + hashed

    build_extension = _get_build_extension()
    so_ext = build_extension.get_ext_filename('')

    lib_dir = os.path.join(get_cython_cache_dir(), 'snippet')
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    module_path = os.path.join(lib_dir, module_name + so_ext)
    pyx_file = os.path.join(lib_dir, module_name + '.pyx')

    if force or not os.path.isfile(module_path):
        with io.open(pyx_file, 'w', encoding='utf-8') as f:
            f.write(code)

        args['sources'] = [pyx_file] + args.get('sources', [])
        if 'numpy' in code:
            import numpy
            args['include_dirs'] = ([numpy.get_include()] +
                                    args.get('include_dirs', []))

        directives = {'boundscheck': boundscheck,
                      'wraparound': wraparound,
                      }

        extension = Extension(name=module_name, **args)
        extensions = cythonize([extension],
                               force=force,
                               quiet=False,
                               compiler_directives=directives,
                               )

        build_extension.extensions = extensions
        build_extension.build_temp = lib_dir
        build_extension.build_lib = lib_dir
        build_extension.run()

    module = imp.load_dynamic(module_name, module_path)
    if export is not None:
        _export_all(module.__dict__, export)
    return module
