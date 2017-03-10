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
                directives={}, boundscheck=True, wraparound=True,
                lib_dir=os.path.join(get_cython_cache_dir(), 'snippet'),
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
    directives : dict
        Cython compiler directives, e.g. `directives={'nonecheck':True}`
        http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives
    boundscheck, wraparound : bool
        Cython compiler directives, will be overrided by `directives`.
        Set False for better performance with arrays operations.
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
    Usage:
        cythonmagic('''
            def f(x):
                return 2.0*x
        ''')

    Export the names from compiled module:
        cythonmagic(code, globals())

    Get better performance (with risk) with arrays:
        cythonmagic(code,
                    boundscheck=False, wraparound=False,
                    )
    or set a header comment at the top of the code
        code = '''
        # cython: boundscheck=False, wraparound=False
        ...
        '''

    Compile OpenMP codes with gcc:
        cythonmagic(openmpcode, 
                    extra_compile_args=['-fopenmp'], 
                    extra_link_args=['-fopenmp'],
                    )

    Suppress all warnings (not recommended) with gcc:
        cythonmagic(code, 
                    quiet=True, extra_compile_args=['-w'],
                   )

    Use icc to compile:
        import os
        os.environ['CC'] = 'icc'
        cythonmagic(code)

    References
    ----------
    https://github.com/cython/cython/blob/master/Cython/Build/IpythonMagic.py
    https://github.com/cython/cython/blob/master/Cython/Build/Inline.py
    """
    code = strip_common_indent(to_unicode(code))

    old_directives = directives
    directives = dict(boundscheck=boundscheck, wraparound=wraparound)
    directives.update(old_directives)

    # generate module name
    key = (code, directives, args, sys.version_info, sys.executable,
           Cython.__version__)
    if force:
        # Force a new module name by adding the current time into hash
        key += time.time(),
    hashed = hashlib.md5(str(key).encode('utf-8')).hexdigest()
    module_name = "_cython_magic_" + hashed

    build_extension = _get_build_extension()
    so_ext = build_extension.get_ext_filename('')

    # module path
    # lib_dir = os.path.join(get_cython_cache_dir(), 'snippet')
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

        extension = Extension(name=module_name, **args)
        extensions = cythonize([extension],
                               force=force,
                               quiet=quiet,
                               compiler_directives=directives,
                               )

        # make the build_dir to be the same as lib_dir
        # note build_dir = os.path.join(temp_dir, lib_dir.strip('/'))
        temp_dir = '/' if os.path.isabs(lib_dir) else ''

        build_extension.extensions = extensions
        build_extension.build_temp = temp_dir
        build_extension.build_lib = lib_dir
        build_extension.run()

    module = imp.load_dynamic(module_name, module_path)
    if export is not None:
        _export_all(module.__dict__, export)
    return module
