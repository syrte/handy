from .stats import *
from .interpolate import *
from .optimize import *
from .misc import *
from .funcs import *
from .cython import *

from .hist import *
from .scatter import *
from .line import *
from .axes import *
from .helper import *

__all__ = []
for mod in [stats, interpolate, optimize, misc, funcs, cython,
            hist, scatter, line, axes, helper]:
    __all__.extend(mod.__all__)
del mod
