import stats
import interpolate
import optimize
import misc
import hist
import scatter

from stats import *
from interpolate import *
from optimize import *
from misc import *
from hist import *
from scatter import *

__all__ = []
for mod in [stats, interpolate, optimize, misc, hist, scatter]:
    __all__.extend(mod.__all__)

del mod
del stats
del interpolate
del optimize
del misc
del hist
del scatter
