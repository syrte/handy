from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = ['try_minimize', 'findroot']


def try_minimize(func, guess, args=(), show=True, **kwds):
    '''Minimization of scalar function of one or more variables.
    See the docstring of `scipy.optimize.minimize`.

    Example
    -------
    from scipy.optimize import rosen
    res = try_minimize(rosen, [0.5, 0.5])
    '''
    from scipy.optimize import minimize
    from numpy import array2string
    from time import clock
    kwds.pop('method', None)

    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
               'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
    results = []
    for method in methods:
        try:
            time = clock()
            res = minimize(func, guess, args=args, method=method, **kwds)
            res.time = clock() - time
            res.method = method
            results.append(res)
        except ValueError as err:
            if show:
                print("{:>12s}: {}".format(method, err))
            continue

    results.sort(key=lambda res: res.fun)
    if show:
        print("---------------------------------------------")
        for res in results:
            formatter = {'all': (lambda x: "%9.3g" % x)}
            x = array2string(res.x, formatter=formatter, separator=',')
            out = (res.method, str(res.success), res.fun, x, res.time)
            print("{:>12s}: {:5s}  {:10.4g}  {}  {:.1e}".format(*out))
    return results[0]


def findroot(y0, x, y):
    """
    find multiple roots.
    y0: scalar
    x: 1D array
    y: function or 1D array
    """
    x = np.asarray(x)
    assert x.ndim == 1

    if callable(y):
        y = y(x)
    y = np.asarray(y)
    assert x.shape == y.shape

    ix1 = np.diff(y - y0 >= 0).nonzero()[0]
    ix2 = ix1 + 1
    x1, x2 = x[ix1], x[ix2]
    y1, y2 = y[ix1], y[ix2]
    x0 = (y0 - y1) / (y2 - y1) * (x2 - x1) + x1
    return x0
