from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = ['try_minimize', 'findroot']


def try_minimize(func, guess, args=(), quiet=False, timeout=5, max_show=10, **kwds):
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
    from .funcs import timeout as timer

    guess = np.asarray(guess)
    kwds.pop('method', None)

    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
               'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
    results = []
    for method in methods:
        try:
            time = clock()
            if timeout > 0:
                with timer(timeout, ValueError):
                    res = minimize(func, guess, args=args, method=method, **kwds)
            else:
                res = minimize(func, guess, args=args, method=method, **kwds)
            res.time = clock() - time
            res.method = method
            results.append(res)
        except (ValueError, MemoryError, TypeError) as err:
            if not quiet:
                print("{:>12s}: {}".format(method, err))
            continue

    results.sort(key=lambda res: res.fun)
    if not quiet:
        print("---------------------------------------------")
        param_len = min(guess.size, max_show) * 10 + 1
        print("{:>12s}  {:^5s}  {:^10s}  {:^{}s}  {:^s}".format(
            "method", "OK", "fun", "x", param_len, "time"))
        for res in results:
            formatter = {'all': (lambda x: "%9.3g" % x)}
            x = array2string(res.x[:max_show], formatter=formatter, separator=',')
            out = (res.method, str(res.success), res.fun, x, res.time)
            print("{:>12s}: {:5s}  {:10.4g}  {}  {:.1e}".format(*out))
    if results:
        return results[0]
    else:
        raise ValueError('Failed.')


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


def root_safe(func, dfunc, x1, x2, rtol=1e-5, xtol=1e-8, maxit=100):
    """
    Find root for vector function in given intervals.
    Adopted from Numerical Recipe 3rd P460, rtsafe

    Input function and its first derivative,
        y = func(x), dy/dx = dfunc(x)
    where x, y both have shape (n,).
    Find roots in given intervals x1 < x < x2 for each elements, 
    therefor func(x1) * func(x2) < 0.

    Not fully optimized yet, though seems well workable.

    Examples
    --------
    def f(x):
    return x*(x-1)*(x-2)

    def j(x):
        return 3*x**2 - 6*x + 2

    import numpy as np
    x1 = np.random.rand(1000000)
    x = root_safe(f, j, x1, x1 + 1)
    """
    import numpy as np

    x1, x2 = x1.copy(), x2.copy()
    f1, f2 = func(x1), func(x2)

    if (f1 * f2 > 0).any():
        raise ValueError
    ix = (f1 > 0).nonzero()
    x1[ix], x2[ix] = x2[ix], x1[ix]

    rt = np.empty_like(x1, dtype='f4')
    ix_status = np.ones_like(rt, dtype='b1')

    ix = (f1 == 0).nonzero()
    rt[ix] = x1[ix]
    ix_status[ix] = False

    ix = (f2 == 0).nonzero()
    rt[ix] = x2[ix]
    ix_status[ix] = False

    dx = abs(x2 - x1)
    rt = 0.5 * (x1 + x2)
    f = func(rt)
    df = dfunc(rt)

    tol = np.fmax(xtol, dx * rtol)

    for i in range(maxit):
        dx_old = dx
        dx_new = f / df
        rt_new = rt - dx_new
        if_bisect = ((rt_new - x1) * (rt_new - x2) > 0) | (np.abs(dx_new) > 0.5 * dx_old)

        ix_bisect = (ix_status & if_bisect).nonzero()
        x1_, x2_ = x1[ix_bisect], x2[ix_bisect]
        dx_ = 0.5 * (x2_ - x1_)
        dx[ix_bisect] = dx_
        rt[ix_bisect] = x1_ + dx_

        ix_newton = (ix_status & ~if_bisect).nonzero()
        dx[ix_newton] = dx_new[ix_newton]
        rt[ix_newton] = rt_new[ix_newton]

        dx = np.abs(dx)
        ix_status[dx < tol] = False
        print (i, ix_status.mean())
        if ~ix_status.any():
            break

        f = func(rt)
        df = dfunc(rt)

        if_low = f < 0
        ix = (if_low).nonzero()
        x1[ix] = rt[ix]
        ix = (~if_low).nonzero()
        x2[ix] = rt[ix]
    else:
        print ("Maximum number of iterations exceeded")
    return rt
