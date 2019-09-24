import GPy
import numpy as np
from scipy.stats import norm, chi2

__all__ = ['robust_gp']


def robust_gp(X, Y, nsigs=np.repeat(2, 5), callback=None, callback_args=(),
              **kwargs):
    """
    Robust Gaussian process for data with outliers.

    Parameters
    ----------
    X: array shape (n, p)
    Y: array shape (n, 1)
        Input data.
    nsigs: array shape (niter,)
        List of n-sigma for iterations, should be a decreasing list.
        Setting the last several n-sigma to be the same can give better
        self-consistency.
        Default: [2, 2, 2, 2, 2]
        Alternative: 2**np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0])
    callback: callable
        Function for checking the iteration process. It takes
        the iteration number `i` and GPRegression object `gp` as input
        e.g.
            callback=lambda gp, i: print(i, gp.num_data, gp.param_array)
        or
            callback=lambda gp, i: gp.plot()
    callback_args:
        Extra parameters for callback.
    **kwargs:
        GPy.models.GPRegression parameters.

    Returns
    -------
    gp:
        GPy.models.GPRegression object.
    """
    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")
    if (np.asarray(nsigs) <= 0).any():
        raise ValueError("nsigs should be positive array.")
    if (np.diff(nsigs) > 0).any():
        raise ValueError("nsigs should be decreasing array.")

    gp = GPy.models.GPRegression(X, Y, **kwargs)
    gp.optimize()
    if callback is not None:
        callback(gp, 0, *callback_args)

    niter = len(nsigs)
    for i in range(niter):
        mean, var = gp.predict(X)
        if i > 0:
            # reference: Croux & Haesbroeck 1999
            alpha = 2 * norm.cdf(nsigs[i - 1]) - 1
            consistency_factor = alpha / chi2(p + 2).cdf(chi2(p).ppf(alpha))
            var = var * consistency_factor
        width = var**0.5 * nsigs[i]
        ix = ((Y >= mean - width) & (Y <= mean + width)).ravel()

        if i == 0:
            ix_old = ix
        elif (nsigs[i - 1] == nsigs[-1]) and (ix == ix_old).all():
            break
        else:
            ix_old = ix

        gp = GPy.models.GPRegression(X[ix], Y[ix], **kwargs)
        gp.optimize()
        if callback is not None:
            callback(gp, i + 1, *callback_args)

    return gp
