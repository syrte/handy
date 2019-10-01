import GPy
import numpy as np
from scipy.stats import norm, chi2

__all__ = ['robust_GP']


def robust_gp_old(X, Y, nsigs=np.repeat(2, 5), callback=None, callback_args=(),
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


def robust_GP(X, Y, alpha1=0.50, alpha2=0.95, alpha3=0.95,
              niter0=0, niter1=10, niter2=1,
              callback=None, callback_args=(),
              **kwargs):
    """
    Robust Gaussian process for data with outliers.

    Three steps:
        1. contraction
        2. refinement
        3. outlier detection

    Parameters
    ----------
    X: array shape (n, p)
    Y: array shape (n, 1)
        Input data.
    alpha1, alpha2:
        Coverage fraction used in contraction step and refinement step respectively.
    alpha3:
        Outlier threshold.
    niter0:
        Extra iteration before start.
    niter1, niter2:
        Maximum iteration allowed in contraction step and refinement step respectively.
    callback: callable
        Function for checking the iteration process. It takes
        the GPRegression object `gp`, consistency factor and iteration number `i` as input
        e.g.
            callback=lambda gp, c, i: print(i, gp.num_data, gp.param_array)
        or
            callback=lambda gp, c, i: gp.plot()
    callback_args:
        Extra parameters for callback.
    **kwargs:
        GPy.core.GP parameters.

    Returns
    -------
    gp:
        GPy.core.GP object.
    consistency:
        Consistency factor.
    ix_out:
        Boolean index for outliers.
    """
    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")

    kwargs.setdefault('likelihood', GPy.likelihoods.Gaussian(variance=1.0))
    kwargs.setdefault('kernel', GPy.kern.RBF(X.shape[1]))
    kwargs.setdefault('name', 'Robust GP regression')

    gp = GPy.core.GP(X, Y, **kwargs)
    gp.optimize()
    consistency = 1

    if callback is not None:
        callback(gp, consistency, 0, *callback_args)

    ix_old = None
    niter1 = niter0 + niter1

    # contraction step
    for i in range(niter1):
        mean, var = gp.predict(X)
        d = (Y - mean) / var**0.5

        alpha_ = alpha1 + (1 - alpha1) * (max(niter0 - 1 - i, 0) / niter0)
        h = min(np.ceil(n * alpha_), n) - 1
        d_th = np.partition(d, h)[h]
        eta_sq1 = chi2(p).ppf(alpha_)
        ix_sub = d <= d_th

        if (i > niter0) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize()
        consistency = alpha_ / chi2(p + 2).cdf(eta_sq1)

        if callback is not None:
            callback(gp, consistency, i + 1, *callback_args)

    # refinement step
    for i in range(niter1, niter1 + niter2):
        mean, var = gp.predict(X)
        d = (Y - mean) / var**0.5

        eta_sq2 = chi2(p).ppf(alpha2)
        ix_sub = d <= (eta_sq2 * consistency)**0.5

        if (i > niter1) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize()
        consistency = alpha2 / chi2(p + 2).cdf(eta_sq2)

        if callback is not None:
            callback(gp, consistency, i + 1, *callback_args)

    # outlier detection
    score = d / consistency**0.5

    eta_sq3 = chi2(p).ppf(alpha3)
    ix_out = score > eta_sq3**0.5

    return gp, consistency, score, ix_out
