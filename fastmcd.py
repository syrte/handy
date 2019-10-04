import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import pinvh
from scipy.stats import chi2
from handy import quantile

__all__ = ['fast_mcd']


def _fit_cov(X, weights=None, index=slice(None)):
    "Only X[index] is used for calculation"
    X1 = X[index]
    if weights is None:
        loc = np.mean(X1, axis=0)
        cov = np.cov(X1.T, bias=True)
    else:
        weights = weights[index]
        loc = np.sum(X1.T * weights, axis=1) / np.sum(weights)
        cov = np.cov(X1.T, aweights=weights, bias=True)

    cov = np.atleast_2d(cov)
    dX = X - loc
    dist = (np.dot(dX, pinvh(cov)) * dX).sum(axis=1)

    return loc, cov, dist


def _c_step(X, weights, index, h, nstep):
    """
    Parameters
    ----------
    X, weights:
        Data and weights [optional].
    index:
        initial index.
    h:
        subsample size for covariance estimation.
    nstep:
        maximum iterations.
    """
    det_best = np.inf
    for i in range(nstep):
        loc, cov, dist = _fit_cov(X, weights=weights, index=index)
        index = np.argsort(dist)[:h]
        sign, det = slogdet(cov)

        if sign <= 0 or np.isnan(det):
            break
        elif np.isclose(det, det_best):
            break
        else:
            det_best = det
    return loc, cov, det, dist, index


def fast_mcd(X, weights=None, alpha_mcd=None, alpha_wgt=0.975, exact_cut=False,
             niter=50, nsamp_trial=10, niter_trial=5):
    """
    Algorithm
    ---------
    - trials:
        run c-step for `nsamp_trial` trial subsamples of `p + 1` cases,
        iterate `niter_trial` times for each at most,
        the best result is taken as input of the following
    - raw MCD:
        run c-step with `alpha_mcd * n` cases until converge or `niter` times
    - weighted MCD:
        calculate result with `alpha_wgt` cases [optional]

    Parameters
    ----------
    X, weights:
        Data and weights [optional].
    alpha_mcd:
        The fraction of points used for MCD, default is (n + p + 1) / 2N.
    alpha_wgt:
        The weighted fraction of points used for re-weighting.
        Set None to disable re-weighting.
        Treat (1 - alpha_wgt) as outliers.
    exact_cut:
        If True, use the actual percentile of dist to estimate consistency factor.
        Otherwise, an estimation based on chi2 of alpha is used.
    niter:
        Maximum iteration for MCD stage.
    nsamp_trial:
        Use the best result of `nsamp_trial` trial subsample to start.
    niter_trial:
        Maximum iteration for initialization stage.

    References
    ----------
    Hubert et al. 2017, Minimum covariance determinant and extensions
    """
    n, p = X.shape

    if n <= 2 * p:
        raise ValueError('n_sample must be larger than 2*n_feature!')

    if alpha_mcd is None:
        h = (n + p + 1) // 2
    else:
        h = int(n * alpha_mcd)
    alpha_mcd = h / n

    if weights is not None:
        weights = weights / weights.sum()

    # trials
    det_best = np.inf
    for i in range(nsamp_trial):
        index = np.random.choice(n, p + 1, replace=False, p=weights)
        loc, cov, det, dist, index = _c_step(X, weights, index, h, nstep=niter_trial)
        if det < det_best:
            index_best = index
    loc, cov, det, dist, index = _c_step(X, weights, index_best, h, nstep=niter)

    # the actual fraction within MCD
    if weights is None:
        alpha_tmp = alpha_mcd
    else:
        alpha_tmp = weights[index].sum()

    # consistency factor
    if exact_cut:
        # factor = quantile(dist, weights=weights, q=alpha_tmp) / chi2(p).ppf(alpha_tmp)
        factor = dist[index[-1]] / chi2(p).ppf(alpha_tmp)
    else:
        factor = alpha_tmp / chi2(p + 2).cdf(chi2(p).ppf(alpha_tmp))
    dist /= factor
    cov *= factor

    # weight to enhance the asymptotic efficiency
    if alpha_wgt is not None:
        index = dist < chi2(p).ppf(alpha_wgt)
        loc, cov, dist = _fit_cov(X, weights=weights, index=index)

        # consistency factor
        if exact_cut:
            factor = quantile(dist, weights=weights, q=alpha_wgt) / chi2(p).ppf(alpha_wgt)
        else:
            factor = alpha_wgt / chi2(p + 2).cdf(chi2(p).ppf(alpha_wgt))
        dist /= factor
        cov *= factor

    return loc, cov, dist
