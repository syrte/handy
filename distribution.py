from __future__ import division
from scipy.stats import rv_continuous
from numpy import log, exp, nan
from numba import vectorize
import numpy as np
from matplotlib import pyplot as plt

__all__ = ['powdist', 'expdist']


class rv_custom(rv_continuous):
    def ppf(self, *args, **kwargs):
        return self._ppf(*args, **kwargs)

    def isf(self, *args, **kwargs):
        return self._isf(*args, **kwargs)

    def fit(self, *args, **kwargs):
        kwargs.update(floc=0, fscale=1)
        return super(rv_custom, self).fit(*args, **kwargs)[:-2]

    def plot_pdf(self, n, a, b, *args, **kwargs):
        x = np.linspace(a, b, kwargs.pop('num', 50))
        y = self.pdf(x, n, a, b)
        return plt.plot(x, y, *args, **kwargs)

    def plot_cdf(self, n, a, b, *args, **kwargs):
        x = np.linspace(a, b, kwargs.pop('num', 50))
        y = self.cdf(x, n, a, b)
        return plt.plot(x, y, *args, **kwargs)


class powlaw_gen(rv_custom):
    """
    A power-function continuous random variable.
    The probability density function is
        powdist.pdf(x, n, a, b) = A * x**n
    for `0 <= a <= x <= b, n > -1`
    or  `0 < a <= x <= b, n <= -1`,
    where A is normalization constant.

    Examples
    --------
    n, a, b = 2, 0, 1
    p = powdist(n, a, b)
    x = np.linspace(a, b, 121)
    plt.hist(p.rvs(100000), 30, normed=True)
    plt.plot(x, p.pdf(x))
    """
    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _pdf(x, n, a, b):
        if x < a or x > b:
            return 0.
        elif n != -1:
            return (n + 1) / (b**(n + 1) - a**(n + 1)) * x**n
        else:
            return 1 / (log(b) - log(a)) / x

    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _cdf(x, n, a, b):
        if x <= a:
            return 0.
        elif x >= b:
            return 1.
        elif n != -1:
            x, a, b = x**(n + 1), a**(n + 1), b**(n + 1)
            return (x - a) / (b - a)
        else:
            x, a, b = log(x), log(a), log(b)
            return (x - a) / (b - a)

    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _ppf(q, n, a, b):
        if q < 0 or q > 1:
            return nan
        if n != -1:
            return ((1 - q) * a**(n + 1) + q * b**(n + 1)) ** (1 / (n + 1))
        else:
            return exp((1 - q) * log(a) + q * log(b))

    @staticmethod
    @vectorize("b1(f8, f8, f8)")
    def _argcheck(n, a, b):
        if n > -1:
            return (0 <= a < b)
        else:
            return (0 < a < b)


class expon_gen(rv_custom):
    """
    A Exponential continuous random variable.
    The probability density function is
        powlaw.pdf(x, n, a, b) = A * exp(n*x)
    for ``0 <= a <= x <= b``, where A is normalization constant.

    Examples
    --------
    n, a, b = 2, 0, 1
    p = expdist(n, a, b)
    x = np.linspace(a, b, 121)
    plt.hist(p.rvs(100000), 30, normed=True)
    plt.plot(x, p.pdf(x))
    """
    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _pdf(x, n, a, b):
        if x < a or x > b:
            return 0.
        elif n == 0:
            return 1 / (b - a)
        else:
            return n * exp(n * x) / (exp(n * b) - exp(n * a))

    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _cdf(x, n, a, b):
        if x <= a:
            return 0.
        elif x >= b:
            return 1.
        elif n == 0:
            return (x - a) / (b - a)
        else:
            a, b, x = exp(n * a), exp(n * b), exp(n * x)
            return (x - a) / (b - a)

    @staticmethod
    @vectorize("f8(f8, f8, f8, f8)")
    def _ppf(q, n, a, b):
        if n == 0:
            return (1 - q) * a + q * b
        else:
            return log((1 - q) * exp(n * a) + q * exp(n * b)) / n

    @staticmethod
    @vectorize("b1(f8, f8, f8)")
    def _argcheck(n, a, b):
        return (0 <= a < b)


powdist = powlaw_gen(name="powerlaw", shapes="n, a, b")
expdist = expon_gen(name="exponential", shapes="n, a, b")
