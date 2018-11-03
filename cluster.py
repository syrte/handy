from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
# from collections import namedtuple

__all__ = ['DensPeak']


class DensPeak:
    """
    Fast clustering with k-d tree.
    Clustering By Fast Search And Find Of Density Peaks. Alex Rodriguez, Alessandro Laio. Science, 2014

    Examples
    --------
    import numpy as np
    from matplotlib import pyplot as plt

    pts = np.random.randn(1000, 2)
    pts[:300] += 3
    dpeak = DensPeak(pts, k=10)
    peak, ix_peak, group = dpeak.peak_plot()
    """

    def __init__(self, pts, k=None, r=None, kmax=None, rmax=None):
        """
        Parameters
        ----------
        pts : array, shape(n, d)
            Data points. Should be already normalized if necessary.
        k : int
            Neighbors used to estimate the local density rho.
        kmax : int
            If given, only search the nearest kmax neighbors to calculate delta.
            kmax is equivalent to search a sphere of size about kmax**(1/d) times
            the local average separation between points.
            Default is to search all points.
        rmax : float
            If given, only search the neighbors within rmax to calculate delta.
            Default is to search all points.

        Todos
        -----
        Check references of optimal choice of k
        """
        if (k is not None) and (r is not None):
            raise ValueError("Only one of 'k' or 'r' can be specified!")
        if (kmax is not None) and (rmax is not None):
            raise ValueError("Only one of 'kmax' or 'rmax' can be specified!")

        pts = np.asfarray(pts)
        npts, ndim = pts.shape
        Rmax = np.linalg.norm(pts.max(0) - pts.min(0))
        tree = KDTree(pts)

        # density
        if r is not None:
            k = tree.query_radius(pts, count_only=True)
        elif k is not None:
            r = tree.query(pts, k)[0][:, -1]

        sphere_coeff = np.pi**(0.5 * ndim) / gamma_func(0.5 * ndim + 1)
        rho = k / (sphere_coeff * r**ndim)

        # delta
        delta = np.full(npts, Rmax, dtype='float')
        chief = np.full(npts, -1, dtype='int')  # superior neighbor
        if kmax is not None:
            dists, index = tree.query(pts, kmax, return_distance=True)
            for i in range(npts):
                for j, dist in zip(index[i], dists[i]):
                    if (rho[j] > rho[i]):
                        chief[i] = j
                        delta[i] = dist
                        break
        elif rmax is not None:
            index, dists = tree.query_radius(pts, return_distance=True)
            for i in range(npts):
                for j, dist in zip(index[i], dists[i]):
                    if (rho[j] > rho[i]) and (dist < delta[i]):
                        chief[i] = j
                        delta[i] = dist
        else:
            dists = squareform(pdist(pts))
            for i in range(npts):
                for j, dist in enumerate(dists[i]):
                    if (rho[j] > rho[i]) and (dist < delta[i]):
                        chief[i] = j
                        delta[i] = dist

        # gamma
        gamma = rho * delta**ndim
        lg_gamma = np.log10(gamma)
        # both rho and delta will be always positive
        # lg_gamma[gamma <= 0] = lg_gamma[gamma > 0].min()

        # properties
        self.npts = npts
        self.pts = pts
        self.rho = rho
        self.delta = delta
        self.gamma = gamma
        self.lg_gamma = lg_gamma
        self.chief = chief

    def peak(self, gamma_th=None, cluster=False):
        """
        Parameters
        ----------
        gamma_th : float
            Threshold for peak identification.
        cluster : bool
            If true, the groupid will also be returned.

        Returns
        -------
        peak : array
            Position of peak points
        ix_peak : array
            Index of peak points in the original array.
        group : array
            Group id of each point, start from 0.
        """
        if gamma_th is None:
            gamma_th = 10**(self.lg_gamma.mean() + 4.5 * self.lg_gamma.std())

        ix_peak = np.where(self.gamma > gamma_th)[0]
        peak = self.pts[ix_peak]
        npeak = len(ix_peak)

        if cluster:
            chief = self.chief
            group = np.full(self.npts, -1, dtype='int')
            group[ix_peak] = np.arange(npeak)

            ix_sort_rho = np.argsort(self.rho)[::-1]
            for i in ix_sort_rho:
                j = chief[i]
                if j != -1 and group[i] == -1:
                    group[i] = group[j]
            return peak, ix_peak, group
        else:
            return peak, ix_peak

    def peak_plot(self, gamma_th=None):
        dpeak = self
        peak, ix_peak, group = dpeak.peak(gamma_th, cluster=True)

        xlims, ylims = np.quantile(dpeak.pts, q=[0.05, 0.95], axis=0).T[:2]

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.scatter(*dpeak.pts.T[:2], c=group, s=5)
        plt.colorbar()
        plt.scatter(*peak.T[:2], s=200, c='k', marker='x')
        plt.xlim(xlims)
        plt.ylim(ylims)

        plt.subplot(223)
        plt.scatter(*dpeak.pts.T[:2], c=dpeak.rho, s=5, vmax=np.quantile(dpeak.rho, q=0.95))
        plt.colorbar()
        plt.scatter(*peak.T[:2], s=200, c='k', marker='x')
        plt.xlim(xlims)
        plt.ylim(ylims)

        plt.subplot(222)
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(dpeak.rho, dpeak.delta, s=5)
        plt.scatter(dpeak.rho[ix_peak], dpeak.delta[ix_peak], s=200, c='k', marker='x')

        plt.subplot(224)
        plt.hist(dpeak.lg_gamma, 51)
        plt.yscale('log')

        return peak, ix_peak, group
