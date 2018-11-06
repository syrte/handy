from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
# from collections import namedtuple

__all__ = ['DensPeakFinder']


class DensPeakFinder:
    """
    Fast clustering with k-d tree.
    Clustering By Fast Search And Find Of Density Peaks. Alex Rodriguez, Alessandro Laio. Science, 2014

    Examples
    --------
    import numpy as np
    from matplotlib import pyplot as plt

    n, d = 1000, 3
    pts = np.random.randn(n, d)
    pts[:int(n*0.3), :2] *= 0.5
    pts[:int(n*0.3), :2] += [3, 1]
    pts[int(n*0.3):int(n*0.5), :2] += [1, 3]

    dpeak = DensPeakFinder(pts, k=10)
    peak, ix_peak, group = dpeak.plot_peak()
    # if you don't want the plot
    peak, ix_peak, group = dpeak.find_peak(400, cluster=True)
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
        Optimal choice of k and gamma
        Performance optimization with Cython or Numba
        Substructure within density saddle point
        Labeling the noise
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
            k = tree.query_radius(pts, r, count_only=True)
        elif k is not None:
            r = tree.query(pts, k)[0][:, -1]

        sphere_coeff = np.pi**(0.5 * ndim) / gamma_func(0.5 * ndim + 1)
        rho = k / (sphere_coeff * r**ndim)
        rho[rho == 0] = rho[rho > 0].min() / 2  # reduce by an arbitrary factor

        # delta
        delta = np.full(npts, Rmax, dtype='float')
        chief = np.full(npts, -1, dtype='int')  # superior neighbor
        if kmax is not None or rmax is not None:
            if kmax is not None:
                dists, index = tree.query(
                    pts, kmax, return_distance=True, sort_results=True)
            else:
                index, dists = tree.query_radius(
                    pts, rmax, return_distance=True, sort_results=True)
            for i in range(npts):
                rho_i = rho[i]
                for j, dist in zip(index[i], dists[i]):
                    if (rho[j] > rho_i):
                        chief_i, delta_i = j, dist
                        break
                chief[i], delta[i] = chief_i, delta_i
        else:
            dists = squareform(pdist(pts))
            for i in range(npts):
                rho_i, delta_i = rho[i], delta[i]
                for j, dist in enumerate(dists[i]):
                    if (rho[j] > rho_i) and (dist < delta_i):
                        chief_i, delta_i = j, dist
                chief[i], delta[i] = chief_i, delta_i

        # gamma
        gamma = sphere_coeff * rho * delta**ndim  # need sphere_coeff?
        sorted_index = np.argsort(gamma)
        sorted_gamma = gamma[sorted_index]

        # properties
        self.npts = npts
        self.ndim = ndim
        self.pts = pts
        self.rho = rho
        self.delta = delta
        self.gamma = gamma
        self.chief = chief
        self.sorted_index = sorted_index
        self.sorted_gamma = sorted_gamma

    def get_gamma_threshold(self, gamma_th=None):
        # XXX
        if gamma_th is None:
            gamma = self.gamma
            lg_gamma = np.log10(gamma)
            gamma_threshold = 10**(np.nanmean(lg_gamma) + 4.5 * np.nanstd(lg_gamma))
            return gamma_threshold
        else:
            return gamma_th

    def find_peak(self, gamma_th=None, npeak=None, rho_th=None, cluster=False):
        """
        Parameters
        ----------
        gamma_th : float
            Threshold for peak identification.
        rho_th : float
            Threshold for noisy points.
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
        if rho_th is None:
            sorted_index = self.sorted_index
            sorted_gamma = self.sorted_gamma
        else:
            ix_rho_th = np.where(self.rho[self.sorted_index] >= rho_th)[0]
            sorted_index = self.sorted_index[ix_rho_th]
            sorted_gamma = self.sorted_gamma[ix_rho_th]

        if npeak is not None:
            ix_peak = sorted_index[-npeak:][::-1]
        else:
            if gamma_th is None:
                gamma_th = self.get_gamma_threshold()
            ix_th = np.searchsorted(sorted_gamma, gamma_th, side='right')
            ix_peak = sorted_index[ix_th:][::-1]
        peak = self.pts[ix_peak]
        npeak = len(ix_peak)

        if cluster:
            if rho_th is None:
                chief = self.chief
            else:
                # don't assign group for low density points
                chief = self.chief.copy()
                chief[self.rho < rho_th] = -1
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

    def plot_peak(self, gamma_th=None, npeak=None, rho_th=None, axes=[0, 1]):
        """
        Show the decision graph and return peaks.

        Parameters
        ----------
        gamma_th : float
            Threshold for peak identification.
        rho_th : float
            Threshold for noisy points.
        axes : list of length 2
            Specify the axes of n-d data points to show.
        """
        if npeak is None and gamma_th is None:
            gamma_th = self.get_gamma_threshold()
        if len(axes) != 2:
            raise ValueError("Argument 'axes' should be shape (2,)")

        dpeak = self
        peak, ix_peak, group = dpeak.find_peak(
            gamma_th, npeak=npeak, rho_th=rho_th, cluster=True)
        npeak = len(peak)

        xlims, ylims = np.percentile(dpeak.pts, q=[5, 95], axis=0).T[axes]

        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.subplot(222)
        plt.scatter(*dpeak.pts.T[axes], c=group, s=5,
                    cmap=plt.get_cmap(lut=npeak + 1), vmin=-1.5, vmax=npeak - 0.5)
        plt.colorbar(ticks=np.arange(-1, npeak), label='group')
        plt.scatter(*peak.T[axes], s=150, lw=3, c='k', marker='x')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel(r'$X%d$' % axes[0])
        plt.ylabel(r'$X%d$' % axes[1])

        plt.subplot(224)
        plt.scatter(*dpeak.pts.T[axes], c=dpeak.rho, s=5, vmax=np.percentile(dpeak.rho, q=90))
        plt.colorbar(label=r'$\rho$')
        plt.scatter(*peak.T[axes], s=150, lw=3, c='k', marker='x')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel(r'$X%d$' % axes[0])
        plt.ylabel(r'$X%d$' % axes[1])

        plt.subplot(221)
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(dpeak.rho, dpeak.delta, s=5)
        plt.scatter(dpeak.rho[ix_peak], dpeak.delta[ix_peak], s=150, lw=3, c='k', marker='x')
        if gamma_th is not None:
            xlims = np.array(plt.gca().get_xlim())
            plt.plot(xlims, (gamma_th / xlims)**(1 / dpeak.ndim), ls='--', color='gray')
        if rho_th is not None:
            plt.axvline(rho_th, ls='--', color='gray')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\delta$')

        plt.subplot(223)
        gamma_sorted = dpeak.sorted_gamma
        gamma_sorted_mid = np.sqrt(gamma_sorted[1:] * gamma_sorted[:-1])
        n_cum = np.arange(dpeak.npts, 0, -1)
        dlngamma_dlnN = -np.diff(np.log(n_cum)) / np.diff(np.log(gamma_sorted))

        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(gamma_sorted, n_cum, s=5)
        if gamma_th is not None:
            plt.axvline(gamma_th, ls='--', color='gray')
        plt.xlabel(r'$\gamma=\delta^d\rho$')
        plt.ylabel(r'$N(>\gamma)$')

        plt.twinx()
        plt.plot(gamma_sorted_mid[-10:], dlngamma_dlnN[-10:], ls='--', lw=0.75, color='gray')
        plt.ylim(0, 2)
        return peak, ix_peak, group
