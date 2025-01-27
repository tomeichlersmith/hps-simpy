"""Helpers for interacting with limit lines in data/"""

from pathlib import Path

import numpy as np
from scipy import interpolate

_line_data_dir = Path(__file__).parent / 'data'


def log_interpolate_griddata(
    ref_xy, ref_z,
    eval_x, eval_y
):
    log_ref_z = np.full_like(ref_z, -200)
    log_ref_z[ref_z > 0] = np.log(ref_z[ref_z > 0])
    loggrid = interpolate.griddata(
        np.log(ref_xy), log_ref_z,
        (np.log(eval_x), np.log(eval_y)),
        method = 'linear'
    )
    return np.exp(loggrid)


def abundances(mpioverfpi = '4pi'):
    if mpioverfpi == '4pi':
        abund_1e8 = np.loadtxt(_line_data_dir / 'der_line_rV_1p8_aD_1m2_fine.dat')
        abund_1e6 = np.loadtxt(_line_data_dir / 'der_line_rV_1p6_aD_1m2_fine.dat')
    else:
        abund_1e8 = np.loadtxt(_line_data_dir / 'abun_cont_mPi_vs_eps_rV_1p8_aD_1m2_neutral_V_only_combined.dat')
        abund_1e6 = np.loadtxt(_line_data_dir / 'abun_cont_mPi_vs_eps_rV_1p6_aD_1m2_neutral_V_only.dat')

    # the abundances are given in mPi vs eps so we scale the x by 3 to get it up to A'
    abund_1e8[:,0] = 3*abund_1e8[:,0]
    abund_1e6[:,0] = 3*abund_1e6[:,0]

    return abund_1e8, abund_1e6


def cmb_decay_bound(mpioverfpi = '4pi'):
    return np.loadtxt(_line_data_dir / f'cmb_decay_bound_Delta_0p05_mPiOverFpi_{mpioverfpi}_aD_1em2.dat')


def babar(mpioverfpi = '4pi'):
    return np.loadtxt(_line_data_dir / f'babar_lim_aD_1em2_mPiOverFpi_{mpioverfpi}_two_body.dat')


class ScanFile:
    """The 'scan file' scans the mA/eps plane and records information at
    various points.

    """

    def __init__(self, mpioverfpi = '4pi'):
        self._table = np.loadtxt(
            _line_data_dir / 
            f'ft_reach_mpi_mrho_ma_1_1.8_3_ad_0.01_mpioverfpi_{mpioverfpi}_ldmx_theory_paper.dat'
        )

    @property
    def mA(self):
        return self._table[:,0]

    @property
    def eps(self):
        return self._table[:,1]

    @property
    def self_interaction(self):
        return self._table[:,10]

    @property
    def e137(self):
        return self._table[:,7]

    @property
    def orsay(self):
        return self._table[:,8]

    def draw_exclusion(
        self,
        experiment,
        level,
        ax = None,
        color='lightgray'
    ):
        mAp, eps = np.meshgrid(
            np.logspace(-2, 1, 100),
            np.logspace(-6,-2,100)
        )
        interp = log_interpolate_griddata(
            self._table[:,(0,1)],
            getattr(self, experiment),
            mAp,
            eps
        )
        c_art = ax.contour(
            mAp, eps, interp, levels = [level],
            linewidths=[0] # just using contour to get the bounding line
        )
        c_pts = c_art.allsegs[0][0][:,[0,1]]
        fill_art = ax.fill_between(c_pts[:,0], c_pts[:,1], 1, color=color)
        return c_pts, c_art, fill_art


def draw_non_hps(mpioverfpi):
    from .plot import plt
    a1, a2 = abundances(mpioverfpi = mpioverfpi)
    plt.plot(a1[:,0], a1[:,1], color='black')
    plt.plot(a2[:,0], a2[:,1], color='black', linestyle='--')
    cmb = cmb_decay_bound(mpioverfpi = mpioverfpi)
    plt.plot(cmb[:,0], cmb[:,1], color='gray', linestyle=':')
    
    babar_line = babar(mpioverfpi = mpioverfpi)
    plt.fill_between(
        babar_line[:,0],
        babar_line[:,1],
        1e-1,
        color='lightgray'
    )
    
    sf = ScanFile(mpioverfpi = mpioverfpi)
    sf.draw_exclusion('e137', 10, ax = plt.gca())
    sf.draw_exclusion('orsay', 3, ax = plt.gca())
    # plt.contour(sf.mA, sf.eps, sf.e137, levels = [10], colors='gray')

    if mpioverfpi == '4pi':
        plt.text(0.13, 2e-5, 'E137', color='gray')
        plt.text(5e-2, 1e-4, 'Orsay', color='gray')
        plt.text(1, 2.5e-3, 'BaBar', color='gray')
        # plt.text(0.56, 6e-7, r'$1\,\mathrm{cm}^2/\mathrm{g}$', color='gray', rotation=90)
        # plt.text(0.32, 6e-7, r'$5\,\mathrm{cm}^2/\mathrm{g}$', color='gray', rotation=90)
        plt.text(5e-2, 4e-6, r'$m_V/m_\pi = 1.8$', color='black', rotation=18, va='bottom')
        plt.text(5e-2, 7e-7, r'$m_V/m_\pi = 1.6$', color='black', rotation=18, va='bottom')
        plt.text(1, 2.5e-6, 'CMB', color='gray')
        plt.title(r'$m_\pi/f_\pi = 4\pi$')
    elif mpioverfpi == '3':
        plt.text(0.18, 8e-6, 'E137', color='gray')
        plt.text(5e-2, 1e-4, 'Orsay', color='gray')
        plt.text(1, 2e-3, 'BaBar', color='gray', va='top')
        # plt.text(0.56, 6e-7, r'$1\,\mathrm{cm}^2/\mathrm{g}$', color='gray', rotation=90)
        # plt.text(0.32, 6e-7, r'$5\,\mathrm{cm}^2/\mathrm{g}$', color='gray', rotation=90)
        plt.text(5e-2, 3.5e-6, r'$m_V/m_\pi = 1.8$', color='black', rotation=18, va='bottom')
        plt.text(5e-2, 6e-7, r'$m_V/m_\pi = 1.6$', color='black', rotation=18, va='bottom')
        plt.text(1, 2.5e-6, 'CMB', color='gray')
        plt.title(r'$m_\pi/f_\pi = 3$')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(3e-2,2)
    plt.ylim(1e-7,1e-2)
    plt.ylabel(r'$\epsilon$')
    plt.xlabel(r"$m_{A'}$ / GeV")
        