"""Optimization of cuts using the binomial significance estimate ZBi"""

from dataclasses import dataclass, field
import functools

import numpy as np
from scipy.special import betainc, erfinv
import hist
import uproot

from .exclusion.models import ctau, simp
from .exclusion import production
from .exclusion.fit import weightedmean
from . import signal_yield
from .plot import plt, plot2d, annotate, show


def get_edges_with_flow_pad(ax):
    return np.concatenate([
        [ax.edges[0]-ax.widths[0]],
        ax.edges,
        [ax.edges[-1]+ax.widths[-1]]
    ])


def bins_from_centers(c):
    return np.concatenate([
        [3*c[0]/2 - c[1]/2],
        (c[1:]+c[:-1])/2,
        [3*c[-1]/2 - c[-2]/2]
    ])


def centers_from_bins(b):
    return (b[1:]+b[:-1])/2


def cumulate(arr, *, axis = 0, up = None):
    if up is None:
        raise ValueError('Direction of cumulation not defined!')
    if up:
        return np.add.accumulate(arr, axis=axis)
    else:
        return np.flip(np.add.accumulate(np.flip(arr, axis=axis), axis=axis), axis=axis)


def pbi(S, B):
    return betainc(S+B, 1+B, 0.5)


def zbi(S, B):
    return np.sqrt(2) * erfinv(1 - 2*pbi(S,B))


@dataclass
class ZBiOptimum:
    variable_name: str
    variable_label: str
    up: bool
    h : dict
    mass: np.array = field(default_factory = lambda : np.arange(20,126,2))
    eps2: np.array = field(default_factory = lambda : np.logspace(-8,-4,100))
    simp_parameters: dict = field(default_factory = dict)
    
    def __post_init__(self):
        self.variable = get_edges_with_flow_pad(
            self.h[f'simp{self.mass[0]}'][self.variable_name].axes[-1]
        )
        n_test_cuts = self.variable.shape[0]-1
        self.Z = np.full(
            self.mass.shape+self.eps2.shape+(n_test_cuts,), np.nan)
        self.S = np.full(self.Z.shape, 0.0)
        self.B = np.full(self.mass.shape+(n_test_cuts,), 0.0)

        model = simp.Parameters(**self.simp_parameters)
        total_prompt_signal_yield_per_eps2 = production.from_calculators(
            production.radiative_fraction.alic_2016_simps,
            production.TridentDifferentialProduction.from_hist(self.h['data']['cr']),
            production.radiative_acceptance.alic_2016_simps
        )
        _cumulate = functools.partial(cumulate, up = self.up)
    
        for i_m, m in enumerate(self.mass):
            counts_h = self.h[f'simp{m}'][self.variable_name]
            energy_h = self.h[f'simp{m}']['true_vd_energy']
            mean, _stdd, _merr = weightedmean(energy_h.axes[0].centers, energy_h.values())
            signal_diff_yield = signal_yield(
                mass = m,
                eps2 = self.eps2,
                z = counts_h.axes[0],
                prompt_signal_yield_per_eps2 = (
                    total_prompt_signal_yield_per_eps2(m*model.mass_ratio_Ap_to_Vd)
                ),
                final_selection_counts_h = counts_h,
                mean_energy_GeV = mean,
                model = model
            )
            sy = np.sum(signal_diff_yield, axis=1)
            self.S[i_m,...] = _cumulate(sy, axis=1)
            self.B[i_m,...] = _cumulate(
                self.h['data'][m][self.variable_name][sum,:].values(flow=True),
                axis=0
            )
            self.Z[i_m,...] = zbi(self.S[i_m,...], self.B[i_m,...])

    
    def view_mass(self, m, prefix = None):
        i_m = np.digitize(m, bins=self.mass)-1
        eps2_bins = bins_from_centers(self.eps2)
        note = functools.partial(annotate, r'$m_{V_D} = $'+f'{self.mass[i_m]:.0f}MeV')
        
        plot2d(
            self.S[i_m,...].T,
            xbins=self.variable,
            ybins=eps2_bins,
            cbarlabel = 'Signal Yield',
            norm='log'
        )
        plt.yscale('log')
        plt.ylabel(r'$\epsilon^2$')
        plt.xlabel(f'{self.variable_label} Cut')
        note()
        show(
            filename = f'{prefix}signal-yield.pdf' if prefix is not None else None
        )

        plt.plot(
            self.variable[slice(1,None,None) if self.up else slice(None,-1,None)],
            self.B[i_m,:]
        )
        plt.ylabel('Bkgd Yield')
        plt.xlabel(f'{self.variable_label} Cut')
        note()
        show(
            filename = f'{prefix}bkgd-yield.pdf' if prefix is not None else None
        )

        plot2d(
            self.Z[i_m,...].T,
            xbins = self.variable,
            ybins = eps2_bins,
            cbarlabel = r'$Z_\mathrm{Bi}$'
        )
        plt.yscale('log')
        plt.ylabel(r'$\epsilon^2$')
        plt.xlabel(f'{self.variable_label} Cut')
        note()
        show(
            filename = f'{prefix}zbi.pdf' if prefix is not None else None
        )

    
    def view_slice(self, m = None, e = None, contour = None, z = None, **kwargs):
        if m is None and e is None:
            raise ValueError('A mass or eps2 slice must be chosen.')
        elif m is not None and e is not None:
            raise ValueError('Only one of mass or eps2 choice must be made.')
    
    
        if m is not None:
            i_m = np.digitize(m, bins=self.mass)-1
            z_sl = (i_m,slice(None),slice(None))
            y = (
                bins_from_centers(self.eps2),
                r'$\epsilon^2$',
                'log'
            )
            note = r'$m_{V_D} = $'+f'{self.mass[i_m]:.0f}MeV'
        elif e is not None:
            i_e = np.digitize(e, bins=self.eps2)-1
            z_sl = (slice(None), i_e, slice(None))
            y = (
                bins_from_centers(self.mass),
                'Mass / MeV',
                'linear'
            )
            note = r'$\epsilon^2 = $'+f'{self.eps2[i_e]:.1e}'
    
        to_plot, xbins, cbarlabel = self.Z[z_sl], self.variable, r'$Z_\mathrm{Bi}$'
        if z is not None:
            if z == 'diff':
                to_plot = (
                    to_plot[...,1:]-to_plot[...,:-1]
                    if self.up else
                    to_plot[...,:-1]-to_plot[...,1:]
                )
                xbins = xbins[slice(1,None,None) if self.up else slice(None,-1,None)]
                cbarlabel = r'$\Delta Z_\mathrm{Bi}$'
            else:
                to_plot = z
                if 'cbarlabel' in kwargs:
                    cbarlabel = kwargs['cbarlabel']
                    del kwargs['cbarlabel']
        
        fig, ax = plt.subplots()
        ybins, ylabel, yscale = y
        plot2d(
            H = to_plot.T,
            xbins = xbins,
            ybins = ybins,
            cbarlabel = cbarlabel,
            ax=ax
        )
        c_pts = None
        if contour is not None:
            c_art = ax.contour(
                centers_from_bins(xbins),
                centers_from_bins(ybins),
                to_plot,
                levels = [contour],
                colors='tab:red'
            )
            # get points before interupting line with label
            c_pts = c_art.allsegs[0][0]
            ax.clabel(c_art, c_art.levels, inline=True, fmt=lambda l: f'{l:.1e}', fontsize=10)
        ax.set_xlabel(f'{self.variable_label} Cut')
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        annotate(note, color='white')
        show(ax=ax, **kwargs)
        return c_pts
