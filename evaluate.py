from pathlib import Path
from typing import List
import os
from tqdm import tqdm

import uproot
import awkward as ak
import numpy as np
import hist
import hist.dask
import pickle

from . import Analyzer
from .exclusion.fit import weightedmean
from . import exclusion_estimate, search
from .plot import plt, mpl, mplhep, show, from_dir, plt_mass_by_eps2
from . import lumi
from . import search
import functools
show = functools.partial(show, display = False)

def annotate(*args, i_axis=0, **kwargs):
    def _annotation_impl(fig, axes):
        axes[i_axis].annotate(*args, **kwargs)
    return _annotation_impl


def plot(
    evaluation: str|Path,
    out_dir: str|Path = Path.cwd(),
    label: List[str] = [],
    vmax_expected = None, # 0.5 for 1.6%
    vmax_allowed  = None, # 10. for 1.6%
    vmax_ratio    = None, # 0.1 for 1.6%
    excl_level    = None,
    data_frac = 0.016
):
    """plot an evaluation

    Parameters
    ----------
    evaluation: str|Path
        path to pickle file holding the evaluation results (written by run usually)
    out_dir: str|Path, optional, default is CWD
        directory in which to put images
    label: List[str], optional, default []
        extra labels to include in plots
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(evaluation, 'rb') as f:
        r = pickle.load(f)

    if 'data_frac' in r:
        data_frac = r['data_frac']

    if excl_level is None:
        excl_level = data_frac
    
    if 'search' in r:
        search.show(
            r['search'],
            extras = annotate(
                '\n'.join([
                    f'${r["selections"].mass_window}\sigma$ InvM Window',
                    f'${r["selections"].mass_sideband}\sigma$ InvM Sideband'
                ]+label),
                xy=(0.5,0.95),
                xycoords='axes fraction',
                ha='center', va='top',
                size='x-small'
            ),
            legend_kw = dict(title='SR L1L2'),
            lumi = data_frac*lumi.data.total,
            display = False,
            filename = out_dir / 'search.pdf'
        )

    
    _show = functools.partial(show, lumi = data_frac*lumi.data.total)
    
    ee = r['excl_estimate']
    plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.max_allowed, 'Max Signal Allowed Estimate',
        vmax = vmax_allowed
    )
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top', color='white'
    )
    _show(filename=out_dir / 'max-signal-allowed.pdf')
    
    plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.expected, 'Expected Signal',
        vmax = vmax_expected
    )
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top', color='white'
    )
    _show(filename=out_dir / 'expected-signal.pdf')

    plt.figure(figsize=(12,10))
    ax = plt.gca()
    im = ax.pcolormesh(ee.mass, ee.z, ee.signal_efficiency.T)
    ax.set_ylabel(r'True Vertex $z$ / mm')
    ax.set_xlabel('Invariant Mass / MeV')
    cbar = plt.colorbar(im, label='Signal Efficiency')
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top', color='white'
    )
    show(filename=out_dir / 'signal-efficiency.pdf', lumi = 'SIMP Sim')
    
    im, cbar = plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.expected/ee.max_allowed, 'Expected / Max Allowed',
        # norm=mpl.colors.TwoSlopeNorm(vcenter=1.),
        cmap='Blues',
        vmax = vmax_ratio
    )
    cbar.ax.axhline(excl_level, color='tab:red')
    plt.contour(
        ee.mass, ee.eps2, (ee.expected/ee.max_allowed).T,
        [excl_level],
        colors='tab:red'
    )
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top'
    )
    _show(filename=out_dir / 'exclusion-estimate.pdf')


class TightEvaluation(Analyzer):
    def process_simp(self, events, mass):
        """From signal, we need three different distributions.
    
        The true z distribution both at pre-selection and after all tight cuts.
        The pre-selection distribution is used to estimate the scaling factor
        that undoes double-counting of the prompt acceptance (beta).
        The tight cuts distribution becomes F(z) after division by the simulated
        z distribution.
    
        The true energy distribution is used to calculate the mean energy
        for the decay reweighting factors. It also can be used to assure ourselves
        that the distribution is tight enough that doing something more precise
        is not worth it.
        """
        sl = self.selections(events)
    
        invm = events['vertex.invM_']*1000
        sigma_m = self.selections.mass_resolution(mass)
        invm_pull = abs(invm - mass)/sigma_m
    
        cats = {
            'pre': sl.preselection,
            'one-left': sl.search&(invm_pull < self.selections.excl_mass_window),
            'final': sl.exclusion&(invm_pull < self.selections.excl_mass_window)
        }
    
        o = {}
        o['z'] = (
            hist.dask.Hist.new
            .StrCategory(list(cats.keys()))
            .Reg(250,-4.3,250-4.3,label='Truth $z$ / mm')
            .Double()
        )
        o['energy'] = (
            hist.dask.Hist.new
            .StrCategory(list(cats.keys()))
            .Reg(230,0,2.3,label=r'True $V_D$ Energy / GeV')
            .Double()
        )
        for name, cat in cats.items():
            o['z'].fill(name, events[cat]['true_vd.vtx_z_'])
            o['energy'].fill(name, events[cat]['true_vd.energy_'])
    
        return o


    def process_data(self, events):
        """The data requires a few more distributions.
    
        From the control region (CR), we need the invariant mass distribution
        after the pre-selection cuts in order to estimate the trident differential
        production rate.
    
        From the signal region (SR) we do a scan over all the mass points including
        all the tight selection cuts and applying a mass window so that we select
        the final data events that are candidates within each mass search. These
        events can then be used with the differential signal yield and OIM to estimate
        and exclusion limit.
        """
        sl = self.selections(events)
        invm = events['vertex.invM_']*1000
        ele_y0 = events['ele.track_.z0_']
        pos_y0 = events['pos.track_.z0_']
        min_y0 = np.minimum(abs(ele_y0), abs(pos_y0))
        vtx_z = events['vertex.pos_'].fZ
    
        h = {}
        h['cr'] = hist.dask.Hist.new.Reg(220,0,220,label=r'$m_\text{reco}$ / MeV').Double()
        h['cr'].fill(invm[sl.cr])
        h['invm_vs_min_y0'] = (
            hist.dask.Hist.new
            .Reg(250,0,250,label=r'$m_\text{reco}$ / MeV')
            .Reg(600,0,6,label=r'$y_{0,\min}$ / mm')
            .Double()
        ).fill(invm[sl.search], min_y0[sl.search])
    
        h['invm_vs_z'] = (
            hist.dask.Hist.new
            .StrCat(['search','excl'])
            .Reg(250,0,250,label=r'$m_\text{reco}$ / MeV')
            .Reg(250,-4.3,250-4.3,label=r'Vertex $z$ / mm')
            .Double()
        ).fill(
            'search', invm[sl.search], vtx_z[sl.search]
        ).fill(
            'excl', invm[sl.exclusion], vtx_z[sl.exclusion]
        )
    
    
        for mass in range(20,126,2):
            h[mass] = {}
    
            sigma_m = self.selections.mass_resolution(mass)
            invm_pull = abs(invm - mass)/sigma_m
            
            excl_sl = sl.exclusion&(invm_pull < self.selections.excl_mass_window)
            h[mass]['z'] = events['vertex.pos_'].fZ[excl_sl]
    
        return h

    def plot(self, o):
        print('Search')
        # deduce mass range from histogram availability
        # go up until mass+mass_sideband*resolution is above axis maximum
        sampled_mass = o['data']['invm_vs_min_y0'].axes[0].edges
        search_max = sampled_mass[(
            np.argmax(
                sampled_mass+self.selections.mass_sideband*self.selections.mass_resolution(sampled_mass) > sampled_mass[-1])
        )]
    
        r = {}
        r['data_frac'] = self.data_frac
        r['selections'] = self.selections
        r['search'] = search.invm_y0(
            mass = np.arange(20,search_max,1),
            data = o['data']['invm_vs_min_y0'],
            y0_edges = search.deduce_y0_edges_from_y0_cut(
                o['data']['invm_vs_min_y0'],
                self.selections.mass_window,
                self.selections.y0_cut
            ),
            invm_edges = (self.selections.mass_window, self.selections.mass_sideband),
            n_trials = 50000
        )
        
        with open(self.outdir / 'eval.pkl', 'wb') as f:
            pickle.dump(r, f)
    
        print('Exclusion')
        def get_mean_energy(m):
            energy_h = o[f'simp{m}']['energy']
            mean, stdd, merr = weightedmean(
                energy_h.axes[1].centers,
                energy_h['pre',:].values()
            )
            return mean
        
        r['excl_estimate'] = exclusion_estimate(
            mass = np.arange(20,126,2),
            eps2 = np.logspace(-8,-4,50),
            z = o['simp20']['z'].axes[1],
            invm_cr_h = o['data']['cr'],
            final_selection_counts_by_mass_h = {
                m: o[f'simp{m}']['z']['final',:]
                for m in range(20,126,2)
            },
            mean_energy_GeV_by_mass = {
                m:  get_mean_energy(m)
                for m in range(20,126,2)
            },
            data_z = {
                m: o['data'][m]['z']
                for m in range(20,126,2)
            }
        )
        
        with open(self.outdir / 'eval.pkl', 'wb') as f:
            pickle.dump(r, f)
