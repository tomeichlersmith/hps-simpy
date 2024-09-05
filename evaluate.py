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

import dask
from dask.diagnostics import ProgressBar

from simpy.dask import SelectionSet, sample_list
from simpy.exclusion.fit import weightedmean
from simpy import exclusion_estimate, search
from simpy import get_true_vd_z_file
from simpy.exclusion.production import mass_resolution
from simpy.exclusion.production._polynomial import polynomial
from simpy.plot import plt, mpl, mplhep, show, from_dir, plt_mass_by_eps2
from simpy import lumi
from simpy import search
lumi.data.lumi = 0.016*10.7
import functools
show = functools.partial(show, lumi = lumi.data.lumi, display = False)

from dataclasses import dataclass, asdict
from typing import Callable


def polynomial_with_limit(*coefficients, floor = None, ceiling = None):
    fit = polynomial(*coefficients)
    # np.clip is not defined for dask_awkward, so we need to do max/min manually
    if floor is None and ceiling is None:
        return fit
    elif floor is not None and ceiling is not None:
        return lambda x: np.minimum(np.maximum(fit(x), floor), ceiling)
    elif floor is not None:
        return lambda x: np.maximum(fit(x), floor)
    else:
        return lambda x: np.minimum(fit(x), ceiling)    


@dataclass
class CutByMass:
    coefficients: list[float]
    floor : float = None
    ceiling : float = None

    def __post_init__(self):
        """define the cut function callable now that we have been configured

        np.clip is unfortunately not defined for dask_awkward, so
        we have to manually do it ourselves using np.maximum and np.minimum
        """
        self._cut_function = polynomial_with_limit(
            *self.coefficients,
            floor = self.floor,
            ceiling = self.ceiling
        )

    def __call__(self, mass):
        return self._cut_function(mass)

    def __getstate__(self):
        """when pickling, use dataclasses.asdict to avoid attempting to pickle
        the transient local functions"""
        return asdict(self)

    def __setstate__(self, d):
        """when unpickling, make sure to call __init__ so that __post_init__
        gets called as well and then transient local functions are recreated"""
        self.__init__(**d)


@dataclass
class Selections:
    y0_cut : CutByMass
    vprojsig: CutByMass
    reco_category: str
    cr_range : tuple = (1.9, 2.4)
    sr_range : tuple = (1.0, 1.9)
    target_pos : float = -4.3
    excl_mass_window: float = 2.8
    mass_window: float = 1.5
    mass_sideband: float = 4.5


    def mass_resolution(self, mass):
        """return the mass resolution for the input mass given
        our knowledge of the reco category that these selections are for"""
        mr_func = (
            mass_resolution.tom_2016_simps_l1l1
            if self.reco_category == 'l1l1' else
            mass_resolution.tom_2016_simps_l1l2
        )
        return mr_func(mass)


    def __call__(self, events):
        """apply the selections to the events and return the necessary categories in a SelectionSet"""
        pele = np.sqrt(
                events['ele.track_.px_']**2
                +events['ele.track_.py_']**2
                +events['ele.track_.pz_']**2
        )
        ppos = np.sqrt(
                events['pos.track_.px_']**2
                +events['pos.track_.py_']**2
                +events['pos.track_.pz_']**2
        )
        psum = pele+ppos
        ele_y0 = events['ele.track_.z0_']
        pos_y0 = events['pos.track_.z0_']
        min_y0 = np.minimum(abs(ele_y0), abs(pos_y0))
        vtx_proj_sig = events['vtx_proj_sig']
        z = events['vertex.pos_'].fZ
        invm = events['vertex.invM_']*1000
    
        both_l1 = events.eleL1&events.posL1
        either_l1 = (~both_l1)&(events.eleL1|events.posL1)
        rc = both_l1 if self.reco_category == 'l1l1' else either_l1
        cr = (psum > self.cr_range[0])&(psum < self.cr_range[1])
        sr = (psum > self.sr_range[0])&(psum < self.sr_range[1]) 

        return SelectionSet(
            reco_category = rc,
            cr = cr,
            sr = sr,
            vtx_proj_sig = vtx_proj_sig < self.vprojsig(invm),
            min_y0 = (min_y0 > self.y0_cut(invm)),
            after_target = (z > self.target_pos),
            aliases = {
                'preselection': ['reco_category','sr'],
                'exclusion': [
                    'reco_category',
                    'sr',
                    'vtx_proj_sig',
                    'min_y0',
                    'after_target'
                ],
                'search': ['reco_category','sr', 'vtx_proj_sig', 'after_target']
            }
        )


def z_h(*axes, prefix = None, **axis_kw):
    label = axis_kw.get('label','$z$ / mm')
    if prefix is not None:
        label = f'{prefix} {label}'
    axis_kw['label'] = label
    return hist.dask.Hist(
        *axes,
        hist.axis.Regular(
            250,-4.3,245.7,**axis_kw
        )
    )


def process_signal(selections, events, mass):
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
    sl = selections(events)

    invm = events['vertex.invM_']*1000
    sigma_m = selections.mass_resolution(mass)
    invm_pull = abs(invm - mass)/sigma_m

    cats = {
        'pre': sl.preselection,
        'final': sl.exclusion&(invm_pull < selections.excl_mass_window)
    }

    o = {}
    o['z'] = z_h(hist.axis.StrCategory(list(cats.keys())), prefix='Truth')
    o['energy'] = (
        hist.dask.Hist.new
        .StrCategory(list(cats.keys()))
        .Reg(230,0,2.3,label=r'True $V_D$ Energy / GeV')
        .Double()
    )
    for name, cat in cats.items():
        o['z'].fill(name, events[cat]['true_vd.vtx_z_'])
        o['energy'].fill(name, events[cat]['true_vd.energy_'])

    ele_y0 = events['ele.track_.z0_']
    pos_y0 = events['pos.track_.z0_']
    min_y0 = np.minimum(abs(ele_y0), abs(pos_y0))

    o['invm_pull_vs_min_y0'] = (
        hist.dask.Hist.new
        .Reg(100,0,10,label=r'$|m_\text{reco}-m_\text{true}|/\sigma_m$')
        .Reg(800,0,4,label=r'$\min(|y_0^{e^-}|,|y_0^{e^+}|)$')
        .Double()
    )
    o['invm_pull_vs_min_y0'].fill(invm_pull[sl.search], min_y0[sl.search])
    o['invm_vs_min_y0'] = (
        hist.dask.Hist.new
        .Reg(250,0,250,label=r'$m_\text{reco}$')
        .Reg(800,0,4,label=r'$\min(|y_0^{e^-}|,|y_0^{e^+}|)$')
        .Double()
    )
    o['invm_vs_min_y0'].fill(invm[sl.search], min_y0[sl.search])

    
    return o


def process_data(selections, events):
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
    sl = selections(events)
    invm = events['vertex.invM_']*1000
    ele_y0 = events['ele.track_.z0_']
    pos_y0 = events['pos.track_.z0_']
    min_y0 = np.minimum(abs(ele_y0), abs(pos_y0))

    h = {}
    h['cr'] = hist.dask.Hist.new.Reg(220,0,220,label=r'$m_\text{reco}$ / MeV').Double()
    h['cr'].fill(
        events[sl('reco_category','cr')]['vertex.invM_']*1000
    )
    h['invm_vs_min_y0'] = (
        hist.dask.Hist.new
        .Reg(250,0,250,label=r'$m_\text{reco}$')
        .Reg(800,0,4,label=r'$\min(|y_0^{e^-}|,|y_0^{e^+}|)$')
        .Double()
    )
    h['invm_vs_min_y0'].fill(invm[sl.search], min_y0[sl.search])

    for mass in range(20,126,2):
        h[mass] = {}

        sigma_m = selections.mass_resolution(mass)
        invm_pull = abs(invm - mass)/sigma_m
        
        excl_sl = sl.exclusion&(invm_pull < selections.excl_mass_window)
        h[mass]['z'] = events['vertex.pos_'].fZ[excl_sl]


        h[mass]['invm_pull_vs_min_y0'] = (
            hist.dask.Hist.new
            .Reg(100,0,10,label=r'$|m_\text{reco}-m_\text{true}|/\sigma_m$')
            .Reg(800,0,4,label=r'$\min(|y_0^{e^-}|,|y_0^{e^+}|)$')
            .Double()
        )
        h[mass]['invm_pull_vs_min_y0'].fill(
            invm_pull[sl.search],
            min_y0[sl.search]
        )

    return h


def fill_histograms(selections, data_filter):
    print('Constructing Task Graph')
    def process(name, tree):
        events = uproot.dask(tree, open_files = False)
        if name.startswith('simp'):
            return process_signal(selections, events, int(name[4:]))
        elif name == 'data':
            return process_data(selections, events)
        else:
            return ()
    o = {
        name: process(name, tree)
        for name, tree in tqdm(sample_list(test = False, data_filter = data_filter))
    }

    print('Computing Task Graph')
    with ProgressBar():
        o, = dask.compute(o)

    return o


def run(
    output: str,
    data_filter: str,
    **selection_kw
):
    selections = Selections(**selection_kw)
    o = fill_histograms(selections, data_filter)
    o['selections'] = selections

    with open(output, 'wb') as f:
        pickle.dump(o, f)

    print('Search')
    # deduce mass range from histogram availability
    # go up until mass+mass_sideband*resolution is above axis maximum
    sampled_mass = o['data']['invm_vs_min_y0'].axes[0].edges
    search_max = sampled_mass[(
        np.argmax(sampled_mass+selections.mass_sideband*selections.mass_resolution(sampled_mass) > sampled_mass[-1])
    )]

    o['search'] = search.invm_y0(
        mass = np.arange(20,search_max,1),
        data = o['data']['invm_vs_min_y0'],
        y0_edges = search.deduce_y0_edges_from_y0_cut(
            o['data']['invm_vs_min_y0'],
            selections.mass_window,
            selections.y0_cut
        ),
        invm_edges = (selections.mass_window, selections.mass_sideband),
        n_trials = 50000
    )
    
    with open(output, 'wb') as f:
        pickle.dump(o, f)

    print('Exclusion')
    def get_mean_energy(m):
        energy_h = o[f'simp{m}']['energy']
        mean, stdd, merr = weightedmean(energy_h.axes[1].centers, energy_h['pre',:].values())
        return mean
    
    o['excl_estimate'] = exclusion_estimate(
        mass = np.arange(20,126,2),
        eps2 = np.logspace(-12,-2,50),
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
    
    with open(output, 'wb') as f:
        pickle.dump(o, f)


def annotate(*args, i_axis=0, **kwargs):
    def _annotation_impl(fig, axes):
        axes[i_axis].annotate(*args, **kwargs)
    return _annotation_impl


def plot(
    evaluation: str|Path,
    out_dir: str|Path = Path.cwd(),
    label: List[str] = []
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
    
    if 'search' in r:
        plot_invm_miny0 = functools.partial(
            r['data']['invm_vs_min_y0'].plot,
            cbarpad=0.4,
            norm='log'
        )
        
        search.show(
            r['search'],
            extras = annotate(
                '\n'.join([
                    'Run 7800',
                    f'${r["selections"].mass_window}\sigma$ InvM Window',
                    f'${r["selections"].mass_sideband}\sigma$ InvM Sideband'
                ]+label),
                xy=(0.5,0.95),
                xycoords='axes fraction',
                ha='center', va='top',
                size='x-small'
            ),
            legend_kw = dict(title='SR L1L2'),
            lumi = lumi.data.lumi,
            display = False,
            filename = out_dir / 'search.pdf'
        )
    
        mass_with_min_pval = r['search']['mass'][np.argmin(r['search']['p_value'])]
        plot_invm_miny0()
        search.show_with_calculation(
            mass_with_min_pval,
            r['search'],
            lumi = lumi.data.lumi,
            display = False,
            filename = out_dir / f'min-p-val-search-calculation-{mass_with_min_pval}.pdf'
        )
    
    ee = r['excl_estimate']
    plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.max_allowed, 'Max Signal Allowed Estimate',
        vmax = 10
    )
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top', color='white'
    )
    show(filename=out_dir / 'max-signal-allowed.pdf')
    
    plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.expected, 'Expected Signal',
        vmax = 0.5
    )
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top', color='white'
    )
    show(filename=out_dir / 'expected-signal.pdf')
    
    im, cbar = plt_mass_by_eps2(
        ee.mass, ee.eps2, ee.expected/ee.max_allowed, 'Expected / Max Allowed',
        # norm=mpl.colors.TwoSlopeNorm(vcenter=1.),
        cmap='Blues',
        vmax=0.1
    )
    excl_level = lumi.data.lumi/10.7
    cbar.ax.axhline(excl_level, color='tab:red')
    plt.contour(ee.mass, ee.eps2, (ee.expected/ee.max_allowed).T, [excl_level], colors='tab:red')
    plt.annotate(
        '\n'.join(label),
        xy=(0.95,0.95), xycoords='axes fraction', ha='right', va='top'
    )
    show(filename=out_dir / 'exclusion-estimate.pdf')
