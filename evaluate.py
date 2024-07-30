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

from dataclasses import dataclass
from typing import Callable

@dataclass
class Selections:
    y0_cut : list[float]
    vprojsig: float
    reco_category: str
    cr_range : tuple = (1.9, 2.4)
    sr_range : tuple = (1.0, 1.9)
    target_pos : float = -4.3
    excl_mass_window: float = 2.8
    mass_window: float = 1.5
    mass_sideband: float = 4.5



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
        the_y0_cut = polynomial(*self.y0_cut)
        return SelectionSet(
            reco_category = rc,
            cr = cr,
            sr = sr,
            vtx_proj_sig = vtx_proj_sig < self.vprojsig,
            min_y0 = (min_y0 > the_y0_cut(invm)),
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
    sigma_m = mass_resolution.alic_2016_simps(mass)
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

        sigma_m = mass_resolution.alic_2016_simps(mass)
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
        np.argmax(sampled_mass+selections.mass_sideband*mass_resolution.alic_2016_simps(sampled_mass) > sampled_mass[-1])
    )]

    o['search'] = search.invm_y0(
        mass = np.arange(20,search_max,1),
        data = o['data']['invm_vs_min_y0'],
        y0_edges = search.deduce_y0_edges_from_y0_cut(
            o['data']['invm_vs_min_y0'],
            selections.mass_window,
            polynomial(*selections.y0_cut)
        ),
        invm_edges = (selections.mass_window, selections.mass_sideband),
        n_trials = 50000
    )
    
    with open(output, 'wb') as f:
        pickle.dump(o, f)

    print('Exclusion')
    pre_eff, final_eff = {}, {}
    with uproot.open(get_true_vd_z_file()) as f:
        for mass in range(20,126,2):
            sampled_z = f[f'{mass}/true_z_h'].values()
            # set zero values to one since we know the filtered
            # histograms will also be zero if the sampled bin is zero
            sampled_z[sampled_z==0] = 1
            pre_eff[mass] = o[f'simp{mass}']['z']['pre',:]/sampled_z
            final_eff[mass] = o[f'simp{mass}']['z']['final',:]/sampled_z
    
    def get_mean_energy(m):
        energy_h = o[f'simp{m}']['energy']
        mean, stdd, merr = weightedmean(energy_h.axes[1].centers, energy_h['pre',:].values())
        return mean
    
    o['excl_estimate'] = exclusion_estimate(
        mass = np.arange(20,126,2),
        eps2 = np.logspace(-12,-2,50),
        z = pre_eff[20].axes[0],
        invm_cr_h = o['data']['cr'],
        #pre_selection_eff_by_mass_h = pre_eff, # leaving off preselection scale factor
        final_selection_eff_by_mass_h = final_eff,
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
