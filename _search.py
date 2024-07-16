"""basic search looking for excess of events"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

import hist

from .exclusion.production.mass_resolution import alic_2016_simps as default_mass_resolution


def _process_edges_arg(edges):
    if isinstance(edges, (tuple,list)):
        if len(edges) != 2:
            raise ValueError('Edge boundaries need to be a pair')
        return lambda m: edges
    return edges


def _edges_to_slices(*edges, share = True):
    if share:
        return (
            slice(hist.loc(lo), hist.loc(up), sum)
            for lo, up in zip(edges, edges[1:])
        )

    if len(edges) % 2 == 1:
        raise ValueError('If not sharing edges, the number of edges must be even!')
    
    return (
        slice(hist.loc(edges[i]), hist.loc(edges[i+1]), sum)
        for i in range(0, len(edges), 2)
    )


@dataclass
class deduce_y0_edges_from_y0_cut:
    data: hist.Hist
    window: float
    y0_cut_f: Callable
    mass_resolution: Callable = default_mass_resolution
      

    def __call__(self, mass):
        sigma_m = self.mass_resolution(mass)
        sr_mass, = _edges_to_slices(mass-self.window*sigma_m, mass+self.window*sigma_m)
        y0_cut_v = self.y0_cut_f(mass)
        # accumulate mass counts along /flipped/ axis (accumulate and flip)
        # find the index for the entry that first goes above 1000 (argmax over >)
        # select bin in this index _from the end_ (since we accumulated on the flip)
        i_y0_floor = -np.argmax(np.add.accumulate(np.flip(self.data[sr_mass,:].values()))>1000)
        return self.data.axes[1].centers[i_y0_floor], y0_cut_v


def invm_y0(
    mass,
    data,
    *,
    y0_edges = (0.1,1.0),
    invm_edges = (2,6),
    n_trials = 10_000,
    apply_mass_resolution = True
):
    """data is histogram of counts in InvM vs Min-y0 space"""

    # convert static tuples into functions that are callable by m
    y0_edges = _process_edges_arg(y0_edges)
    invm_edges = _process_edges_arg(invm_edges)

    mass = np.array(mass)
    search_result = np.full(
        mass.shape,
        np.nan,
        dtype = [
            (field, float)
            for field in [
                'mass', 'y0_floor', 'y0_cut',
                'invm_left', 'invm_sr_left',
                'invm_sr_right', 'invm_right',
                'a', 'b', 'c', 'd', 'e', 'ae', 'bd',
                'f_exp', 'f_unc', 'f_obs', 'p_value'
            ]
        ]
    )

    for i, m in tqdm(enumerate(mass), total=len(mass)):
        y0_floor, y0_cut = y0_edges(m)
        window, sideband = invm_edges(m)
        if apply_mass_resolution:
            sigma_m = default_mass_resolution(m)
            window *= sigma_m
            sideband *= sigma_m

        (
            low_mass_sideband,
            sr_mass,
            high_mass_sideband
        ) = _edges_to_slices(m-sideband,m-window,m+window,m+sideband)

        (
            low_y0_sideband,
            sr_y0
        ) = _edges_to_slices(y0_floor, y0_cut, None)

        a, b, c, d, e, f_obs = (
            data[low_mass_sideband, sr_y0],
            data[low_mass_sideband, low_y0_sideband],
            data[sr_mass, low_y0_sideband],
            data[high_mass_sideband, low_y0_sideband],
            data[high_mass_sideband, sr_y0],
            data[sr_mass, sr_y0]
        )

        ae = max(a+e, 0.4)
        bd = b+d
        
        f_exp = c*(ae/bd)

        ae_s = np.random.poisson(lam=ae, size=n_trials)
        bd_s = np.random.normal(loc=bd, scale=np.sqrt(bd), size=n_trials)
        c_s = np.random.normal(loc=c, scale=np.sqrt(c), size=n_trials)
        f_s = np.random.poisson(lam=c_s*(ae_s/bd_s))
        f_unc = np.std(f_s)
        p_value = np.sum(f_s > f_obs)/n_trials
    
        search_result[i] = (
            m,
            y0_floor, y0_cut,
            m-sideband, m-window, m+window, m+sideband,
            a, b, c, d, e,
            ae, bd,
            f_exp, f_unc, f_obs, p_value
        )

    return search_result
        
