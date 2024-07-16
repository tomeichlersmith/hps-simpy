"""basic search looking for excess of events"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

import hist

from .exclusion.production.mass_resolution import alic_2016_simps as default_mass_resolution
from .plot import show as _show
from .plot import plt


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


def show(
    result, *,
    extras = None,
    **show_kw
):
    """Given a array of search results, plot them showing the observed and expected
    in an upper panel and the p-value from the toys in a lower panel"""
    
    fig, (raw, pval) = plt.subplots(
        nrows=2,
        height_ratios = [2,1],
        gridspec_kw = dict(
            hspace = 0.05
        ),
        sharex = 'col'
    )
    
    raw.errorbar(
        result['mass'], result['f_exp'], yerr = result['f_unc'],
        marker='o',
        linewidth=0, elinewidth=2,
        color='tab:blue', label='Expected'
    )
    raw.scatter(result['mass'], result['f_obs'], color='black', label='Observed')
    raw.legend(title='SR L1L1')
    raw.set_ylabel('Events')
    
    pval.scatter(result['mass'], result['p_value'], color='black')
    pval.set_yscale('log')
    pval.set_xlabel('Invariant Mass / MeV')

    approximate_number_of_independent_search_regions = (
        (np.max(result['mass'])-np.min(result['mass']))
        /np.mean(default_mass_resolution(result['mass']))
    )
    
    for p in 1-np.array([0.6827,0.9545,0.997]):
        pval.axhline(p, linestyle=':', color='gray')
        pval.axhline(p/approximate_number_of_independent_search_regions,
                     linestyle=':', color='tab:red')
    if extras is not None:
        extras(fig, (raw, pval))
    _show(ax=raw,**show_kw)


def show_with_calculation(
    mass, result, *,
    extra_notes = [],
    **show_kw
):
    """Show the InvM vs min-y0 histogram along with a specific search calculation
    drawn on top

    Example
    -------

        invm_miny0_h.plot(norm='log', cbarpad=0.4)
        show_with_calculation(60, result_from_search)
    """
    (
        mass, y0_floor, y0_cut,
        invm_left, invm_sr_left,
        invm_sr_right, invm_right,
        a, b, c, d, e,
        ae, bd,
        f_exp, f_unc, f_obs, p_value 
    ) = result[result['mass']==mass][0]

    for x in [invm_left, invm_sr_left, invm_sr_right, invm_right]:
        plt.plot(3*[x], [y0_floor, y0_cut, 2], color='tab:red')
    
    for y in [y0_floor, y0_cut]:
        plt.plot([invm_left, invm_right], 2*[y], color='tab:red')
    
    for name, xy in zip(
        'ABCDEF',
        [
            (0.5*(invm_left+invm_sr_left), y0_cut),
            (0.5*(invm_left+invm_sr_left), y0_floor),
            (0.5*(invm_sr_left+invm_sr_right), y0_floor),
            (0.5*(invm_right+invm_sr_right), y0_floor),
            (0.5*(invm_right+invm_sr_right), y0_cut),
            (0.5*(invm_sr_left+invm_sr_right), y0_cut),            
        ],
        strict=True
    ):
        plt.annotate(
            f'{name}', xy=xy,
            ha='center', va='bottom',
            color='tab:red'
        )
    
    plt.annotate(
        '\n'.join(extra_notes+[
            r'$m_\text{true} = '+f'{mass:.0f}$ MeV',
            ' '.join(f'{name} = {val:.0f}' for name, val in zip('ABCDE', (a,b,c,d,e))),
            r'$F_\text{exp} = C\times(\max(A+E,0.4)/(B+D)) = '+f"{f_exp:.1f}\pm{f_unc:.1f}$",
            r'$F_\text{obs} = '+f'{f_obs:.0f}$',
            f"P Value = {p_value:.1e}"
        ]),
        xy=(0.95,0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        color='black',
        bbox = dict(fill=True, color='white', alpha=0.76)
    )

    _show(**show_kw)