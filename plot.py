"""various plotting helpers"""
from pathlib import Path

import numpy as np

import matplotlib as mpl
import mplhep
import mplhep.error_estimation
mpl.style.use(mplhep.style.ROOT)
import matplotlib.pyplot as plt


def poisson_interval_ignore_empty(sumw, sumw2):
    interval = mplhep.error_estimation.poisson_interval(sumw, sumw2)
    lo, hi = interval[0,...], interval[1,...]
    to_ignore = np.isnan(lo)
    lo[to_ignore] = 0.0
    hi[to_ignore] = 0.0
    return np.array([lo,hi])


def histplot(h, **kwargs):
    for disallowed in ['bins','w2method','w2']:
        if disallowed in kwargs:
            raise KeyError(f'Cannot manually pass {disallowed} to our histplot')
    values, variances = h.values(), h.variances()
    return mplhep.histplot(
        values,
        bins = h.axes[0].edges,
        w2method = poisson_interval_ignore_empty,
        w2 = variances,
        **kwargs
    )


def plot2d(*args, cbarlabel = None, **kwargs):
    kwargs['flow'] = None
    kwargs['cbarpad'] = 0.4
    art = mplhep.hist2dplot(*args, **kwargs)
    art.cbar.set_label(cbarlabel)
    return art


def annotate(s, loc = 'upper right', **kwargs):
    if isinstance(loc, str):
        if loc == 'upper right':
            kwargs.update({
                'xy': (0.95,0.95),
                'xycoords': 'axes fraction',
                'ha': 'right',
                'va': 'top'
            })
        else:
            raise ValueError(f'loc={loc} is not a known location name')
    elif isinstance(loc, tuple):
        kwargs['xy'] = loc
    else:
        raise ValueError(f'Value of loc {loc} not understood')
    return plt.annotate(s, **kwargs)


def show(
    ax = None,
    filename = None,
    display = True,
    exp_loc = 0,
    lumi = 10.7
):
    if isinstance(lumi, (float,int)):
        lumi = f'${lumi} '+r'pb^{-1}$'
    mplhep.label.lumitext(lumi, ax = ax)
    mplhep.label.exp_text('HPS','Internal','2016', loc=exp_loc, ax=ax)
    if filename is not None:
        plt.savefig(
            filename,
            bbox_inches='tight'
    )
    if display:
        plt.show()
    else:
        plt.clf()


def plt_mass_by_eps2(
    mass,
    eps2,
    arr,
    label,
    **kwargs
):
    plt.figure(figsize=(12,10))
    ax = plt.gca()
    im = ax.pcolormesh(mass, eps2, arr.T, **kwargs)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\epsilon^2$')
    ax.set_xlabel('Invariant Mass / MeV')
    cbar = plt.colorbar(im, label=label)
    return im, cbar


def _from_options(options, *, run = None, quiet = False):
    # sort options by last modified time
    options.sort(key = lambda f: f.stat().st_mtime)
    if run is None:
        run = options[-1]
    elif isinstance(run, int):
        run = options[run]
    elif isinstance(run, str):
        match = [ o for o in options if o.stem == run ]
        if len(match) == 0:
            raise ValueError(f'No run matching {run} in {options}.')
        elif len(match) > 1:
            raise ValueError(f'More than one matching {run} in {options}.')
        run = match[0]
    elif not isinstance(run, Path):
        raise TypeError(f'{run} is not None, an int, str, or Path.')
    if not quiet:
        print(f'Selected {run}')
    return run


def from_dir(dir, *, suffix = 'pkl', run = None):
    return _from_options(
        [ fp for fp in Path(dir).iterdir() if fp.suffix == f'.{suffix}' ],
        run = run
    )
