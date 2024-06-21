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


def show(
    ax = None,
    filename = None,
    display = True,
    exp_loc = 0
):
    mplhep.label.lumitext('$10.7 pb^{-1}$', ax = ax)
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
