"""various plotting helpers"""

import numpy as np

# since we are in a container that redefines HOME, we need to explicitly
# tell MPL where to store its config and cache
import os
os.environ['MPLCONFIGDIR'] = '/sdf/home/e/eichl008/.config/matplotlib'

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

