"""various plotting helpers"""
from pathlib import Path

import numpy as np
import awkward as ak

import hist
import matplotlib as mpl
import mplhep
import mplhep.error_estimation
mpl.style.use(mplhep.style.ROOT)
import matplotlib.pyplot as plt


def define_known_variance(d):
    """I think since dask split the filling over multiple processes,
    the variance knowledge is lost for data
    We just tell it we do actually know the variance,
    forcing it to use the values as the variance
    which we know is correct in this case since they are unweighted counts
    """
    if isinstance(d, hist.Hist):
        d._variance_known = True
    elif isinstance(d, dict):
        for v in d.values():
            define_known_variance(v)
    elif isinstance(d, (list,tuple)):
        for v in d:
            define_known_variance(v)
    elif isinstance(d, (ak.Array,np.ndarray)):
        return
    else:
        raise ValueError(f'Unknown object while walking tree {d}')


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


def efficiency_plot(counts_h, denominator = None, **kwargs):
    import scipy
    for disallowed in ['w2method','w2','density','yerr']:
        if disallowed in kwargs:
            raise KeyError(f'Cannot manually pass {disallowed} to efficiency_plot')
    k = counts_h.values() if isinstance(counts_h, hist.Hist) else counts_h
    n = denominator if denominator is not None else k[0]
    # Eq 11 from https://arxiv.org/pdf/physics/0701199v1
    # defines the probability of a true efficiency epsilon
    # given the observation of k events passing out of n events
    # we can use this probability distribution to find the confidence
    # bands around our estimate of efficiency
    # Integrating Eq 11 from 0 up to some upper limit x is
    # equal to the regularized incomplete beta function
    #   scipy.special.betainc
    # with parameters
    #   a = k+1 and b = n-k+1
    # The inverse of this function is also defined in scipy
    #  scipy.special.betainc
    # which we can use to find different values of the efficiency
    # that correspond to upper limits depending on the probability
    # that the true efficiency is below it
    def efficiency_upper_limit(probability):
        return scipy.special.betaincinv(k+1, n-k+1, np.array(probability)[:,np.newaxis])

    # also as suggested by the reference above,
    # use the mode estimator for the estimate of the true efficiency
    eff = k/n

    # find the 1sigma = 68.3% confidence band
    # by starting with a symmetric assumption (68.3/2% on each side of the mode)
    # and then clipping these probabilities depending on the probability that
    # the true efficiency is below the mode
    coverage = 0.683

    prob_true_below_mode = scipy.special.betainc(k+1, n-k+1, eff)

    lowlim = np.maximum(prob_true_below_mode-coverage/2, 0.0)
    uplim  = lowlim+coverage

    over = uplim>1.0
    lowlim[over] = uplim[over]-coverage
    uplim[over] = 1.0
    
    lims = efficiency_upper_limit([lowlim, uplim])
    yerr = abs(lims - np.vstack((eff,eff)))

    # pass axes information to plotting if possible
    if 'bins' not in kwargs and isinstance(counts_h, hist.Hist):
        eff_h = hist.Hist(*counts_h.axes)
        eff_h[:] = eff
        return mplhep.histplot(eff_h, yerr = yerr, **kwargs)

    # otherwise just have user supply binning with bins kwarg
    return mplhep.histplot(eff, yerr = lim, **kwargs)



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
    if isinstance(s, (list,tuple)):
        s = '\n'.join(s)
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
        plt.close()


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
