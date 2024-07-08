"""basic search looking for excess of events"""

from collections import namedtuple

import numpy as np
from tqdm import tqdm

import hist

def _slices_from_edges(edges):
    if len(edges) != 3:
        raise ValueError('Edges need to be fully defined. Use None if you wish to specify one edge to be to +/- infinity')
    return (
        slice(hist.loc(edges[0]),hist.loc(edges[1]),sum),
        slice(hist.loc(edges[1]),hist.loc(edges[2]),sum)
    )


ABCDSearchResult = namedtuple(
    'ABCDSearchResult',
    [
        'a', 'b', 'c',
        'd_exp', 'd_unc', 'd_obs',
        'p_value',
        'n_trials'
    ]
)


def abcd(
    data,
    x_edges,
    y_edges,
    *,
    n_trials = 10_000
):
    low_x, high_x = _slices_from_edges(x_edges)
    low_y, high_y = _slices_from_edges(y_edges)
    a, b, c, d_obs = (
        data[high_x,high_y],
        data[high_x,low_y],
        data[low_x,low_y],
        data[low_x,high_y]
    )
    d_exp = c*(a/b)

    a_s = np.random.poisson(lam=a, size=n_trials)
    b_s = np.random.normal(loc=b, scale=np.sqrt(b), size=n_trials)
    c_s = np.random.normal(loc=c, scale=np.sqrt(c), size=n_trials)
    d_s = np.random.poisson(lam=c_s*(a_s/b_s))
    d_unc = np.std(d_s)
    p_value = np.sum(d_s > d_obs)/n_trials
    return ABCDSearchResult(
        a = a,
        b = b,
        c = c,
        d_exp = d_exp,
        d_unc = d_unc,
        d_obs = d_obs,
        p_value = p_value,
        n_trials = n_trials
    )


def _process_edges_arg(edges):
    if isinstance(edges, (tuple,list)):
        if len(edges) != 2:
            raise ValueError('Edge boundaries need to be a pair')
        return lambda m: edges
    return edges


def invm_y0(
    mass,
    data,
    *,
    y0_edges = (0.2,1.0),
    invm_edges = (2,6),
    n_trials = 10_000
):
    y0_edges = _process_edges_arg(y0_edges)
    invm_edges = _process_edges_arg(invm_edges)

    mass = np.array(mass)
    search_result = np.full(
        mass.shape,
        np.nan,
        dtype = [
            ('mass', float),
            ('y0_floor', float),
            ('y0_cut', float),
            ('window', float),
            ('sideband', float),
            *(
                (field, float)
                for field in ABCDSearchResult._fields
            )
        ]
    )

    for i, m in tqdm(enumerate(mass), total=len(mass)):
        data_h = data(m)
        y0_floor, y0_cut = y0_edges(m)
        window, sideband = invm_edges(m)

        search_result[i] = (
            m,
            y0_floor, y0_cut,
            window, sideband,
            *abcd(
                data_h,
                (0.0, window, sideband),
                (y0_floor, y0_cut, None)
            )
        )

    return search_result
        