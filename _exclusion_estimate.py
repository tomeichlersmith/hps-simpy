"""exclusion estimate specialized for SIMPs search"""

from collections import namedtuple


import numpy as np
from tqdm import tqdm
import hist

from .exclusion import optimum_interval_method as oim
from ._signal_yield import signal_yield


ExclusionEstimateResult = namedtuple(
        'ExclusionEstimateResult',
        ['mass', 'eps2', 'z',
        'diff_yield','expected','max_allowed']
)


def construct_yield_cdf_lut(differential_yield_distribution):
    unnormalized_cumulants = np.add.accumulate(differential_yield_distribution, axis=1)
    total_cumulant = unnormalized_cumulants[...,-1]
    yield_cdf_lut = np.full(unnormalized_cumulants.shape, 1.0)
    yield_cdf_lut[total_cumulant > 0] = np.transpose(
        np.transpose(unnormalized_cumulants[total_cumulant > 0.0]) / total_cumulant[total_cumulant > 0]
    )
    return yield_cdf_lut


def exclusion_estimate(*,
    data_z,
    **signal_yield_kw
):
    sy = signal_yield(**signal_yield_kw)
    _oim_table = oim.load_or_new(
        max_signal_strength = 50.0,
        n_test_mu = 200,
        n_trials = 10_000
    )
    max_signal_allowed = np.full(mass.shape+eps2.shape, np.nan)
    for i_mass, m in tqdm(enumerate(mass), total=len(mass)):
        yield_cdf_lut = construct_yield_cdf_lut(sy.diff_yield[i_mass,...])
        data_x = (
            yield_cdf_lut[..., z.index(data_z[m])]
            if len(data_z[m]) > 0 else np.full((*yield_cdf_lut.shape[:-1],0), 0.0)
        )
        max_signal_allowed[i_mass,:] = oim.max_signal_strength_allowed(
            data_x,
            confidence_level=0.9
        )

    return ExclusionEstimateResult(
        *sy,
        max_allowed = max_signal_allowed,
    )

