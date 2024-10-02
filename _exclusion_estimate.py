"""exclusion estimate specialized for SIMPs search"""

from collections import namedtuple


import numpy as np
from tqdm import tqdm
import hist

from .exclusion import optimum_interval_method as oim
from .exclusion import production
from .exclusion.models import ctau, simp
from ._signal_yield import signal_yield


ExclusionEstimateResult = namedtuple(
        'ExclusionEstimateResult',
        ['mass', 'eps2', 'z',
        'diff_yield','expected','max_allowed',
        'signal_efficiency']
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
    mass,
    eps2,
    z,
    invm_cr_h,
    final_selection_counts_by_mass_h,
    mean_energy_GeV_by_mass,
    data_z,
    simp_parameters = {},
):
    _oim_table = oim.load_or_new(
        mu_values = np.concatenate([
            np.arange(0,10,0.1),
            np.arange(10,100,1.0)
        ]),
        n_trials = 10_000
    )
    total_prompt_signal_yield_per_eps2 =  production.from_calculators(
        production.radiative_fraction.alic_2016_simps,
        production.TridentDifferentialProduction.from_hist(invm_cr_h),
        production.radiative_acceptance.alic_2016_simps
    )
    model = simp.Parameters(**simp_parameters)
    eff = np.full(mass.shape+z.centers.shape, np.nan)
    diff_yield = np.full(mass.shape+eps2.shape+z.centers.shape, np.nan)
    max_signal_allowed = np.full(mass.shape+eps2.shape, np.nan)
    expected_signal = np.full(mass.shape+eps2.shape, 0.0)
    for i_mass, m in tqdm(enumerate(mass), total=len(mass)):
        diff_yield[i_mass,...], eff[i_mass,...] = signal_yield(
            mass = m,
            eps2 = eps2,
            z = z,
            prompt_signal_yield_per_eps2 = (
                total_prompt_signal_yield_per_eps2(m*model.mass_ratio_Ap_to_Vd)
            ),
            final_selection_counts_h = final_selection_counts_by_mass_h[m],
            mean_energy_GeV = mean_energy_GeV_by_mass[m],
            model = model
        )
        yield_cdf_lut = construct_yield_cdf_lut(diff_yield[i_mass,...])
        expected_signal[i_mass,:] = np.sum(diff_yield[i_mass,...], axis=1)
        data_x = (
            yield_cdf_lut[..., z.index(data_z[m])]
            if len(data_z[m]) > 0 else np.full((*yield_cdf_lut.shape[:-1],0), 0.0)
        )
        max_signal_allowed[i_mass,:] = oim.max_signal_strength_allowed(data_x, confidence_level=0.9)

    return ExclusionEstimateResult(
        mass = mass,
        eps2 = eps2,
        z = z.centers,
        diff_yield = diff_yield,
        expected = expected_signal,
        max_allowed = max_signal_allowed,
        signal_efficiency = eff
    )

