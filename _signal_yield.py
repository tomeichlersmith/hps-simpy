"""signal yield estimate specialized for SIMPs search"""


from collections import namedtuple


import numpy as np
from tqdm import tqdm
import hist

from .exclusion import production
from .exclusion.models import ctau, simp


SignalYield = namedtuple(
        'SignalYield',
        [
            'mass', 'eps2', 'z',
            'diff_yield','expected'
        ]
)


def signal_yield(*,
    mass,
    eps2,
    z,
    invm_cr_h,
    final_selection_eff_by_mass_h,
    mean_energy_GeV_by_mass,
    simp_parameters = {}
):
    total_prompt_signal_yield_per_eps2 =  production.from_calculators(
        production.radiative_fraction.alic_2016_simps,
        production.TridentDifferentialProduction.from_hist(invm_cr_h),
        production.radiative_acceptance.alic_2016_simps
    )
    model = simp.Parameters(**simp_parameters)
    diff_yield = np.full(mass.shape+eps2.shape+z.centers.shape, np.nan)
    expected_signal = np.full(mass.shape+eps2.shape, 0.0)
    for i_mass, m in tqdm(enumerate(mass), total=len(mass)):
        beta_F_z = final_selection_eff_by_mass_h[m].values()
        mean_gamma = mean_energy_GeV_by_mass[m]*1000 / m
        decay_gct_eps2_rho = mean_gamma*ctau(model.rate_Vd_decay_2l_eps2(m, rho=True))
        decay_gct_eps2_phi = mean_gamma*ctau(model.rate_Vd_decay_2l_eps2(m, rho=False))
        decay_weight = (
            model.br(model.rate_Vrho_pi, m)*eps2/decay_gct_eps2_rho*np.exp(
                np.multiply.outer(-4.3-z.centers, eps2)/decay_gct_eps2_rho
            )
            +
            model.br(model.rate_Vphi_pi, m)*eps2/decay_gct_eps2_phi*np.exp(
                np.multiply.outer(-4.3-z.centers, eps2)/decay_gct_eps2_phi
            )
        )
        Nprompt = eps2*total_prompt_signal_yield_per_eps2(m*model.mass_ratio_Ap_to_Vd)
        diff_yield[i_mass,...] = np.transpose(Nprompt*decay_weight)*beta_F_z*z.widths
        expected_signal[i_mass,:] = np.sum(diff_yield[i_mass,...], axis=1)

    return SignalYield(
        mass = mass,
        eps2 = eps2,
        z = z.centers,
        diff_yield = diff_yield,
        expected = expected_signal,
    )

