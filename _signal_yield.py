"""signal yield estimate specialized for SIMPs search"""


from collections import namedtuple


import numpy as np
from tqdm import tqdm
import hist

from .exclusion.models import ctau, simp


def signal_yield(*,
    mass,
    eps2,
    z,
    prompt_signal_yield_per_eps2,
    final_selection_eff_h,
    mean_energy_GeV,
    model = simp.Parameters(),
):
    # keep all flow bins except those along the zero'th (z) axis
    eff_wf = final_selection_eff_h.values(flow=True)
    eff = eff_wf[1:-1,...]

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
    Nprompt = eps2*prompt_signal_yield_per_eps2
    # eff is indexed by (z, ...extras...)
    # Nprompt is indexed by (eps2)
    # decay_weight is indexed by (z, eps2)
    # return differential yield indexed by (z, eps2, ...extras...)
    return (
        (eff*z.widths[:,np.newaxis])[:,np.newaxis,...]
        *(Nprompt*decay_weight)[:,:,np.newaxis]
    )
