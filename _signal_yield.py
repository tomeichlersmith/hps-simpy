"""signal yield estimate specialized for SIMPs search"""


from collections import namedtuple


import numpy as np
from tqdm import tqdm
import hist

from .exclusion.models import ctau, simp


def _get_sampled_z_by_mass():
    import uproot
    from . import get_true_vd_z_file

    the_map = {}
    
    with uproot.open(get_true_vd_z_file()) as f:
        for mass in range(20,126,2):
            sampled_z = f[f'{mass}/true_z_h'].values(flow=True)
            sampled_z[sampled_z==0] = 1
            the_map[mass] = sampled_z

    return the_map


def normalize_along_z(sign_h, sampled_z):
    """convert histogram of counts into efficiency along z-axis

    assuming z-axis is axis 0"""

    h = sign_h.copy()
    counts = h.values(flow=True)
    # the swapping of axes gets the z-axis to line up during the division
    # and then returns it to position 0
    # this works for situations where there aren't other axes as well
    counts = np.swapaxes(np.swapaxes(counts,0,-1)/sampled_z, 0,-1)
    h[...] = counts
    return h


def signal_yield(*,
    mass,
    eps2,
    z,
    prompt_signal_yield_per_eps2,
    final_selection_counts_h,
    mean_energy_GeV,
    model = simp.Parameters(),
):
    if getattr(signal_yield, 'sampled_z', None) is None:
        signal_yield.sampled_z = _get_sampled_z_by_mass()

    eff_h = normalize_along_z(final_selection_counts_h, signal_yield.sampled_z[mass])
    # keep all flow bins except those along the zero'th (z) axis
    eff_wf = eff_h.values(flow=True)
    eff = eff_wf[1:-1,...]

    mean_gamma = mean_energy_GeV*1000 / mass
    decay_gct_eps2_rho = mean_gamma*ctau(model.rate_Vd_decay_2l_eps2(mass, rho=True))
    decay_gct_eps2_phi = mean_gamma*ctau(model.rate_Vd_decay_2l_eps2(mass, rho=False))
    decay_weight = (
        model.br(model.rate_Vrho_pi, mass)*eps2/decay_gct_eps2_rho*np.exp(
            np.multiply.outer(-4.3-z.centers, eps2)/decay_gct_eps2_rho
        )
        +
        model.br(model.rate_Vphi_pi, mass)*eps2/decay_gct_eps2_phi*np.exp(
            np.multiply.outer(-4.3-z.centers, eps2)/decay_gct_eps2_phi
        )
    )
    Nprompt = eps2*prompt_signal_yield_per_eps2
    # want to return differential yield indexed by (eps2, z, ...extras...)
    # where ...extras... is omitted if they weren't given with the counts
    #   EXPRESSION                        INDEX
    #   eff                               (z,) OR (z, ...extras...)
    #   Nprompt                           (eps2,)
    #   decay_weight                      (z, eps2)
    #   eff*z.widths[:,np.newaxis]        (z, ...extras...)
    #   Nprompt*decay_weight              (z, eps2)
    if eff.ndim == 1:
        # no ...extras..., so a simple transpose will suffice
        return np.transpose(Nprompt*decay_weight)*(eff*z.widths)
    # some ...extras..., need to do some newaxis stuff to tell np where to broadcast
    return (
        (eff*z.widths[:,np.newaxis])[np.newaxis,...]
        *np.transpose(Nprompt*decay_weight)[:,:,np.newaxis]
    )
