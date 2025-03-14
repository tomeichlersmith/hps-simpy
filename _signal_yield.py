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


def signal_yield_from_eff(*,
    mass,
    eps2,
    z,
    prompt_signal_yield_per_eps2,
    eff,
    mean_energy_GeV,
    model = simp.Parameters()
):
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


def signal_yield_from_events(*,
    mass,
    eps2,
    z,
    prompt_signal_yield_per_eps2,
    events,
    model = simp.Parameters()
):
    """The np.digitize function is not dask-capable, so we can't use dask
    arrays for the events. That's okay, the signal samples are small enough
    to hold an entire one in memory at a time.

    """
    if len(events) == 0:
        return np.full(eps2.shape+z.centers.shape, 0.0), np.full(z.centers.shape, 0.0)

    # 1D arrays indexed by event
    vd_gamma = events['true_vd.energy_'].to_numpy()*1000/mass
    vd_decay = events['true_vd.vtx_z_'].to_numpy()
    i_zbin = z.index(vd_decay) # not dask compatible

    if getattr(signal_yield_from_events, 'sampled_z', None) is None:
        signal_yield_from_events.sampled_z = _get_sampled_z_by_mass()

    sim_counts = signal_yield_from_events.sampled_z[mass]
    
    # (event,)
    sim_sample_weight = 1/sim_counts[i_zbin]

    # (event,)
    decay_gct_eps2_rho = vd_gamma*ctau(model.rate_Vd_decay_2l_eps2(mass, rho=True))
    decay_gct_eps2_phi = vd_gamma*ctau(model.rate_Vd_decay_2l_eps2(mass, rho=False))

    # (event, eps2)
    decay_weight = (
        model.br(model.rate_Vrho_pi, mass)*eps2*np.transpose(np.exp(
            np.multiply.outer(eps2, -4.3-vd_decay)/decay_gct_eps2_rho
        )/decay_gct_eps2_rho)
        +
        model.br(model.rate_Vphi_pi, mass)*eps2*np.transpose(np.exp(
            np.multiply.outer(eps2, -4.3-vd_decay)/decay_gct_eps2_phi
        )/decay_gct_eps2_phi)
    )

    # (eps2,)
    Nprompt = eps2*prompt_signal_yield_per_eps2

    # (eps2, event)
    total_weight = sim_sample_weight*np.transpose(decay_weight*Nprompt)

    # histogram total_weight along events into z to get differential yield
    nbins = z.size
    scaled_idx = (
        nbins*np.arange(eps2.shape[0])[:,np.newaxis]
        +np.vstack(eps2.shape[0]*(i_zbin,))
    )
    diff_yield = np.bincount(
        scaled_idx.ravel(),
        weights = total_weight.ravel(),
        minlength = nbins*eps2.shape[0]+1
    )[:-1]
    diff_yield.shape = eps2.shape+(nbins,)
    sig_eff = np.bincount(
        i_zbin,
        weights = sim_sample_weight,
        minlength = nbins
    )
    return diff_yield, sig_eff


def signal_yield(*,
    final_selection_counts_h,
    mass,
    **kwargs,
):
    if getattr(signal_yield, 'sampled_z', None) is None:
        signal_yield.sampled_z = _get_sampled_z_by_mass()

    eff_h = normalize_along_z(final_selection_counts_h, signal_yield.sampled_z[mass])
    # keep all flow bins except those along the zero'th (z) axis
    eff_wf = eff_h.values(flow=True)
    eff = eff_wf[1:-1,...]
    return signal_yield_from_eff(
        eff = eff,
        mass = mass,
        **kwargs
    ), eff
