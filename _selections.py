"""various types of selections that create selection sets and are pickleable"""

from dataclasses import dataclass, asdict

import numpy as np

from .dask import SelectionSet
from .exclusion.production._polynomial import polynomial
from .exclusion.production import mass_resolution

def polynomial_with_limit(*coefficients, floor = None, ceiling = None):
    fit = polynomial(*coefficients)
    # np.clip is not defined for dask_awkward, so we need to do max/min manually
    if floor is None and ceiling is None:
        return fit
    elif floor is not None and ceiling is not None:
        return lambda x: np.minimum(np.maximum(fit(x), floor), ceiling)
    elif floor is not None:
        return lambda x: np.maximum(fit(x), floor)
    else:
        return lambda x: np.minimum(fit(x), ceiling)    


@dataclass
class CutByMass:
    coefficients: list[float]
    floor : float = None
    ceiling : float = None

    def __post_init__(self):
        """define the cut function callable now that we have been configured

        np.clip is unfortunately not defined for dask_awkward, so
        we have to manually do it ourselves using np.maximum and np.minimum
        """
        self._cut_function = polynomial_with_limit(
            *self.coefficients,
            floor = self.floor,
            ceiling = self.ceiling
        )

    def __call__(self, mass):
        return self._cut_function(mass)

    def __getstate__(self):
        """when pickling, use dataclasses.asdict to avoid attempting to pickle
        the transient local functions"""
        return asdict(self)

    def __setstate__(self, d):
        """when unpickling, make sure to call __init__ so that __post_init__
        gets called as well and then transient local functions are recreated"""
        self.__init__(**d)


@dataclass
class StandardSelections:
    """A few standard selections on top of preselection for studying cut variables"""
    cr_range : tuple = (1.9, 2.4)
    sr_range : tuple = (1.0, 1.9)
    target_pos : float = -4.3
    excl_mass_window: float = 2.8
    mass_window: float = 1.5
    mass_sideband: float = 4.5
    reco_category: str = 'l1l2'

    def mass_resolution(self, mass):
        """return the mass resolution for the input mass given
        our knowledge of the reco category that these selections are for"""
        mr_func = (
            mass_resolution.tom_2016_simps_l1l1
            if self.reco_category == 'l1l1' else
            mass_resolution.tom_2016_simps_l1l2
        )
        return mr_func(mass)


    def __call__(self, events):
        pele = np.sqrt(
                events['ele.track_.px_']**2
                +events['ele.track_.py_']**2
                +events['ele.track_.pz_']**2
        )
        ppos = np.sqrt(
                events['pos.track_.px_']**2
                +events['pos.track_.py_']**2
                +events['pos.track_.pz_']**2
        )
        psum = pele+ppos
        z = events['vertex.pos_'].fZ
    
        both_l1 = events.eleL1&events.posL1
        either_l1 = (~both_l1)&(events.eleL1|events.posL1)
        cr = (psum > self.cr_range[0])&(psum < self.cr_range[1])
        sr = (psum > self.sr_range[0])&(psum < self.sr_range[1]) 
        rc = both_l1 if self.reco_category == 'l1l1' else either_l1
        return SelectionSet(
            l1l1 = both_l1,
            l1l2 = either_l1,
            reco_category = rc,
            cr = cr,
            sr = sr,
            after_target = (z > self.target_pos),
        )


@dataclass
class TightSelections(StandardSelections):
    """Selections use for final search and exclusion"""
    y0_cut : CutByMass = None
    vprojsig: CutByMass = None
    absy: CutByMass = None
    y0err: CutByMass = None
    y0sig: CutByMass = None


    def __call__(self, events):
        """apply the selections to the events and return the necessary categories in a SelectionSet"""
        pele = np.sqrt(
                events['ele.track_.px_']**2
                +events['ele.track_.py_']**2
                +events['ele.track_.pz_']**2
        )
        ppos = np.sqrt(
                events['pos.track_.px_']**2
                +events['pos.track_.py_']**2
                +events['pos.track_.pz_']**2
        )
        psum = pele+ppos
        ele_y0 = events['ele.track_.z0_']
        ele_y0_err = np.sqrt(events['ele.track_.cov_'][:,9])
        pos_y0 = events['pos.track_.z0_']
        pos_y0_err = np.sqrt(events['pos.track_.cov_'][:,9])
        min_y0 = np.minimum(abs(ele_y0), abs(pos_y0))
        max_y0_err = np.maximum(ele_y0_err, pos_y0_err)
        min_y0_sig = np.minimum(abs(ele_y0)/ele_y0_err, abs(pos_y0)/pos_y0_err)
        vtx_proj_sig = events['vtx_proj_sig']
        z = events['vertex.pos_'].fZ
        invm = events['vertex.invM_']*1000
        absy = abs(events['vertex.pos_'].fY)
    
        both_l1 = events.eleL1&events.posL1
        either_l1 = (~both_l1)&(events.eleL1|events.posL1)
        rc = both_l1 if self.reco_category == 'l1l1' else either_l1
        cr = (psum > self.cr_range[0])&(psum < self.cr_range[1])
        sr = (psum > self.sr_range[0])&(psum < self.sr_range[1]) 

        return SelectionSet(
            reco_category = rc,
            cr = cr,
            sr = sr,
            absy = (absy > -1 if self.absy is None else absy < self.absy(invm)),
            max_y0_err = (max_y0_err > -1 if self.y0err is None else max_y0_err < self.y0err(invm)),
            vtx_proj_sig = vtx_proj_sig < self.vprojsig(invm),
            min_y0 = (min_y0 > -1 if self.y0_cut is None else min_y0 > self.y0_cut(invm)),
            min_y0_sig = (min_y0_sig > -1 if self.y0sig is None else min_y0_sig > self.y0sig(invm)),
            after_target = (z > self.target_pos),
            aliases = {
                'preselection': ['reco_category','sr'],
                'exclusion': [
                    'reco_category',
                    'sr',
                    'vtx_proj_sig',
                    'after_target',
                    'absy',
                    'max_y0_err',
                    'min_y0',
                    'min_y0_sig',
                ],
                'search': [
                    'reco_category',
                    'sr',
                    'vtx_proj_sig',
                    'absy',
                    'max_y0_err',
                    'after_target'
                ]
            }
        )
