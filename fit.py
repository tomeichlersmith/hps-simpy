"""various helpers to make fitting (and plotting the results) easier"""

import numpy as np
import scipy

# fitting peaks with gaussians has been done in exclusion.fit
from .exclusion.fit import weightedmean, itermean, scaled_normal, fitnorm

def _series(x, *coefficients):
    return sum(c*x**p for p, c in enumerate(coefficients))


class series:
    def __init__(self, order):
        self._order = order
        self.coefficients = np.full(self._order+1, 1.0)

    
    def fit(self, x, y, **kwargs):
        opt, cov = scipy.optimize.curve_fit(
            _series,
            x, y, p0 = self.coefficients,
            **kwargs
        )
        self.coefficients = opt
        self.covariance = cov
        return self

    
    def __call__(self, x):
        return _series(x, *self.coefficients)

    
    def with_ceiling(self, x, ceiling):
        return np.minimum(self(x), ceiling)

    
    def with_floor(self, x, floor):
        return np.maximum(self(x), floor)

    