"""luminosity data"""

from types import SimpleNamespace

data = SimpleNamespace(
    tritrig = SimpleNamespace(
        nfiles = 10_000,
        nevents_per_file = 50_000,
        xsec = 1.416e9
    ),
    wab = SimpleNamespace(
        nfiles = 10_000,
        nevents_per_file = 100000,
        xsec = 0.1985e12
    ),
    rad = SimpleNamespace(
        nfiles = 10_000,
        nevents_per_file = 10000,
        xsec = 66.36e6
    ),
    total = 10.7
)

def scale(bkgd):
    """Calculate the scaling factor for the input MC background name"""
    d = getattr(data, bkgd)
    return d.xsec/(d.nevents_per_file*d.nfiles)


def mc_bkgd_sum(hist_by_bkgd):
    """Sum over the two MC background samples with their appropriate scaling factors

    The argument is a function that returns the histogram corresponding to the
    input background name. For example, if the histograms are stored in a dictionary d.

        mc_bkgd_sum(lambda b: d[b])

    Or if they are stored as StrCategory in a hist.Hist `h`.

        mc_bkgd_sum(lambda b: h[b,...])
    """
    
    return sum(
        hist_by_bkgd(bkgd)*scale(bkgd)
        for bkgd in ['tritrig', 'wab']
    )