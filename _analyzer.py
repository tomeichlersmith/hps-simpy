from pathlib import Path

from tqdm import tqdm

import uproot
import awkward as ak
import numpy as np
import hist
import hist.dask
import pickle

import dask
from dask.diagnostics import ProgressBar

from simpy.dask import SelectionSet, sample_list
from simpy.exclusion.fit import weightedmean
from simpy import exclusion_estimate, search
from simpy.exclusion.production import mass_resolution
from simpy import get_true_vd_z_file
from simpy.exclusion.production._polynomial import polynomial
from simpy.plot import mpl, plt, show, histplot, define_known_variance

# selections

# target position for 2016 data
target_pos = -4.3
# mass window in units of mass resolution sigma
excl_mass_window = 2.8


class Analyzer:
    """Analysis is done in two stages - fill and plot.
    
    The fill stage is done using dask and its child libraries (dask_histogram, dask_awkward)
    to make selections and fill histograms. Technically, you don't need to only fill histograms,
    other structures are accumlatable by dask.

    The plot stage is done after the fill stage and is just there to do things after all of the
    histograms are filled. This could include plotting but could also include other histogram-analysis
    tasks.

    Example
    -------
    The most basic example just fills histograms and the makes a plot of them.
    The default implementation of the constructor has a basic CLI that can be used to
    specify the output histogram pickle and the directory in which the plots can be stored.

        class MyAnalysis(Analyzer):
            def fill(self, events, **kwargs):
                h = hist.dask.Hist(hist.axis.Regular(100,0,1)).fill(events['variable'])
                return h
            def plot(self, h):
                h.plot()
                plt.savefig(self.outdir / 'variable_histogram.pdf')

        if __name__ == '__main__':
            MyAnalysis()
    """

    def __init__(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--mass', type=int, nargs='+', default=[30,60,90], help='mass points to plot')
        parser.add_argument('--replot', action='store_true', help='histograms already filled, just re-plotting')
        parser.add_argument('output',type=Path,help='pickle file for histograms')
        args = parser.parse_args()
    
        args.output.parent.mkdir(exist_ok=True, parents=True)

        self.masses = args.mass
        self.output = args.output
        self.outdir = args.output.parent

        if args.replot:
            with open(args.output, 'rb') as f:
                o = pickle.load(f)
        else:
            o = self._run_fill()
            with open(args.output, 'wb') as f:
                pickle.dump(o, f)

        self.plot(o)

    def selections(self, events):
        """apply the selections to the events and return the necessary categories
    
        1. Pre-Selected CR
        2. Pre-Selected SR
        3. Tight Selection SR
        """
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
        cr = (psum > 1.9)&(psum < 2.4)
        sr = (psum > 1.0)&(psum < 1.9)
        return SelectionSet(
            l1l1 = both_l1,
            l1l2 = either_l1,
            cr = cr,
            sr = sr, after_target = (z > target_pos),
        )

    def fill(self, events, *, true_z = False):
        # create and fill histograms
        raise NotImplemented

    def plot(self, h):
        # make plots
        raise NotImplemented

    def process_simp(self, events, mass):
        sl = self.selections(events)
        invm = events['vertex.invM_']*1000
        sigma_m = mass_resolution.tom_2016_simps_l1l2(mass)
        mass_window = (abs(invm-mass)/sigma_m < excl_mass_window)
        h = self.fill(events[sl('sr','l1l2')&mass_window], true_z = True)
        return h

    def process_data(self, events):
        sl = self.selections(events)
        invm = events['vertex.invM_']*1000
        o = {}
        for mass in range(20,126,2):
            sigma_m = mass_resolution.tom_2016_simps_l1l2(mass)
            mass_window = (abs(invm - mass)/sigma_m < excl_mass_window)
            o[mass] = self.fill(events[sl('sr','l1l2')&mass_window], true_z = False)
        return o

    def process(self, name, tree):
        events = uproot.dask(tree, open_files = False)
        if name.startswith('simp'):
            return self.process_simp(events, int(name[4:]))
        elif name == 'data':
            return self.process_data(events)
        else:
            # ignore simulated background by default
            return None
    
    def _run_fill(self):
        print('Constructing Task Graph')
        o = {
            name: self.process(name, tree)
            for name, tree in tqdm(sample_list(test = False, data_filter = '10pct'))
        }
    
        print('Computing Task Graph')
        with ProgressBar():
            o, = dask.compute(o)
        
        define_known_variance(o['data'])
        return o


    def show_data_signal(
        self, r, name,
        legend_kw = dict(),
        plotter = None,
        **kwargs
    ):
        if plotter is None:
            plotter = lambda h, **kwargs: h.plot(**kwargs)

        for mass, color in zip(self.masses, mpl.colors.TABLEAU_COLORS):
            plotter(
                r['data'][mass][name],
                density=True, label=f'Data ({mass} MeV)',
                color = color
            )
        for mass, color in zip(self.masses, mpl.colors.TABLEAU_COLORS):
            plotter(
                r[f'simp{mass}'][name],
                density=True, label=f'SIMP {mass} MeV',
                color = color, ls = ':'
            )
        legend_kw.setdefault('ncol', 2)
        plt.ylabel('Event Density')
        plt.xlabel(r['data'][self.masses[0]][name].axes[0].label)
        plt.yscale('log')
        yrange = plt.ylim()
        plt.ylim(ymax = 10*yrange[-1])
        plt.legend(**legend_kw)
        show(**kwargs)
