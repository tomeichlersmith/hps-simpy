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

from .dask import sample_list
from .plot import mpl, plt, show, histplot, define_known_variance
from ._selections import StandardSelections


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

    def __init__(self, *, selections = StandardSelections(), args = None):
        if args is None:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument(
                '--mass', type=int, nargs='+',
                default=[30,60,90], help='mass points to plot'
            )
            parser.add_argument(
                '--replot', action='store_true',
                help='histograms already filled, just re-plotting'
            )
            parser.add_argument(
                '--data-filter', type=str,
                help='how to choose subsample of data',
                choices=['golden-run','10pct','is-data']
            )
            parser.add_argument(
                'output',type=Path,
                help='pickle file for histograms'
            )
            args = parser.parse_args()
    
        args.output.parent.mkdir(exist_ok=True, parents=True)
        self.masses = args.mass
        self.output = args.output
        self.outdir = args.output.parent
        self.data_filter = args.data_filter
        self.selections = selections

        if args.replot:
            with open(args.output, 'rb') as f:
                o = pickle.load(f)
        else:
            o = self._run_fill()
            o['data_filter'] = self.data_filter
            o['data_frac'] = self.data_frac
            o['selections'] = self.selections
            with open(args.output, 'wb') as f:
                pickle.dump(o, f)

        self.plot(o)


    @property
    def data_frac(self):
        if self.data_filter == 'golden-run':
            return 0.016
        if self.data_filter == '10pct':
            return 0.1
        if self.data_filter == 'is-data':
            return 1.0
        return None


    def fill(self, events, *, true_z = False):
        # create and fill histograms
        raise NotImplemented

    def plot(self, h):
        # make plots
        raise NotImplemented

    def process_simp(self, events, mass):
        sl = self.selections(events)
        invm = events['vertex.invM_']*1000
        sigma_m = self.selections.mass_resolution(mass)
        mass_window = (abs(invm-mass)/sigma_m < self.selections.excl_mass_window)
        h = self.fill(events[sl('sr','l1l2')&mass_window], true_z = True)
        h['true_vd_energy'] = (
            hist.dask.Hist.new
            .Reg(230,0,2.3,label=r'True $V_D$ Energy / GeV')
            .Double()
            .fill(events[sl('sr','l1l2')]['true_vd.energy_'])
        )
        return h

    def process_data(self, events):
        sl = self.selections(events)
        invm = events['vertex.invM_']*1000
        o = {}
        o['cr'] = (
            hist.dask.Hist.new
            .Reg(220,0,220,label=r'$m_\text{reco}$ / MeV')
            .Double()
            .fill(invm[sl.cr])
        )
        for mass in range(20,126,2):
            sigma_m = self.selections.mass_resolution(mass)
            mass_window = (abs(invm - mass)/sigma_m < self.selections.excl_mass_window)
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
            for name, tree in tqdm(sample_list(test = False, data_filter = self.data_filter))
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
