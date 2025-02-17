"""list data in dask-ready way"""

from .._data_dir import get_data_dir, _pass_name

def _test_sample_list():
    data_d = get_data_dir()
    name_to_fp = [
        ('simp100', data_d / 'mc' / _pass_name / 'simps_beam' / 'analysis' / 'mass_100_hadd-simp-beam.root'),
        ('wab', data_d / 'mc' / _pass_name / 'wab_beam' / 'analysis' / 'wab-beam-hadd-100files_0.root'),
        ('rad', data_d / 'mc' / _pass_name / 'rad_beam' / 'analysis' / 'rad_beam_pass4b_tuples_100files_0.root'),
        ('tritrig', data_d / 'mc' / _pass_name / 'tritrig_beam' / 'analysis' / 'tritrig-beam-hadd-100files_0.root'),
        ('data', data_d / 'data' / _pass_name / 'physics' / 'analysis' / 'data-physrun2016-pass4kf-recon-5.2.1-run-007800-part-0.root')
    ]
    return [
        (name, {fp:'preselection'})
        for name, fp in name_to_fp
    ]


def _is_data(fp):
    return fp.stem.startswith('data-')


def _data_10pct(fp):
    return fp.stem.endswith('part-0')


def _golden_run(fp):
    return 'run-007800' in fp.stem


def _alic_10pct(fp):
    return _is_data(fp) and 'blpass4c' in fp.stem


def _full_lumi(fp):
    return _is_data(fp) and 'pass4kf' in fp.stem


_data_filters = {
    'full-lumi': _full_lumi,
    '10pct' : _data_10pct,
    'golden-run': _golden_run,
    #'alic-10pct': _alic_10pct # while historically helpful, not being carried forward
}

def _full_sample_list(data_filter = None):
    data_d = get_data_dir()
    if data_filter is None:
        data_filter = _is_data
    elif isinstance(data_filter, str):
        data_filter = _data_filters[data_filter]
    samples = {
        'data': {
            fp : 'preselection'
            for fp in (data_d / 'data' / _pass_name / 'physics' / 'analysis').iterdir()
            if fp.suffix == '.root' and data_filter(fp)
        }
    }
    samples.update({
        bkgd : {
            fp: 'preselection'
            for fp in (data_d / 'mc' / _pass_name / f'{bkgd}_beam' / 'analysis').iterdir()
            if fp.suffix == '.root'
        }
        for bkgd in ['wab','rad','tritrig']
    })
    samples.update({
        f'simp{fp.stem.split("_")[1]}': {
            fp: 'preselection'
        }
        for fp in (data_d / 'mc' / _pass_name / 'simps_beam' / 'analysis').iterdir()
        if fp.suffix == '.root'
    })
    return samples.items()


def sample_list(*, test = True, **kwargs):
    if test:
        return _test_sample_list()
    else:
        return _full_sample_list(**kwargs)
