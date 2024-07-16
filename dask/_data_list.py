"""list data in dask-ready way"""

from .._data_dir import get_data_dir

def _test_sample_list():
    data_d = get_data_dir()
    name_to_fp = [
        ('simp100', data_d / 'mass_100_hadd-simp-beam.root'),
        ('wab', data_d / 'wab-beam-hadd-100files_0.root'),
        ('rad', data_d / 'rad_beam_pass4b_tuples_100files_0.root'),
        ('tritrig', data_d / 'tritrig-beam-hadd-100files_0.root'),
        ('data', data_d / 'data-physrun2016-pass4kf-recon-5.2.1-run-007800-part-0.root')
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

_data_filters = {
    'is-data' : _is_data,
    '10pct' : _data_10pct,
    'golden-run': _golden_run,
    'alic-10pct': _alic_10pct
}

def _full_sample_list(data_filter = None):
    data_d = get_data_dir()
    if data_filter is None:
        data_filter = _is_data
    elif isinstance(data_filter, str):
        data_filter = _data_filters[data_filter]
    from collections import defaultdict
    samples = defaultdict(dict)
    for fp in data_d.iterdir():
        if fp.suffix != '.root' or fp.stem == 'true-vd-z-pre-readout':
            continue
        if _is_data(fp):
            if data_filter(fp):
                samples['data'][fp] = 'preselection'
        elif fp.stem.endswith('simp-beam'):
            m = fp.stem.split('_')[1]
            samples[f'simp{m}'][fp] = 'preselection'
        elif fp.stem.startswith('wab'):
            samples['wab'][fp] = 'preselection'
        elif fp.stem.startswith('rad'):
            samples['rad'][fp] = 'preselection'
        elif fp.stem.startswith('tritrig'):
            samples['tritrig'][fp] = 'preselection'
        else:
            raise ValueError(fp)
    return samples.items()


def sample_list(*, test = True, **kwargs):
    if test:
        return _test_sample_list()
    else:
        return _full_sample_list(**kwargs)
