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


def _full_sample_list():
    data_d = get_data_dir()
    from collections import defaultdict
    samples = defaultdict(dict)
    for fp in data_d.iterdir():
        if fp.suffix != '.root':
            continue
        if fp.stem.startswith('data-'):
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


def sample_list(*, test = True):
    if test:
        return _test_sample_list()
    else:
        return _full_sample_list()