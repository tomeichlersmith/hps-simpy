"""SIMPy - SIMP analysis on HPS with Python"""

from . import mfsa
from . import schema
from .._data_dir import get_data_dir

from pathlib import Path

def full_sample_list():
    data_d = get_data_dir()
    return [
        fp
        for fp in full_sample_list.data_d.iterdir()
        if fp.suffix == '.root'
    ]


def test_sample_list(sample_type = None):
    """one file from each sample type"""
    data_d = get_data_dir()
    test_samples = {
        'simp': data_d / 'mass_100_hadd-simp-beam.root',
        'wab': data_d / 'wab-beam-hadd-100files_0.root',
        'rad': data_d / 'rad_beam_pass4b_tuples_100files_0.root',
        'tritrig': data_d / 'tritrig-beam-hadd-100files_0.root',
        'data': data_d / 'data-physrun2016-pass4kf-recon-5.2.1-run-007800-part-0.root'
    }
    if sample_type is None:
        return test_samples
    return test_samples[sample_type]


def run(
    analysis,
    test = True,
    ncores = 8,
    chunk_kw = dict()
):
    return mfsa.run(
        analysis,
        test_sample_list().values() if test else full_sample_list(),
        preprocess = schema.load_event_chunks(**chunk_kw),
        ncores = ncores
    )
