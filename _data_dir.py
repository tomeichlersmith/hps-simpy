"""Deduce the root location of the data files depending on hostname"""

from pathlib import Path

_pass_name = 'final-simps-pass'

def get_data_dir():
    import socket
    hn = socket.getfqdn()
    if hn.endswith('slac.stanford.edu') or 'sdfiana' in hn:
        return Path('/sdf/data/hps/physics2016/')
    elif hn.endswith('zebra01.spa.umn.edu'):
        return Path('/export/scratch/users/eichl008/hps/simp-l1l2/data')
    elif hn.endswith('zuko'):
        return Path('/home/tom/code/hps/simp-l1l2/data')
    else:
        raise ValueError(f'Unrecognized host {hn}')


def get_true_vd_z_file():
    # on my personal computers that don't have the full dataset
    # but I still want to do the downstream statistical analyses
    attempt = Path(__file__).resolve().parent.parent / 'data' / 'true-vd-z-pre-readout.root'
    if attempt.is_file():
        return attempt
    
    # on S3DF, keep the file in the parent directory of the analysis tuples
    attempt = get_data_dir() / 'mc' / _pass_name / 'true-vd-z-pre-readout.root'
    if attempt.is_file():
        return attempt
    
    raise ValueError('Unable to find true-vd-z file automatically.')
