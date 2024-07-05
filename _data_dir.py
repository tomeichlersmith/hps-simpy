"""Deduce the root location of the data files depending on hostname"""

from pathlib import Path

def get_data_dir():
    import socket
    hn = socket.getfqdn()
    if hn.endswith('slac.stanford.edu'):
        return Path('/sdf/group/hps/user-data/eichl008/simp-l1l2/analysis')
    elif hn.endswith('zebra01.spa.umn.edu'):
        return Path('/export/scratch/users/eichl008/hps/simp-l1l2/data')
    else:
        raise ValueError(f'Unrecognized host {hn}')