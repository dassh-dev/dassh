#!/usr/bin/env python3
########################################################################
"""
date: 2019-10-19
author: Milos Atz
comment: Read NHFLUX binary file
"""
########################################################################
from collections import OrderedDict
import numpy as np
from . import read_record


class NHFLUX(object):
    """The NHFLUX class reads the NHFLUX file to extract flux data

    Parameters
    ----------
    fname : str (optional)
        Path to NHFLUX file; by default, looks for 'NHFLUX' file in
        working directory.

    """

    def __init__(self, fname="NHFLUX", old3D=False):
        file = open(fname, "rb")
        data = file.read()
        file.close()

        nhflux_data = OrderedDict()

        # Skip the 0V header
        data = data[36:]

        # # Get the records
        data, nhflux_data = get_1D(data, nhflux_data)
        data, nhflux_data = get_2D(data, nhflux_data)
        if old3D:
            data, nhflux_data = get_3D_old(data, nhflux_data)
        else:
            data, nhflux_data = get_3D(data, nhflux_data)
        # data, nhflux_data = get_4D(data, nhflux_data)
        # data, nhflux_data = get_5D(data, nhflux_data)

        # Store results
        self.data = nhflux_data
        self.dimensions = nhflux_data['1D']['ndim']
        self.n_group = nhflux_data['1D']['ngroup']
        self.keff = nhflux_data['1D']['effk']
        self.power = nhflux_data['1D']['power']
        self.n_moments = nhflux_data['1D']['nmom']
        self.n_int_xy = nhflux_data['1D']['nintxy']
        # Other 1D keyword options:
        # 'ninti', 'nintj', 'nintk', 'iter', 'nsurf', 'npcxy', 'nscoef'
        # 'itrord', 'iaprx', 'ileak', 'iaprxz', 'ileakz', 'iorder'
        # 'npcbdy', 'npcsym', 'npcsec', 'iwnhfl'

        self.itrmap = nhflux_data['2D']['itrmap']
        # Other 2D keyword o ptions
        # 'ipcpnt', 'ipcbdy', 'ipcsym', 'ipcsto'

        self.flux = nhflux_data['3D']['flux']


########################################################################


def get_1D(data, nhflux):
    """Read the NHFLUX 1D record (file specifications)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 1D record

    """
    nhflux["1D"] = OrderedDict()
    data = read_record.discard_pad(data)
    # Some float in the middle, split into steps: 6 int
    data, words = read_record.get_array_of_int(6, data)
    keys = ["ndim", "ngroup", "ninti", "nintj", "nintk", "iter"]
    for i in range(len(keys)):
        nhflux["1D"][keys[i]] = words[i]
    # 2 float (keff and power)
    data, words = read_record.get_array_of_float(2, data)
    keys = ["effk", "power"]
    for i in range(len(keys)):
        nhflux["1D"][keys[i]] = words[i]
    # 22 int
    data, words = read_record.get_array_of_int(22, data)
    keys = ["nsurf", "nmom", "nintxy", "npcxy", "nscoef", "itrord",
            "iaprx", "ileak", "iaprxz", "ileakz", "iorder", "npcbdy",
            "npcsym", "npcsec", "iwnhfl", "idum1", "idum2", "idum3",
            "idum4", "idum5", "idum6", "idum7"]
    for i in range(len(keys)):
        nhflux["1D"][keys[i]] = words[i]
    data = read_record.discard_pad(data)
    return data, nhflux


def get_2D(data, nhflux):
    """Read the NHFLUX 2D record (integer pointers)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 2D record

    """
    nhflux['2D'] = OrderedDict()
    nsurf = nhflux['1D']['nsurf']
    nintxy = nhflux['1D']['nintxy']
    if nsurf > 1:
        data = read_record.discard_pad(data)
        # Pointers to incoming XY-plane partial currents
        nhflux['2D']['ipcpnt'] = []
        for i in range(nintxy):
            data, ipcpnt = read_record.get_array_of_int(nsurf, data)
            nhflux['2D']['ipcpnt'].append(ipcpnt)
        nhflux['2D']['ipcpnt'] = np.vstack(nhflux['2D']['ipcpnt'])
        # Pointers to outgoing partial currents on outer
        # XY-plane boundary
        npcbdy = nhflux['1D']['npcbdy']
        data, ipcbdy = read_record.get_array_of_int(npcbdy, data)
        nhflux['2D']['ipcbdy'] = ipcbdy
        # Transformation map between nodal and geodst
        # mesh cell orderings
        data, itrmap = read_record.get_array_of_int(nintxy, data)
        nhflux['2D']['itrmap'] = itrmap
        # Pointers to outgoing partial currents on symmetric and
        # sector XY-plane boundary (hexagonal geometry only)
        npcsto = nhflux['1D']['npcsym'] + nhflux['1D']['npcsec']
        data, ipcsym = read_record.get_array_of_int(npcsto, data)
        nhflux['2D']['ipcsym'] = ipcsym
        # Pointers to ingoing partial currents on symmetric and
        # sector XY-plane boundary (hexagonal geometry only)
        data, ipcsto = read_record.get_array_of_int(npcsto, data)
        nhflux['2D']['ipcsto'] = ipcsto
        data = read_record.discard_pad(data)
    return data, nhflux


def get_3D_old(data, nhflux):
    """Read the NHFLUX 3D record (regular flux moments)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 3D record

    """
    nhflux['3D'] = OrderedDict()
    nhflux['3D']['flux'] = []
    nintk = nhflux['1D']['nintk']
    nintxy = nhflux['1D']['nintxy']
    nmom = nhflux['1D']['nmom']
    ngroup = nhflux['1D']['ngroup']
    for grp in range(ngroup):
        flux_in_group = []
        for k in range(nintk):
            data = read_record.discard_pad(data)
            flux_at_axial_level = []
            for j in range(nintxy):
                data, flux = read_record.get_array_of_double(nmom, data)
                flux_at_axial_level.append(flux)
            flux_in_group.append(np.vstack(flux_at_axial_level))
            data = read_record.discard_pad(data)
        nhflux['3D']['flux'].append(np.stack(flux_in_group))
    return data, nhflux


def get_3D(data, nhflux):
    """Read the NHFLUX 3D record (regular flux moments)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 3D record

    """
    nhflux['3D'] = OrderedDict()
    nhflux['3D']['flux'] = []
    nintk = nhflux['1D']['nintk']
    nintxy = nhflux['1D']['nintxy']
    nmom = nhflux['1D']['nmom']
    ngroup = nhflux['1D']['ngroup']
    pos = nintxy * nmom * 8
    for grp in range(ngroup):
        flux_in_group = []
        for k in range(nintk):
            flux_in_group.append(
                np.ndarray((nintxy, nmom),
                           dtype='d',
                           buffer=data[:(pos + 4)],
                           offset=4))
            data = data[(pos + 8):]
        nhflux['3D']['flux'].append(np.stack(flux_in_group))
    return data, nhflux


def get_4D(data, nhflux):
    """Read the NHFLUX 4D record (regular XY-directed partial currents)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 4D record

    """
    raise NotImplementedError('yolo')
    return data, nhflux


def get_5D(data, nhflux):
    """Read the NHFLUX 5D record (regular Z-directed partial currents)

    Parameters
    ----------
    data : str
        NHFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NHFLUX binary file
    OrderedDict
        Updated object with values from 5D record

    """
    raise NotImplementedError('yolo')
    return data, nhflux
