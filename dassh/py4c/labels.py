#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Milos Atz, Adam Nelson
comment: Read labels data from the LABELS binary file
"""
########################################################################
from collections import OrderedDict
from . import read_record


class LABELS(object):
    """The LABELS class reads the LABELS file to extract problem
    labels that identify regions, materials, zones, etc.

    Parameters
    ----------
    fname : str (optional)
        Path to GEODST file; by default, looks for 'GEODST' file in
        working directory.

    """

    def __init__(self, fname="LABELS"):
        """Open the LABELS binary file, scrape the data into objects,
        then assign parameters based on the recovered data"""
        # OPEN THE LABELS BINARY FILE ----
        file = open(fname, "rb")
        data = file.read()
        file.close()

        # SCRAPE THE DATA INTO PYTHON DATA OBJECTS ----
        # Each "get" method retrieves a record from the file and
        # appends new items to the dict container; the binary file
        # string is returned truncated.
        labels_data = OrderedDict()
        data = data[36:]  # Skip the 0V header
        data, labels_data = get_1D(data, labels_data)
        data, labels_data = get_2D(data, labels_data)
        data, labels_data = get_3D(data, labels_data)
        data, labels_data = get_4D(data, labels_data)
        data, labels_data = get_5D(data, labels_data)
        data, labels_data = get_6D(data, labels_data)
        data, labels_data = get_7D8D(data, labels_data)
        # Not yet implemented - will raise error if present
        data, labels_data = get_9D(data, labels_data)
        data, labels_data = get_10D(data, labels_data)
        data, labels_data = get_11D(data, labels_data)
        self.data = labels_data
        # SET THE PARAMETERS ----
        self.ntzsz = labels_data['1D']['ntzsz']
        self.nreg = labels_data['1D']['nreg']
        self.narea = labels_data['1D']['narea']
        self.lrega = labels_data['1D']['lrega']
        self.nhts1 = labels_data['1D']['nhts1']
        self.nhts2 = labels_data['1D']['nhts2']
        self.nsets = labels_data['1D']['nsets']
        self.nalias = labels_data['1D']['nalias']
        self.ntri = labels_data['1D']['ntri']
        self.nring = labels_data['1D']['nring']
        self.nchan = labels_data['1D']['nchan']
        self.nbanks = labels_data['1D']['nbanks']
        self.lintax = labels_data['1D']['lintax']
        self.maxtim = labels_data['1D']['maxtim']
        self.maxrod = labels_data['1D']['maxrod']
        self.maxmsh = labels_data['1D']['maxmsh']
        self.maxlrd = labels_data['1D']['maxlrd']
        self.maxlch = labels_data['1D']['maxlch']
        self.nvary = labels_data['1D']['nvary']
        self.maxbrn = labels_data['1D']['maxbrn']
        self.maxord = labels_data['1D']['maxord']
        self.nd = labels_data['1D']['nd']

        self.composition_names = labels_data['2D']['cmpnam']
        self.region_names = labels_data['2D']['regnam']
        self.area_names = labels_data['2D']['arnam']


def get_1D(data, labels_dict):
    """Read the LABELS 1D record (file specifications)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(24, data)
    data = read_record.discard_pad(data)
    keys = ["ntzsz", "nreg", "narea", "lrega", "nhts1", "nhts2",
            "nsets", "nalias", "ntri", "nring", "nchan", "nbanks",
            "lintax", "maxtim", "maxrod", "maxmsh", "maxlrd", "maxlch",
            "nvary", "maxbrn", "maxord", "nd", "idum1", "idum2"]
    labels_dict["1D"] = OrderedDict()
    for i in range(len(keys)):
        labels_dict["1D"][keys[i]] = words[i]
    return data, labels_dict


def get_2D(data, labels_dict):
    """Read the LABELS 2D record (label and area data)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 2D record

    """
    labels_dict['2D'] = OrderedDict()
    data = read_record.discard_pad(data)
    # Composition names (zones)
    ntzsz = labels_dict['1D']['ntzsz']
    data, cmp_names = read_record.get_array_of_string(ntzsz, 8, data)
    # Region names
    nreg = labels_dict['1D']['nreg']
    data, reg_names = read_record.get_array_of_string(nreg, 8, data)
    # Area names - areas are collections of regions
    narea = labels_dict['1D']['narea']
    data, area_names = read_record.get_array_of_string(narea, 8, data)
    # Assignment of regions into areas
    lrega = labels_dict['1D']['lrega']
    data, n_ra = read_record.get_array_of_int(lrega, data)
    labels_dict['2D']['cmpnam'] = cmp_names
    labels_dict['2D']['regnam'] = reg_names
    labels_dict['2D']['arnam'] = area_names
    labels_dict['2D']['nra'] = n_ra
    data = read_record.discard_pad(data)
    return data, labels_dict


def get_3D(data, labels_dict):
    """Read the LABELS 3D record (finite-geometry transverse distances)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 3D record

    """
    labels_dict['3D'] = OrderedDict()
    nhts1 = labels_dict['1D']['nhts1']
    nhts2 = labels_dict['1D']['nhts2']
    if nhts1 > 0 or nhts2 > 0:
        data = read_record.discard_pad(data)
        if nhts1 > 0:
            data, hafht1 = read_record.get_array_of_float(nhts1, data)
            data, xtrap1 = read_record.get_array_of_float(nhts1, data)
            labels_dict['3D']['hafht1'] = hafht1
            labels_dict['3D']['xtrap1'] = xtrap1
        if nhts2 > 0:
            data, hafht2 = read_record.get_array_of_float(nhts2, data)
            data, xtrap2 = read_record.get_array_of_float(nhts2, data)
            labels_dict['3D']['hafht2'] = hafht2
            labels_dict['3D']['xtrap2'] = xtrap2
        data = read_record.discard_pad(data)
    return data, labels_dict


def get_4D(data, labels_dict):
    """Read the LABELS 4D record (nuclide set labels)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 4D record

    """
    labels_dict['4D'] = OrderedDict()
    nsets = labels_dict['1D']['nsets']
    if nsets > 1:
        data = read_record.discard_pad(data)
        data, setiso = read_record.get_array_of_string(nsets, 8, data)
        labels_dict['4D']['setiso'] = setiso
        data = read_record.discard_pad(data)
    return data, labels_dict


def get_5D(data, labels_dict):
    """Read the LABELS 5D record (alias zone labels)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 5D record

    Notes
    -----
    When running REBUS-3, the zones are proliferated to the
    regions and the zone labels become identical to the region
    labels. The array alias contains the list of original zone
    labels assigned to the various regions.

    """
    labels_dict['5D'] = OrderedDict()
    nalias = labels_dict['1D']['nalias']
    if nalias > 0:
        data = read_record.discard_pad(data)
        data, alias = read_record.get_array_of_string(nalias, 8, data)
        labels_dict['5D']['alias'] = alias
        data = read_record.discard_pad(data)
    return data, labels_dict


def get_6D(data, labels_dict):
    """Read the LABELS 6D record (general control-rod model data)

    Parameters
    ----------
    data : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 6D record

    """
    labels_dict['6D'] = OrderedDict()
    nbanks = labels_dict['1D']['nbanks']
    lintax = labels_dict['1D']['lintax']
    if nbanks > 0:
        data = read_record.discard_pad(data)
        # Control-rod bank labels
        data, bnklab = read_record.get_array_of_string(nbanks, 8, data)
        labels_dict['6D']['bnklab'] = bnklab
        # Original last-dimension mesh structure (cm)
        data, zmesho = read_record.get_array_of_double(lintax, data)
        labels_dict['6D']['zmesho'] = zmesho
        # Current position of rod bank I
        data, posbnk = read_record.get_array_of_double(nbanks, data)
        labels_dict['6D']['posbnk'] = posbnk
        # Number of rods in bank I
        data, nrods = read_record.get_array_of_int(nbanks, data)
        labels_dict['6D']['nrods'] = nrods
        # Number of time nodes in position vs. time table for bank I
        data, ntimes = read_record.get_array_of_int(nbanks, data)
        labels_dict['6D']['ntimes'] = ntimes
        # Original number of fine mesh between ZMESHO(I) and ZMESHO(I+1)
        data, kfinto = read_record.get_array_of_int(lintax - 1, data)
        labels_dict['6D']['kfinto'] = kfinto
        data = read_record.discard_pad(data)
    return data, labels_dict


def get_7D8D(d, labels_dict):
    """Read the LABELS 7D (control-rod bank data) and 8D
    (control-rod channel data) records.

    Parameters
    ----------
    d : str
        LABELS binary file
    labels_dict : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated LABELS binary file
    OrderedDict
        Updated object with values from 7D record

    """
    labels_dict['7D'] = OrderedDict()
    labels_dict['8D'] = OrderedDict()
    nbanks = labels_dict['1D']['nbanks']
    nd = labels_dict['1D']['nd']
    if nbanks > 0:
        # Time entries in rod position vs. time table
        labels_dict['7D']['rbtime'] = []
        # Rod-bank positions in table (cm)
        labels_dict['7D']['rbpos'] = []
        # Number of planar mesh cells in rod k of current rod bank
        labels_dict['7D']['nmesh'] = []
        # Number of regions defined for the immovable portion
        # of rod channel k
        labels_dict['7D']['lenchn'] = []
        # Number of regions defined for the moveable portion
        # of rod channel k
        labels_dict['7D']['lenrod'] = []
        # Position (relative to the bottom of the model) of the
        # lower boundary of region L in the immoveable portion
        # of the current rod channel.
        labels_dict['8D']['poschn'] = []
        # Position (relative to rod tip) of the lower boundary
        # of region L in the moveable portion of the current
        # rod channel.
        labels_dict['8D']['posrod'] = []
        # Region assignment for regions in the immoveable
        # portion of the current rod channel, starting at the
        # bottom (Z=0.0) of the model.
        labels_dict['8D']['mrchn'] = []
        # Region assignment for regions in the moveable portion
        # of the rod, starting with the region adjacent to the
        # rod tip.
        labels_dict['8D']['mrrod'] = []
        # 1st and 2nd dimension index for planar mesh cell M
        # in the current rod channel.
        labels_dict['8D']['mesh'] = []
        for b in range(nbanks):
            # Get the 7D record for the control-rod bank
            d = read_record.discard_pad(d)
            ltime = labels_dict['6D']['ntimes'][b]
            lrods = labels_dict['6D']['nrods'][b]
            d, rbtime = read_record.get_array_of_double(ltime, d)
            d, rbpos = read_record.get_array_of_double(ltime, d)
            d, nmesh = read_record.get_array_of_int(lrods, d)
            d, lenchn = read_record.get_array_of_int(lrods, d)
            d, lenrod = read_record.get_array_of_int(lrods, d)
            labels_dict['7D']['rbtime'].append(rbtime)
            labels_dict['7D']['rbpos'].append(rbpos)
            labels_dict['7D']['nmesh'].append(nmesh)
            labels_dict['7D']['lenchn'].append(lenchn)
            labels_dict['7D']['lenrod'].append(lenrod)
            d = read_record.discard_pad(d)

            # Get the 8D records for the control-rod bank
            labels_dict['8D']['poschn'].append([])
            labels_dict['8D']['posrod'].append([])
            labels_dict['8D']['mrchn'].append([])
            labels_dict['8D']['mrrod'].append([])
            labels_dict['8D']['mesh'].append([])
            for k in range(labels_dict['6D']['nrods'][b]):
                lchn = lenchn[k]
                lrod = lenrod[k]
                mmsh = nmesh[k]
                if lchn + lrod + mmsh > 0:
                    d = read_record.discard_pad(d)
                    d, poschn = read_record.get_array_of_double(lchn, d)
                    d, posrod = read_record.get_array_of_double(lrod, d)
                    d, mrchn = read_record.get_array_of_int(lchn, d)
                    d, mrrod = read_record.get_array_of_int(lrod, d)
                    d, mesh2d = read_record.get_array_of_int(mmsh * nd,
                                                             d)
                    labels_dict['8D']['poschn'][-1].append(poschn)
                    labels_dict['8D']['posrod'][-1].append(posrod)
                    labels_dict['8D']['mrchn'][-1].append(mrchn)
                    labels_dict['8D']['mrrod'][-1].append(mrrod)
                    labels_dict['8D']['mesh'][-1].append(mesh2d)
                    d = read_record.discard_pad(d)
    return d, labels_dict


def get_9D(data, labels_dict):
    """Read the LABELS 9D record (burnup dependent cross
    section specifications)

    Not implemented

    """
    nvary = labels_dict['1D']['nvary']
    if nvary > 0:
        raise NotImplementedError('9D record reader not implemented')
    return data, labels_dict


def get_10D(data, labels_dict):
    """Read the LABELS 10D record (burnup dependent groups)

    Not implemented

    """
    maxbrn = labels_dict['1D']['maxbrn']
    if maxbrn > 0:
        raise NotImplementedError('10D record reader not implemented')
    return data, labels_dict


def get_11D(data, labels_dict):
    """Read the LABELS 11D record (burnup dependent
    fitting coefficients)

    Not implemented

    """
    maxord = labels_dict['1D']['maxord']
    if maxord > 0:
        raise NotImplementedError('11D record reader not yet implemented')
    return data, labels_dict
