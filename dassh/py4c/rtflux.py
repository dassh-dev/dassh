#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Adam Nelson, Milos Atz
comment: Read RTFLUX binary file
"""
########################################################################
from collections import OrderedDict
import warnings
import numpy as np
from . import read_record


class RTFLUX(object):
    """The RTFLUX class reads the RTFLUX file to extract flux data

    Parameters
    ----------
    fname : str (optional)
        Path to RTFLUX file; by default, looks for 'RTFLUX' file in
        working directory.

    """

    def __init__(self, fname="RTFLUX"):
        file = open(fname, "rb")
        data = file.read()
        file.close()

        rtflux_data = OrderedDict()

        # Skip the 0V header
        data = data[36:]

        # Get the records
        data, rtflux_data = get_specs(data, rtflux_data)
        data, rtflux_data = get_flux(data, rtflux_data)

        # Store results
        self.num_dimensions = rtflux_data["ndim"]
        self.num_iterations = rtflux_data["iter"]
        self.num_blocks = rtflux_data["nblok"]
        self.keff = rtflux_data["keff"]
        self.power = rtflux_data["power"]
        self.flux = rtflux_data["flux"]

        # Warn the user if any fluxes are negative
        if np.any(self.flux < 0):
            warnings.warn("{} contains negative fluxes! ".format(fname) +
                          "Verify convergence!")

    @property
    def shape(self):
        return self.flux.shape

    @property
    def num_groups(self):
        return self.flux.shape[0]

    @property
    def num_k(self):
        return self.flux.shape[1]

    @property
    def num_j(self):
        return self.flux.shape[2]

    @property
    def num_i(self):
        return self.flux.shape[3]

########################################################################


def get_specs(data, rtflux):
    """Read the RTFLUX 1D record (file specifications)

    Parameters
    ----------
    data : str
        RTFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated RTFLUX binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(6, data)
    data, vals = read_record.get_array_of_float(2, data)
    effk, power = vals[:]
    data, vals = read_record.get_array_of_int(1, data)
    nblok = vals[0]
    data = read_record.discard_pad(data)

    keys = ["ndim", "ngroup", "ninti", "nintj", "nintk", "iter"]

    for i in range(len(keys)):
        rtflux[keys[i]] = words[i]
    rtflux["keff"] = effk
    rtflux["power"] = power
    rtflux["nblok"] = nblok

    return data, rtflux


def get_flux(data, rtflux):
    """Read the RTFLUX 2D (one-dimensional regular total flux)
    or 3D (multi-dimensional regular total flux) records

    Parameters
    ----------
    data : str
        RTFLUX binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated RTFLUX binary file
    OrderedDict
        Updated object with values from 2D or 3D record

    """
    ndim = rtflux["ndim"]
    ninti = rtflux["ninti"]
    nintj = rtflux["nintj"]
    nintk = rtflux["nintk"]
    ngroup = rtflux["ngroup"]
    nblok = rtflux["nblok"]

    flux = np.zeros((ngroup, nintk, nintj, ninti))

    if ndim == 1:
        flux_block = []
        for m in range(nblok):
            data = read_record.discard_pad(data)
            jl = m * ((ngroup - 1) // nblok + 1) + 1
            jup = (m + 1) * ((ngroup - 1) // nblok + 1)
            ju = np.min([ngroup, jup])
            data, vals = \
                read_record.get_array_of_double(ninti * (ju - jl + 1),
                                                data)
            flux_block.extend(vals)
            data = read_record.discard_pad(data)

        # Now store in flux
        flux[:, 0, 0, :] = np.array(flux_block).reshape(ngroup, ninti)

    else:
        for l in range(ngroup):
            for k in range(nintk):
                flux_block = []
                for m in range(nblok):
                    data = read_record.discard_pad(data)
                    jl = m * ((nintj - 1) // nblok + 1) + 1
                    jup = (m + 1) * ((nintj - 1) // nblok + 1)
                    ju = np.min([nintj, jup])
                    nvals = ninti * (ju - jl + 1)
                    data, vals = \
                        read_record.get_array_of_double(nvals, data)
                    flux_block.extend(vals)
                    data = read_record.discard_pad(data)
                # Now store in flux
                flux[l, k, :, :] = \
                    np.array(flux_block).reshape(nintj, ninti)

    rtflux["flux"] = flux
    return data, rtflux
