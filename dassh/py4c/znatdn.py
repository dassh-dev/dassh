#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Adam Nelson, Milos Atz
comment: Read nuclide atomic densities
"""
########################################################################
from collections import OrderedDict
import numpy as np
from . import read_record


class ZNATDN(object):
    """The ZNATDN class reads the ZNATDN file to extract nuclide
    atomic densities for each zone (material).

    Parameters
    ----------
    fname : str (optional)
        Path to ZNATDN file; by default, looks for 'ZNATDN' file in
        working directory.

    """

    def __init__(self, fname="ZNATDN"):
        """Open the ZNATDN binary file, scrape the data into objects,
        then assign parameters based on the recovered data"""
        file = open(fname, "rb")
        data = file.read()
        file.close()

        znatdn_data = OrderedDict()

        # Skip the 0V header
        data = data[36:]

        # Get the records
        data, znatdn_data = get_1D(data, znatdn_data)
        data, znatdn_data = get_2D(data, znatdn_data)

        # Now set the parameters
        self.time = znatdn_data["1D"]["time"]
        self.cycle_num = znatdn_data["1D"]["ncy"]
        self.num_zones_subzones = znatdn_data["1D"]["ntzsz"]
        self.max_nuclides_in_set = znatdn_data["1D"]["nns"]
        self.num_blocks = znatdn_data["1D"]["nblkad"]

        # Create the space for the atom densities
        self.atom_densities = np.zeros((self.num_zones_subzones,
                                        self.max_nuclides_in_set))
        # Combine the ADEN blocks into one long array
        aden = [j for i in znatdn_data["2D"]["aden"] for j in i]
        self.atom_densities = \
            np.reshape(aden, (self.num_zones_subzones,
                              self.max_nuclides_in_set))

########################################################################


def get_1D(data, znatdn):
    """Read the 1D record (file specifications)

    Parameters
    ----------
    data : str
        ZNATDN binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ZNATDN binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    # First entry is a float
    data, time = read_record.get_array_of_float(1, data)
    # Remainder are integers
    data, words = read_record.get_array_of_int(4, data)
    data = read_record.discard_pad(data)

    znatdn["1D"] = OrderedDict()
    znatdn["1D"]["time"] = time

    keys = ["ncy", "ntzsz", "nns", "nblkad"]
    for i in range(len(keys)):
        znatdn["1D"][keys[i]] = words[i]

    return data, znatdn


def get_2D(data, znatdn):
    """Read the 2D record (nuclide zone atomic densities)

    Parameters
    ----------
    data : str
        ZNATDN binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ZNATDN binary file
    OrderedDict
        Updated object with values from 2D record

    """
    # one-dim coarse mesh and fine mesh intervals
    znatdn["2D"] = OrderedDict()

    nblkad = znatdn["1D"]["nblkad"]
    ntzsz = znatdn["1D"]["ntzsz"]
    nns = znatdn["1D"]["nns"]

    aden = []
    for m in range(nblkad):
        data = read_record.discard_pad(data)
        jl = (m) * ((ntzsz - 1) // nblkad + 1) + 1
        jup = (m + 1) * ((ntzsz - 1) // nblkad + 1)
        ju = np.min([ntzsz, jup])
        num = nns * (ju - jl + 1)
        data, block_aden = read_record.get_array_of_float(num, data)
        aden.append(block_aden)
        data = read_record.discard_pad(data)

    znatdn["2D"]["aden"] = aden

    return data, znatdn
