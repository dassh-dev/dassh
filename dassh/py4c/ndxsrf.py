#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Adam Nelson, Milos Atz
comment: Read nuclide density and cross section references
"""
########################################################################
from collections import OrderedDict
import numpy as np
from . import read_record


_ATOM_CLASSIF = {0: "undefined", 1: "fissile", 2: "fertile",
                 3: "other actinide", 4: "fission product",
                 5: "structural", 6: "coolant", 7: "control rod"}


class NDXSRF(object):
    """The NDXSRF class reads the NDXSRF file to extract problem
    nuclide densities and cross section references

    Parameters
    ----------
    fname : str (optional)
        Path to NDXSRF file; by default, looks for 'NDXSRF' file in
        working directory.

    """

    def __init__(self, fname="NDXSRF"):
        """Open the NDXSRF binary file, scrape the data into objects,
        then assign parameters based on the recovered data"""
        file = open(fname, "rb")
        data = file.read()
        file.close()

        ndxsrf_data = OrderedDict()

        # Skip the 0V header
        data = data[36:]

        # Get the records
        data, ndxsrf_data = get_1D(data, ndxsrf_data)
        data, ndxsrf_data = get_2D(data, ndxsrf_data)
        data, ndxsrf_data = get_3D(data, ndxsrf_data)

        # Now store the data
        self.num_nuclides = ndxsrf_data["1D"]["non"]
        self.num_nuclide_sets = ndxsrf_data["1D"]["nsn"]
        self.max_nuclides_in_sets = ndxsrf_data["1D"]["nns"]
        self.num_abs_nuclides = ndxsrf_data["1D"]["nan"]
        self.num_zones = ndxsrf_data["1D"]["nzone"]
        self.num_subzones = ndxsrf_data["1D"]["nsz"]
        self.nuclides = ndxsrf_data["2D"]["hnname"]
        self.abs_nuclides = ndxsrf_data["2D"]["haname"]
        self.wpf = ndxsrf_data["2D"]["wpf"]
        self.atomic_weights = ndxsrf_data["2D"]["atwt"]

        classif = ndxsrf_data["2D"]["ncln"]
        for i in range(len(classif)):
            if classif[i] > 7:
                classif[i] = "undefined"
            else:
                classif[i] = _ATOM_CLASSIF[classif[i]]
        self.nuclide_classifications = classif

        # Items 1, 2, 3 in 2nd dim are reserved and have no meaning
        # so no need to keep; therefore we take every 4th item
        self.ref_dataset = ndxsrf_data["2D"]["ndxs"][0::4]
        self.nuclide_order = \
            np.reshape(ndxsrf_data["2D"]["nos"],
                       (self.max_nuclides_in_sets,
                        self.num_nuclide_sets))
        self.nuclide_order_in_set = \
            np.reshape(ndxsrf_data["2D"]["nor"],
                       (self.num_nuclides, self.num_nuclide_sets))

        self.zone_volumes = ndxsrf_data["3D"]["volz"]
        self.zone_fracs = ndxsrf_data["3D"]["vfpa"]
        self.subzone_volumes = ndxsrf_data["3D"]["vlsa"]
        self.nuclide_set_zone_assignments = ndxsrf_data["3D"]["nspa"]
        self.nuclide_set_subzone_assignments = ndxsrf_data["3D"]["nssa"]
        self.zone_subzone_assignments = ndxsrf_data["3D"]["nzsz"]

    def calc_subzone_fracs(self):
        """Calculate subzone volume fractions"""
        subzone_fracs = np.zeros((self.num_zones, self.num_subzones))
        for i_subzone in range(self.num_subzones):
            # Now for each zone figure out how much of each subzone is
            # in this zone
            i_zone = self.zone_subzone_assignments[i_subzone] - 1
            subzone_fracs[i_zone, i_subzone] = \
                (self.subzone_volumes[i_subzone]
                 / self.zone_volumes[i_zone])
        return subzone_fracs

########################################################################


def get_1D(data, ndxsrf):
    """Read the 1D record (file specifications)

    Parameters
    ----------
    data : str
        NDXSRF binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NDXSRF binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(6, data)
    data = read_record.discard_pad(data)
    ndxsrf["1D"] = OrderedDict()
    keys = ["non", "nsn", "nns", "nan", "nzone", "nsz"]
    for i in range(len(keys)):
        ndxsrf["1D"][keys[i]] = words[i]
    return data, ndxsrf


def get_2D(data, ndxsrf):
    """Read the 2D record (nuclide referencing data)

    Parameters
    ----------
    data : str
        NDXSRF binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NDXSRF binary file
    OrderedDict
        Updated object with values from 2D record

    """
    non = ndxsrf["1D"]["non"]
    nan = ndxsrf["1D"]["nan"]
    nsn = ndxsrf["1D"]["nsn"]
    nns = ndxsrf["1D"]["nns"]
    ndxsrf["2D"] = OrderedDict()
    data = read_record.discard_pad(data)
    data, ndxsrf["2D"]["hnname"] = \
        read_record.get_array_of_string(non, 8, data)
    data, ndxsrf["2D"]["haname"] = \
        read_record.get_array_of_string(non, 8, data)
    data, ndxsrf["2D"]["wpf"] = \
        read_record.get_array_of_float(non, data)
    data, ndxsrf["2D"]["atwt"] = \
        read_record.get_array_of_float(nan, data)
    data, ndxsrf["2D"]["ncln"] = \
        read_record.get_array_of_int(non, data)
    data, ndxsrf["2D"]["ndxs"] = \
        read_record.get_array_of_int(nsn * 4, data)
    data, ndxsrf["2D"]["nos"] = \
        read_record.get_array_of_int(nsn * nns, data)
    data, ndxsrf["2D"]["nor"] = \
        read_record.get_array_of_int(nsn * non, data)
    data = read_record.discard_pad(data)
    return data, ndxsrf


def get_3D(data, ndxsrf):
    """Read the 3D record (nuclide concentration assignment data)

    Parameters
    ----------
    data : str
        NDXSRF binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated NDXSRF binary file
    OrderedDict
        Updated object with values from 3D record

    """
    nzone = ndxsrf["1D"]["nzone"]
    nsz = ndxsrf["1D"]["nsz"]
    ndxsrf["3D"] = OrderedDict()
    data = read_record.discard_pad(data)
    data, ndxsrf["3D"]["volz"] = \
        read_record.get_array_of_float(nzone, data)
    data, ndxsrf["3D"]["vfpa"] = \
        read_record.get_array_of_float(nzone, data)
    data, ndxsrf["3D"]["vlsa"] = \
        read_record.get_array_of_float(nsz, data)
    data, ndxsrf["3D"]["nspa"] = \
        read_record.get_array_of_int(nzone, data)
    data, ndxsrf["3D"]["nssa"] = \
        read_record.get_array_of_int(nsz, data)
    data, ndxsrf["3D"]["nzsz"] = \
        read_record.get_array_of_int(nsz, data)
    data = read_record.discard_pad(data)
    return data, ndxsrf
