#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Adam Nelson, Milos Atz
comment: Read macroscopic composition cross sections from COMPXS
"""
########################################################################
from collections import OrderedDict
import subprocess
import os
import warnings
import numpy as np
from . import read_record


class comp_data(object):
    """Cross section data for an individual isotope

    Parameters
    ----------
    ngroup : str
        Number of energy groups
    data : OrderedDict
        Ordered dict with isotope values
    file_chi_vector : numpy.ndarray (optional)
        ISOTXS-file-wide chi vector (default: None)
    file_chi_matrix : numpy.ndarray (optional)
        ISOTXS-file-wide chi array (default: None)

    """

    def __init__(self, ngroup, nkfam, data):
        self.num_families = nkfam
        self.ichi = data["ichi"]
        self.num_upscatter = np.array(data["nup"])
        self.num_downscatter = np.array(data["ndn"])
        # To be implemented, what is numfam is Type 3 card?

        # Type 4 data
        self.absorption = data["xa"]
        self.total = data["xtot"]
        self.removal = data["xrem"]
        self.transport = data["xtr"]
        self.fission = data["xf"]
        self.nu_fission = data["xnf"]
        self.chi = data["chi"]
        self.scatter = data["scatt"]
        self.power_conversion = data["pc"]
        self.diff_coeff_mult = data["a_vals"]
        self.diff_coeff_additive = data["b_vals"]
        self.nu_delayed = data["snudel"]
        self.n2n = data["xn2n"]

        # Type 5 data
        self.fission_energy = data["fpws"]
        self.capture_energy = data["cpws"]

        # Any other metadata
        if np.any(self.fission > 0.):
            self.fissionable = True
        else:
            self.fissionable = False

        if self.fissionable:
            self.nu = self.nu_fission / self.fission
        else:
            self.nu = np.ones_like(self.nu_fission)


class COMPXS(object):
    """The COMPXS class reads the COMPXS file to extract macroscopic
    composition neutron cross sections

    Parameters
    ----------
    fname : str (optional)
        Path to COMPXS file; by default, looks for 'COMPXS' file in
        working directory.

    """

    def __init__(self, fname="COMPXS"):
        file = open(fname, "rb")
        data = file.read()
        file.close()

        compxs_data = OrderedDict()

        # Get the records
        data, compxs_data = get_1D(data, compxs_data)
        data, compxs_data = get_2D(data, compxs_data)
        data, compxs_data = get_comp_data(data, compxs_data)

        # Store the data
        self.num_comps = compxs_data["1D"]["ncmp"]
        self.num_fissionable_comps = compxs_data["1D"]["nfcmp"]
        self.num_groups = compxs_data["1D"]["ngroup"]
        self.num_families = compxs_data["1D"]["nfam"]
        self.is_chi = compxs_data["1D"]["ischi"]
        self.order = compxs_data["1D"]["maxord"]

        self.energy_bounds = np.array(compxs_data["2D"]["ebounds"])
        self.velocities = np.array(compxs_data["2D"]["vel"])
        self.chi = \
            np.array(compxs_data["2D"]["chi"]).reshape(self.num_groups,
                                                       self.is_chi)
        self.file_chi_delay =\
            np.array(compxs_data["2D"]["chid"]).reshape(self.num_families,
                                                        self.num_groups)

        self.lambdas = np.array(compxs_data["2D"]["flam"])

        self.compositions = []
        for c in range(self.num_comps):
            comp = comp_data(self.num_groups, compxs_data["2D"]["nkfam"][c],
                             compxs_data["comp_data"][c])
            self.compositions.append(comp)

    def create_mcnp_mgxs(self, use_transport=False, names=None):
        """Create MCNP multigroup cross sections"""
        xs_strs = []
        if names is not None:
            comp_names = [name for name in names]
        else:
            comp_names = ["5{}.99m".format(str(c + 1).zfill(3))
                          for c in range(self.num_comps)]
        for c in range(self.num_comps):
            try:
                os.remove(comp_names[c])
            except OSError:
                pass

            comp = self.compositions[c]
            zaid = comp_names[c]
            awr = c + 1
            cmd = "-zaid {} -awr {} -groups {}".format(zaid, awr,
                                                       self.num_groups)
            cmd += " -e"
            for g in self.energy_bounds:
                cmd += " {}".format(g * 1.e-6)

            # Determine capture
            capture = comp.absorption - comp.fission - comp.n2n

            if np.any(capture < 0):
                warnings.warn("Capture cross section is negative due to " +
                              "fission or n,2n adjustment of absorption; " +
                              "settings negatives to 0.")
                capture[capture < 0.] = 0.

            if use_transport:
                total = comp.transport
            else:
                total = comp.total

            if comp.fissionable:
                params = ['t', 'c', 'f', 'nu', 'chi', 's']
                if comp.ichi == -1:
                    xs_values = [total, capture, comp.fission, comp.nu,
                                 self.chi[:, 0],
                                 comp.scatter[:, :, 0].flatten()]
                else:
                    xs_values = [total, capture, comp.fission, comp.nu,
                                 comp.chi[:, 0],
                                 comp.scatter[:, :, 0].flatten()]
            else:
                params = ['t', 'c', 's']
                xs_values = [total, capture, comp.scatter[:, :, 0].flatten()]

            for param, vals in zip(params, xs_values):
                cmd += " -{}".format(param)
                for v in vals:
                    cmd += " {}".format(v)

            if self.order > 0 and np.any(comp.scatter[:, :, 1] != 0.):
                cmd += " -{}".format('s1')
                for v in comp.scatter[:, :, 1].flatten():
                    cmd += " {}".format(v)

            output = subprocess.check_output("simple_ace_mg.pl " + cmd,
                                             shell=True)

            # Now get the xss size from output
            substring = "xss size = "
            xss = int(str(output).split(substring)[1].split("\\",
                                                            maxsplit=1)[0])

            xs_str = \
                "XS{} {} {} {} 0 1 1 {} 0 0 {}".format(c + 1, zaid, awr, zaid,
                                                       xss, 2.5301E-8)
            xs_strs.append(xs_str)

        return xs_strs


########################################################################

def get_1D(data, compxs):
    """Read the ISOTXS 1D record (file specifications)

    Parameters
    ----------
    data : str
        COMPXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated COMPXS binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(10, data)

    data = read_record.discard_pad(data)

    keys = ["ncmp", "ngroup", "ischi", "nfcmp", "maxup", "maxdn", "nfam",
            "maxord", "ndum2", "ndum3"]

    compxs["1D"] = OrderedDict()
    for i in range(len(keys)):
        compxs["1D"][keys[i]] = words[i]

    return data, compxs


def get_2D(data, compxs):
    """Read the COMPXS 2D record (composition independent data)

    Parameters
    ----------
    data : str
        COMPXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated COMPXS binary file
    OrderedDict
        Updated object with values from 2D record

    """
    # one-dim coarse mesh and fine mesh intervals
    compxs["2D"] = OrderedDict()
    ischi = compxs["1D"]["ischi"]
    ngroup = compxs["1D"]["ngroup"]
    nfam = compxs["1D"]["nfam"]
    ncmp = compxs["1D"]["ncmp"]

    data = read_record.discard_pad(data)
    data, chi = read_record.get_array_of_double(ischi * ngroup, data)
    data, vel = read_record.get_array_of_double(ngroup, data)
    data, ebounds = read_record.get_array_of_double(ngroup + 1, data)
    data, chid = read_record.get_array_of_double(nfam * ngroup, data)
    data, flam = read_record.get_array_of_double(nfam, data)
    data, nkfam = read_record.get_array_of_int(ncmp, data)
    data = read_record.discard_pad(data)

    compxs["2D"]["chi"] = chi
    compxs["2D"]["vel"] = vel
    compxs["2D"]["ebounds"] = ebounds
    compxs["2D"]["chid"] = chid
    compxs["2D"]["flam"] = flam
    compxs["2D"]["nkfam"] = nkfam

    return data, compxs


def get_comp_data(data, compxs):
    """Read the remaining COMPXS records: 3D (composition
    specifications); 4D (composotion macroscopic group cross
    sections; 5D (power conversion factors)

    Parameters
    ----------
    data : str
        COMPXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated COMPXS binary file
    OrderedDict
        Updated object with values from 3D-5D records

    """
    compxs["comp_data"] = OrderedDict()
    ncmp = compxs["1D"]["ncmp"]
    ngroup = compxs["1D"]["ngroup"]
    maxord = compxs["1D"]["maxord"]
    for c in range(ncmp):
        compxs["comp_data"][c] = OrderedDict()
        nkfami = compxs["2D"]["nkfam"][c]
        data = read_record.discard_pad(data)
        data, vals = read_record.get_array_of_int(1, data)
        ichi = vals[0]
        compxs["comp_data"][c]["ichi"] = ichi
        data, compxs["comp_data"][c]["nup"] = \
            read_record.get_array_of_int(ngroup, data)
        data, compxs["comp_data"][c]["ndn"] = \
            read_record.get_array_of_int(ngroup, data)
        data, compxs["comp_data"][c]["numfam"] = \
            read_record.get_array_of_int(nkfami, data)
        data = read_record.discard_pad(data)

        xa = np.zeros(ngroup)
        xtot = np.zeros(ngroup)
        xrem = np.zeros(ngroup)
        xtr = np.zeros(ngroup)
        xf = np.zeros(ngroup)
        xnf = np.zeros(ngroup)
        if ichi > 0:
            ichi_val = ichi
        else:
            ichi_val = 1
        chi = np.zeros((ngroup, ichi_val))
        scatt = np.zeros((ngroup, ngroup, maxord + 1))  # Ordered in/out
        pc = np.zeros(ngroup)
        a_vals = np.zeros((ngroup, 3))
        b_vals = np.zeros((ngroup, 3))
        snudel = np.zeros((ngroup, nkfami))
        xn2n = np.zeros(ngroup)
        for g in range(ngroup):
            data = read_record.discard_pad(data)
            data, the_set = read_record.get_array_of_double(4, data)
            xa[g], xtot[g], xrem[g], xtr[g] = the_set
            if ichi != 0:
                data, the_set = read_record.get_array_of_double(2, data)
                xf[g], xnf[g] = the_set
                if ichi > 0:
                    data, chi[g, :] = \
                        read_record.get_array_of_double(ichi, data)

            # Get P0 scatter data
            numup = compxs["comp_data"][c]["nup"][g]
            numdn = compxs["comp_data"][c]["ndn"][g]
            data, upscatt = read_record.get_array_of_double(numup, data)
            data, selfscatt = read_record.get_array_of_double(1, data)
            data, downscatt = \
                read_record.get_array_of_double(numdn, data)
            scatt[g - numdn: g + numup + 1, g, 0] = \
                downscatt[::-1] + selfscatt + upscatt[::-1]

            data, the_set = read_record.get_array_of_double(7, data)
            pc[g] = the_set[0]
            a_vals[g, :] = [the_set[i] for i in [1, 3, 5]]
            b_vals[g, :] = [the_set[i] for i in [2, 4, 6]]

            data, snudel[g, :] = \
                read_record.get_array_of_double(nkfami, data)

            data, xn2n_val = read_record.get_array_of_double(1, data)
            xn2n[g] = xn2n_val[0]

            # Now get the remainder of the scattering data
            for l in range(maxord):
                data, upscatt = \
                    read_record.get_array_of_double(numup, data)
                data, selfscatt = \
                    read_record.get_array_of_double(1, data)
                data, downscatt = \
                    read_record.get_array_of_double(numdn, data)
                scatt[g - numdn: g + numup + 1, g, l + 1] = \
                    downscatt[::-1] + selfscatt + upscatt[::-1]
            data = read_record.discard_pad(data)

        # Now store the composition data
        compxs["comp_data"][c]["xa"] = xa
        compxs["comp_data"][c]["xtot"] = xtot
        compxs["comp_data"][c]["xrem"] = xrem
        compxs["comp_data"][c]["xtr"] = xtr
        compxs["comp_data"][c]["xf"] = xf
        compxs["comp_data"][c]["xnf"] = xnf
        compxs["comp_data"][c]["chi"] = chi
        compxs["comp_data"][c]["scatt"] = scatt
        compxs["comp_data"][c]["pc"] = pc
        compxs["comp_data"][c]["a_vals"] = a_vals
        compxs["comp_data"][c]["b_vals"] = b_vals
        compxs["comp_data"][c]["snudel"] = snudel
        compxs["comp_data"][c]["xn2n"] = xn2n

    # Now we move on to the type 5 data, power conversion factors
    data = read_record.discard_pad(data)
    data, fpws = read_record.get_array_of_double(ncmp, data)
    data, cpws = read_record.get_array_of_double(ncmp, data)
    data = read_record.discard_pad(data)
    for c in range(ncmp):
        compxs["comp_data"][c]["fpws"] = fpws[c]
        compxs["comp_data"][c]["cpws"] = cpws[c]

    return data, compxs
