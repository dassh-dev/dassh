#!/usr/bin/env python3
########################################################################
"""
date: 2019-11-19
author: Adam Nelson, Milos Atz
comment: Read microscopic group neutron cross sections from ISOTXS
"""
########################################################################
from collections import OrderedDict
import subprocess
import os
import warnings
import numpy as np
from . import read_record


_ATOM_CLASSIF = {0: "undefined", 1: "fissile", 2: "fertile",
                 3: "other actinide", 4: "fission product",
                 5: "structural", 6: "coolant", 7: "control rod"}


class iso_data(object):
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

    def __init__(self, ngroup, data, file_chi_vector=None,
                 file_chi_matrix=None):
        self.name = data["habsid"]
        self.lib_id = data["hident"]

        self.atomic_mass = data["amass"]
        self.isotope_type = _ATOM_CLASSIF[data["kbr"]]
        if data["ichi"] == 0:
            self.chi_flag = "file-wide"
        elif data["ichi"] == 1:
            self.chi_flag = "vector"
        elif data["ichi"] > 1:
            self.chi_flag = "matrix"
        self.fissionable = bool(data["ifis"])
        self.num_total_moments = data["ltot"]
        self.num_transport_moments = data["ltrn"]
        self.num_coord_dirs = data["istrpd"]

        self.n_gamma = data["sngam"]
        self.total = data["stotpl"]
        self.transport = data["strpl"]
        self.fission = data["sfis"]
        self.nu = data["snutot"]
        if self.chi_flag == "file-wide":
            self.chi_vector = file_chi_vector
            self.chi_matrix = file_chi_matrix
        elif self.chi_flag == "vector":
            self.chi_vector = data["chiso"]
            self.chi_matrix = None
        elif self.chi_flag == "matrix":
            self.chi_vector = None
            # recast chi matrix from chiiso and isopec to just one
            # matrix
            self.chi_matrix = np.zeros((ngroup, ngroup))
            for g in range(ngroup):
                self.chi_matrix[g, :] = \
                    data["chiiso"][:, data["isopec"][g] - 1]
        self.n_alpha = data["snalf"]
        self.n_p = data["snp"]
        self.n_2n = data["sn2n"]
        self.n_d = data["snd"]
        self.n_t = data["snt"]
        self.n_p = data["snp"]
        self.coord_dir_transport = data["strpd"]

        self.total_scatter = data["total_scatter"]
        self.elastic_scatter = data["elastic_scatter"]
        self.inelastic_scatter = data["inelastic_scatter"]
        self.n2n_scatter = 0.5 * data["n2n_scatter"]
        if np.sum(self.total_scatter[:, :, 0]) > 0.:
            self.has_total_scatter = True
        else:
            self.has_total_scatter = False

    @property
    def absorption(self):
        absorption = self.n_gamma + self.fission + self.n_alpha + self.n_p + \
            self.n_d + self.n_t + self.n_p - self.n_2n
        return absorption

    @property
    def capture(self):
        capture = self.n_gamma + self.n_alpha + self.n_p + \
            self.n_d + self.n_t + self.n_p - self.n_2n
        return capture

    @property
    def nu_fission(self):
        return self.nu * self.fission

    @property
    def awr(self):
        # Convert from an atomic mass to awr
        return self.atomic_mass / 1.00866491588

    @property
    def scatter(self):
        if self.has_total_scatter:
            return self.total_scatter
        else:
            scatter = self.elastic_scatter + self.inelastic_scatter + \
                self.n2n_scatter
            # n2n already adjusted from prod'n based (as in the
            # ISOTXS file) to rxn based, as needed here.
            return scatter


class ISOTXS(object):
    """The ISOTXS class reads the ISOTXS file to extract microscopic
    group neutron cross sections

    Parameters
    ----------
    fname : str (optional)
        Path to ISOTXS file; by default, looks for 'ISOTXS' file in
        working directory.

    """

    def __init__(self, fname="ISOTXS"):
        file = open(fname, "rb")
        data = file.read()
        file.close()

        isotxs_data = OrderedDict()

        # Skip the 0V header
        data = data[36:]

        # Get the records
        data, isotxs_data = get_1D(data, isotxs_data)
        data, isotxs_data = get_2D(data, isotxs_data)
        data, isotxs_data = get_3D(data, isotxs_data)
        data, isotxs_data = get_iso_data(data, isotxs_data)

        # Store the data
        self.num_isotopes = isotxs_data["1D"]["niso"]
        self.num_groups = isotxs_data["1D"]["ngroup"]
        self.max_up = isotxs_data["1D"]["maxup"]
        self.max_down = isotxs_data["1D"]["maxdn"]
        if isotxs_data["1D"]["ichist"] == 0:
            self.file_wide_chi_flag = None
        elif isotxs_data["1D"]["ichist"] == 1:
            self.file_wide_chi_flag = "vector"
        elif isotxs_data["1D"]["ichist"] > 1:
            self.file_wide_chi_flag = "matrix"
        self.max_num_scatter_blocks = isotxs_data["1D"]["nscmax"]
        self.order = isotxs_data["1D"]["maxord"]
        if self.file_wide_chi_flag == "vector":
            self.file_wide_chi_vector = isotxs_data["2D"]["chi"]
            self.file_wide_chi_matrix = None
        elif self.file_wide_chi_flag == "matrix":
            self.file_wide_chi_vector = None
            self.file_wide_chi_matrix = isotxs_data["3D"]["chi"]
            self.isspec = isotxs_data["3D"]["isspec"]
            # recast chi matrix from chi and isspec to just one
            # matrix
            self.file_wide_chi_matrix = np.zeros((self.num_groups,
                                                  self.num_groups))
            for g in range(self.num_groups):
                self.chi_matrix[g, :] = \
                    (isotxs_data["3D"]["chi"]
                                [:, isotxs_data["3D"]["isspec"][g] - 1])
        elif self.file_wide_chi_flag is None:
            self.file_wide_chi_vector = None
            self.file_wide_chi_matrix = None
        self.energy_bounds = isotxs_data["2D"]["ebounds"]
        self.velocities = isotxs_data["2D"]["vel"]

        self.isotopes = []
        for i in range(self.num_isotopes):
            iso = iso_data(self.num_groups, isotxs_data["iso_data"][i],
                           self.file_wide_chi_vector,
                           self.file_wide_chi_matrix)
            self.isotopes.append(iso)

    def create_mcnp_mgxs(self, use_transport=False, names=None):
        """Create MCNP multigroup cross sections"""
        xs_strs = []
        if names is not None:
            iso_names = [name for name in names]
        else:
            iso_names = ["{}.99m".format(i + 1)
                         for i in range(self.num_isotopes)]
        for i in range(self.num_isotopes):

            try:
                os.remove(iso_names[i])
            except OSError:
                pass

            iso = self.isotopes[i]
            zaid = iso_names[i]
            awr = iso.awr
            cmd = "-zaid {} -awr {} -groups {}".format(zaid, awr,
                                                       self.num_groups)
            cmd += " -e"
            for g in self.energy_bounds:
                cmd += " {}".format(g * 1.e-6)

            capture = iso.capture
            if np.any(capture < 0):
                warnings.warn("Capture cross section is negative due "
                              "to fission or n,2n adjustment of "
                              "absorption; setting negatives to 0.")
                capture[capture < 0.] = 0.

            if use_transport:
                total = iso.transport[0]
            else:
                total = iso.total[0]

            if iso.fissionable:
                params = ['t', 'c', 'f', 'nu', 'chi', 's']
                xs_values = [total, capture, iso.fission, iso.nu,
                             iso.chi_vector,
                             iso.scatter[:, :, 0].flatten()]
            else:
                params = ['t', 'c', 's']
                xs_values = [total, capture,
                             iso.scatter[:, :, 0].flatten()]

            for param, vals in zip(params, xs_values):
                cmd += " -{}".format(param)
                for v in vals:
                    cmd += " {}".format(v)

            if self.order > 0 and np.any(iso.scatter[:, :, 1] != 0.):
                cmd += " -{}".format('s1')
                for v in iso.scatter[:, :, 1].flatten():
                    cmd += " {}".format(v)

            output = subprocess.check_output("simple_ace_mg.pl " + cmd,
                                             shell=True)

            # Now get the xss size from output
            substring = "xss size = "
            xss = int(str(output).split(substring)[1]
                      .split("\\", maxsplit=1)[0])

            xs_str = \
                "XS{} {} {} {} 0 1 1 {} 0 0 {}".format(i + 1, zaid, awr,
                                                       zaid, xss,
                                                       2.5301E-8)
            xs_strs.append(xs_str)

        return xs_strs


########################################################################


def get_1D(data, isotxs):
    """Read the ISOTXS 1D record (file control)

    Parameters
    ----------
    data : str
        ISOTXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ISOTXS binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(8, data)

    data = read_record.discard_pad(data)

    keys = ["ngroup", "niso", "maxup", "maxdn", "maxord", "ichist",
            "nscmax", "nsblok"]

    isotxs["1D"] = OrderedDict()
    for i in range(len(keys)):
        isotxs["1D"][keys[i]] = words[i]

    return data, isotxs


def get_2D(data, isotxs):
    """Read the ISOTXS 2D record (file data)

    Parameters
    ----------
    data : str
        ISOTXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ISOTXS binary file
    OrderedDict
        Updated object with values from 2D record

    """
    niso = isotxs["1D"]["niso"]
    ngroup = isotxs["1D"]["ngroup"]
    ichist = isotxs["1D"]["ichist"]
    data = read_record.discard_pad(data)
    data, hsetid = read_record.get_array_of_string(1, 69, data)
    data, hnames = read_record.get_terminated_strings(niso, data)
    data = read_record.discard_blanks(data)
    if ichist == 1:
        data, chi = read_record.get_array_of_float(ngroup, data)
    else:
        chi = None
    data, vel = read_record.get_array_of_float(ngroup, data)
    data, ebounds = read_record.get_array_of_float(ngroup + 1, data)
    data, loca = read_record.get_array_of_int(niso, data)
    data = read_record.discard_pad(data)

    isotxs["2D"] = OrderedDict()
    isotxs["2D"]["hsetid"] = hsetid
    isotxs["2D"]["hnames"] = hnames
    isotxs["2D"]["chi"] = np.array(chi)
    isotxs["2D"]["vel"] = np.array(vel)
    isotxs["2D"]["ebounds"] = np.array(ebounds)
    isotxs["2D"]["loca"] = np.array(loca)

    return data, isotxs


def get_3D(data, isotxs):
    """Read the ISOTXS 3D record (file-wide chi data)

    Parameters
    ----------
    data : str
        ISOTXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ISOTXS binary file
    OrderedDict
        Updated object with values from 3D record

    """
    ngroup = isotxs["1D"]["ngroup"]
    ichist = isotxs["1D"]["ichist"]

    isotxs["3D"] = OrderedDict()

    if ichist <= 1:
        return data

    data = read_record.discard_pad(data)
    data, chi = read_record.get_array_of_float(ichist * ngroup, data)
    data, isspec = read_record.get_array_of_int(ngroup)
    data = read_record.discard_pad(data)

    isotxs["3D"]["chi"] = np.array(chi).reshape((ngroup, ichist))
    isotxs["3D"]["isspec"] = np.array(isspec)

    return data, isotxs


def get_iso_data(data, isotxs):
    """Read the ISOTXS 4D (isotope control and group independent
    data), 5D (principle cross sections), 6D (isotope chi data),
    and 7D (scattering sub-block) records

    Parameters
    ----------
    data : str
        ISOTXS binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated ISOTXS binary file
    OrderedDict
        Updated object with values from 4D, 5D, 6D, and 7D records

    """
    isotxs["iso_data"] = OrderedDict()
    niso = isotxs["1D"]["niso"]
    ngroup = isotxs["1D"]["ngroup"]
    maxord = isotxs["1D"]["maxord"]
    nscmax = isotxs["1D"]["nscmax"]
    for i in range(niso):
        isotxs["iso_data"][i] = OrderedDict()
        data = read_record.discard_pad(data)
        data, strs = read_record.get_array_of_string(3, 8, data)
        isotxs["iso_data"][i]["habsid"] = strs[0]
        isotxs["iso_data"][i]["hident"] = strs[1]
        isotxs["iso_data"][i]["hmat"] = strs[2]
        data, vals = read_record.get_array_of_float(6, data)
        for key, val in zip(["amass", "efiss", "ecapt",
                             "temp", "sigpot", "adens"], vals):
            isotxs["iso_data"][i][key] = val

        data, vals = read_record.get_array_of_int(11, data)
        for key, val in zip(["kbr", "ichi", "ifis", "ialf",
                             "inp", "in2n", "ind", "int",
                             "ltot", "ltrn", "istrpd"], vals):
            isotxs["iso_data"][i][key] = val

        data, isotxs["iso_data"][i]["idsct"] = \
            read_record.get_array_of_int(nscmax, data)
        data, isotxs["iso_data"][i]["lord"] = \
            read_record.get_array_of_int(nscmax, data)
        data, jband = \
            read_record.get_array_of_int(ngroup * nscmax, data)
        jband = np.array(jband).reshape(nscmax, ngroup)
        isotxs["iso_data"][i]["jband"] = jband
        data, ijj = \
            read_record.get_array_of_int(ngroup * nscmax, data)
        ijj = np.array(ijj).reshape(nscmax, ngroup)
        isotxs["iso_data"][i]["ijj"] = ijj
        data = read_record.discard_pad(data)

        # Move on to 5D record
        ltrn = isotxs["iso_data"][i]["ltrn"]
        ltot = isotxs["iso_data"][i]["ltot"]
        istrpd = isotxs["iso_data"][i]["istrpd"]
        data = read_record.discard_pad(data)
        data, strpl = \
            read_record.get_array_of_float(ltrn * ngroup, data)
        isotxs["iso_data"][i]["strpl"] = \
            np.array(strpl).reshape(ltrn, ngroup)
        data, stotpl = \
            read_record.get_array_of_float(ltot * ngroup, data)
        isotxs["iso_data"][i]["stotpl"] = \
            np.array(stotpl).reshape(ltot, ngroup)
        keys = ["sngam", "sfis", "snutot", "chiso", "snalf",
                "snp", "sn2n", "snd", "snt"]
        ref = isotxs["iso_data"][i]
        flags = [True, ref["ifis"] > 0, ref["ifis"] > 0,
                 ref["ichi"] == 1, ref["ialf"] > 0, ref["inp"] > 0,
                 ref["in2n"] > 0, ref["ind"] > 0, ref["int"] > 0]
        for key, flag in zip(keys, flags):
            if flag:
                data, vals = \
                    read_record.get_array_of_float(ngroup, data)
                vals = np.array(vals)
            else:
                vals = np.zeros(ngroup)
            isotxs["iso_data"][i][key] = vals

        if istrpd > 0:
            data, vals = \
                read_record.get_array_of_float(istrpd * ngroup, data)
            vals = np.array(vals).reshape((istrpd, ngroup))
        else:
            vals = np.zeros((istrpd, ngroup))
        isotxs["iso_data"][i]["strpd"] = vals
        data = read_record.discard_pad(data)

        # Get the 6D record if present
        ichi = ref["ichi"]
        if ichi > 1:
            data = read_record.discard_pad(data)
            data, chiiso = \
                read_record.get_array_of_float(ichi * ngroup, data)
            data, isopec = read_record.get_array_of_int(ngroup)
            data = read_record.discard_pad(data)

            isotxs["iso_data"][i]["chiiso"] = \
                np.array(chiiso).reshape(ngroup, ichi)
            isotxs["iso_data"][i]["isopec"] = np.array(isopec)
        # Get the 7D records if present
        lord = isotxs["iso_data"][i]["lord"]
        nsblok = isotxs["1D"]["nsblok"]
        if nsblok != 1:
            raise NotImplementedError("Cannot handle nsblok "
                                      "!= 1 for scatter!")

        scatter_data = OrderedDict()
        for case in ["total_scatter", "elastic_scatter",
                     "inelastic_scatter", "n2n_scatter"]:
            scatter_data[case] = np.zeros((ngroup, ngroup, maxord + 1))

        idsct = isotxs["iso_data"][i]["idsct"]
        for n in range(nscmax):
            scatter_block = idsct[n]
            num_orders_in_block = lord[n]
            if num_orders_in_block == 0:
                continue
            # First get the scattering type
            if scatter_block < 100:
                scatter_type = "total_scatter"
            elif scatter_block < 200:
                scatter_type = "elastic_scatter"
            elif scatter_block < 300:
                scatter_type = "inelastic_scatter"
            elif scatter_block < 400:
                scatter_type = "n2n_scatter"
            starting_order = scatter_block % 100
            inscatter_w = jband[n, :]
            self_scatter_pos = ijj[n, :]
            # Now get the groupwise data
            data = read_record.discard_pad(data)
            for order in range(num_orders_in_block):
                ell = starting_order + order
                for g in range(ngroup):
                    data, vals = \
                        read_record.get_array_of_float(inscatter_w[g],
                                                       data)
                    vals = vals[::-1]
                    ss = len(vals) - self_scatter_pos[g]
                    glo = g - ss
                    ghi = glo + len(vals)
                    scatter_data[scatter_type][glo: ghi, g, ell] = \
                        vals[:]
            data = read_record.discard_pad(data)

        for k, v in scatter_data.items():
            isotxs["iso_data"][i][k] = v

    return data, isotxs
