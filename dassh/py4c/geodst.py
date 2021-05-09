#!/usr/bin/env python3
########################################################################
"""
date: 2020-02-12
author: Adam Nelson, Milos Atz
comment: Read geometry data from the GEODST binary file
"""
########################################################################
from collections import OrderedDict
import numpy as np
from . import read_record


# Geometry types available to be specified in the GEODST file
# For more information, see the GEODST file description
_GEOM_TYPES = {0: "point", 1: "slab", 2: "cylinder", 3: "sphere",
               6: "x-y", 7: "r-z", 8: "theta-r", 9: "tri",
               10: "hex", 11: "r-theta", 12: "r-theta-z",
               13: "r-theta-alpha", 14: "x-y-z", 15: "theta-r-z",
               16: "theta-r-alpha", 17: "tri-z", 18: "hex-z"}


# Boundary conditions available in the GEODST file
# For more information, see the GEODST file description
_BC_ZERO_FLUX = 0
_BC_REFLECTED = 1
_BC_EXTRAP = 2
_BC_PERIODIC_OPP = 3
_BC_PERIODIC_ADJ = 4
_BC_INVERTED = 5
_BC_ISOTROPIC = 6


# Options for treatment of triangular mesh units
# For more information, see the GEODST file description
_TRIAG_TYPES = {0: "rhomb-120", 1: "rhomb-60", 2: "rect",
                3: "equil-tri-60", 4: "right-tri", 5: "rhomb-30"}


class GEODST(object):
    """The GEODST class reads the GEODST file to extract problem
    geometry data

    Parameters
    ----------
    fname : str (optional)
        Path to GEODST file; by default, looks for 'GEODST' file in
        working directory.

    """

    def __init__(self, fname="GEODST"):
        """Open the GEODST binary file, scrape the data into objects,
        then assign parameters based on the recovered data"""
        # OPEN THE GEODST BINARY FILE ----
        file = open(fname, "rb")
        data = file.read()
        file.close()

        # SCRAPE THE DATA INTO PYTHON DATA OBJECTS ----
        # Each "get" method retrieves a record from the file and
        # appends new items to the dict container; the binary file
        # string is returned truncated.
        geodst_data = OrderedDict()
        data = data[36:]  # Skip the 0V header
        data, geodst_data = get_1D(data, geodst_data)
        data, geodst_data = get_2D(data, geodst_data)
        data, geodst_data = get_3D(data, geodst_data)
        data, geodst_data = get_4D(data, geodst_data)
        data, geodst_data = get_5D(data, geodst_data)
        data, geodst_data = get_6D(data, geodst_data)
        data, geodst_data = get_7D(data, geodst_data)

        # SET THE PARAMETERS ----
        igom = geodst_data["1D"]["igom"]
        if igom in _GEOM_TYPES.keys():
            self.geom_type = _GEOM_TYPES[igom]
        else:
            raise ValueError("Invalid value of IGOM in GEODST")
        if igom == 0:
            self.dimensions = 0
        elif igom < 6:
            self.dimensions = 1
        elif igom < 12:
            self.dimensions = 2
        else:
            self.dimensions = 3

        self.nzone = geodst_data["1D"]["nzone"]
        self.nreg = geodst_data["1D"]["nreg"]
        self.nzcl = geodst_data["1D"]["nzcl"]
        self.coarse_dims = [geodst_data["1D"]["ncintk"],
                            geodst_data["1D"]["ncintj"],
                            geodst_data["1D"]["ncinti"]]
        self.fine_dims = [geodst_data["1D"]["nintk"],
                          geodst_data["1D"]["nintj"],
                          geodst_data["1D"]["ninti"]]
        self.bcs = np.array([[geodst_data["1D"]["kmb1"],
                              geodst_data["1D"]["kmb2"]],
                             [geodst_data["1D"]["jmb1"],
                              geodst_data["1D"]["jmb2"]],
                             [geodst_data["1D"]["imb1"],
                              geodst_data["1D"]["imb2"]]])
        self.nbs = geodst_data["1D"]["nbs"]
        self.nbcs = geodst_data["1D"]["nbcs"]
        self.nibcs = geodst_data["1D"]["nibcs"]
        self.nzwbb = geodst_data["1D"]["nzwbb"]
        ntriag = geodst_data["1D"]["ntriag"]
        nthpt = geodst_data["1D"]["nthpt"]

        if igom in [9, 17, 10, 18]:
            self.triangle_option = _TRIAG_TYPES[ntriag]
            self.triangle_orientation = nthpt
        else:
            self.triangle_option = None
            self.triangle_orientation = None

        if geodst_data["1D"]["nrass"] == 0:
            self.assign_to_coarse = True
        else:
            self.assign_to_coarse = False
        if geodst_data["2D"]:  # then this is a 1D geometry
            self.xmesh = geodst_data["2D"]["xmesh"]
            self.ymesh = []
            self.zmesh = []
            self.ifints = geodst_data["2D"]["ifints"]
            self.jfints = []
            self.kfints = []
        elif geodst_data["3D"]:  # then this is a 2D geometry
            self.xmesh = geodst_data["3D"]["xmesh"]
            self.ymesh = geodst_data["3D"]["ymesh"]
            self.zmesh = []
            self.ifints = geodst_data["3D"]["ifints"]
            self.jfints = geodst_data["3D"]["jfints"]
            self.kfints = []
        elif geodst_data["4D"]:  # then this is a 3D geometry
            self.xmesh = geodst_data["4D"]["xmesh"]
            self.ymesh = geodst_data["4D"]["ymesh"]
            self.zmesh = geodst_data["4D"]["zmesh"]
            self.ifints = geodst_data["4D"]["ifints"]
            self.jfints = geodst_data["4D"]["jfints"]
            self.kfints = geodst_data["4D"]["kfints"]
        else:
            raise ValueError("0D geometry not supported")

        # Set the geometry data
        self.volr = geodst_data["5D"]["volr"]
        self.bsq = geodst_data["5D"]["bsq"]
        self.bndc = geodst_data["5D"]["bndc"]
        self.bnci = geodst_data["5D"]["bnci"]
        self.nzhbb = geodst_data["5D"]["nzhbb"]
        self.nzc = geodst_data["5D"]["nzc"]
        self.nznr = geodst_data["5D"]["nznr"]

        # Set the mesh-wise region assignments
        if self.dimensions > 0 and self.assign_to_coarse:
            self.reg_assignments = np.zeros(self.coarse_dims, dtype=int)
            mr = np.array(geodst_data["6D"]["mr"])
            for c_k in range(self.coarse_dims[0]):
                ij = 0
                for c_j in range(self.coarse_dims[1]):
                    for c_i in range(self.coarse_dims[2]):
                        self.reg_assignments[c_k, c_j, c_i] = (mr[c_k]
                                                                 [ij])
                        ij += 1
        elif self.dimensions > 0 and not self.assign_to_coarse:
            self.reg_assignments = np.zeros(self.fine_dims, dtype=int)
            mr = np.array(geodst_data["6D"]["mr"])
            for f_k in range(self.fine_dims[0]):
                ij = 0
                for f_j in range(self.fine_dims[1]):
                    for f_i in range(self.fine_dims[2]):
                        self.reg_assignments[f_k, f_j, f_i] = (mr[f_k]
                                                                 [ij])
                        ij += 1
        else:
            raise ValueError("Point geometry not supported")

    def calc_zs(self):
        """Calculate the z-dimension mesh points"""
        if self.dimensions < 3:  # not 3-dimensional, no z-values
            return np.array([0.])

        z_coarse_lo = self.zmesh[0]
        zs = []
        for coarse_k in range(self.coarse_dims[0]):
            z_coarse_hi = self.zmesh[coarse_k + 1]
            dz = ((z_coarse_hi - z_coarse_lo)
                  / float(self.kfints[coarse_k]))
            for k in range(self.kfints[coarse_k]):
                zs.append(z_coarse_lo + dz * float(k))
            z_coarse_lo = z_coarse_hi
        zs.append(self.zmesh[-1])
        return np.array(zs)

    def calc_ys(self):
        """Calculate the y-dimension mesh points"""
        if self.geom_type != "x-y-z":
            msg = "Only implemented for x-y-z geometry"
            raise NotImplementedError(msg)
        if self.dimensions < 2:
            return np.array([0.])

        y_coarse_lo = self.ymesh[0]
        ys = []
        for coarse_j in range(self.coarse_dims[1]):
            y_coarse_hi = self.ymesh[coarse_j + 1]
            dy = ((y_coarse_hi - y_coarse_lo)
                  / float(self.jfints[coarse_j]))
            for j in range(self.jfints[coarse_j]):
                ys.append(y_coarse_lo + dy * float(j))
            y_coarse_lo = y_coarse_hi
        ys.append(self.ymesh[-1])
        return np.array(ys)

    def calc_xs(self):
        """Calculate the x-dimension mesh points"""
        if self.geom_type != "x-y-z":
            msg = "Only implemented for x-y-z geometry"
            raise NotImplementedError(msg)
        if self.dimensions < 1:
            return np.array([0.])

        x_coarse_lo = self.xmesh[0]
        xs = []
        for coarse_i in range(self.coarse_dims[2]):
            x_coarse_hi = self.xmesh[coarse_i + 1]
            dx = ((x_coarse_hi - x_coarse_lo)
                  / float(self.ifints[coarse_i]))
            for i in range(self.ifints[coarse_i]):
                xs.append(x_coarse_lo + dx * float(i))
            x_coarse_lo = x_coarse_hi
        xs.append(self.xmesh[-1])
        return np.array(xs)

    def calc_centroids(self):
        """Find the centroid of every fine-mesh block"""
        centroids = []
        zs = self.calc_zs()
        if self.geom_type == "hex-z":
            pitch = (self.xmesh[1] - self.xmesh[0]) / self.ifints[0]
            half_pitch = 0.5 * pitch
            if self.triangle_option == 'rhomb-120':
                lower_left = self.calc_bounding_box()[0, :]
                for c_idx, f_idx, idx in self._iterate():
                    x = lower_left[0] + pitch * c_idx[2] + \
                        half_pitch * (c_idx[1] + 1.)
                    y = lower_left[1] + pitch / np.sqrt(3.) * \
                        (1.5 * c_idx[1] + 1)
                    zlo = zs[idx[0]]
                    zhi = zs[idx[0] + 1]
                    z = zlo + (zhi - zlo) * 0.5
                    centroids.append((x, y, z))
            elif self.triangle_option == 'rhomb-60':
                cos120 = -0.5
                sin120 = 0.5 * np.sqrt(3)
                for c_idx, f_idx, idx in self._iterate():
                    x0 = pitch * (c_idx[1] + c_idx[2])
                    hypotenuse = pitch * c_idx[1]
                    x = x0 + cos120 * hypotenuse
                    y = hypotenuse * sin120
                    zlo = zs[idx[0]]
                    zhi = zs[idx[0] + 1]
                    z = zlo + (zhi - zlo) * 0.5
                    centroids.append((x, y, z))
        elif self.geom_type == "x-y-z":
            ys = self.calc_ys()
            xs = self.calc_xs()
            for c_idx, f_idx, idx in self._iterate():
                x = 0.5 * (xs[idx[2] + 1] - xs[idx[2]]) + xs[idx[2]]
                y = 0.5 * (ys[idx[1] + 1] - ys[idx[1]]) + ys[idx[1]]
                z = 0.5 * (zs[idx[0] + 1] - zs[idx[0]]) + zs[idx[0]]
                centroids.append((x, y, z))
        else:
            raise NotImplementedError("Only implemented for hex-z, " +
                                      "rhomb-120 and x-y-z geometries!")
        return centroids

    def calc_volumes(self):
        """Calculate mesh cell volumes"""
        zs = self.calc_zs()
        volumes = np.zeros(self.fine_dims)
        if self.geom_type.startswith("hex"):
            pitch = (self.xmesh[1] - self.xmesh[0]) / self.ifints[0]
            hex_area = 0.5 * np.sqrt(3.) * pitch * pitch
            # determine periodicity: 60 deg, 120 deg, or full hex
            x_center = 1.0
            x_edge = 1.0
            if self.bcs[2, 0] == 4:  # 60/120 periodicity
                x_edge = 0.5
                if self.triangle_option == "rhomb-60":
                    x_center = 0.1666666666666666667
                else:  # self.triangle_option == "rhomb-120":
                    x_center = 0.3333333333333333333
        for c_idx, f_idx, idx in self._iterate():
            height = zs[idx[0] + 1] - zs[idx[0]]
            if self.geom_type in ['hex', 'hex-z']:  # igom
                if idx[1] == 0 and idx[2] == 0:
                    volumes[idx] = hex_area * height * x_center
                elif idx[1] == 0 or idx[2] == 0:
                    volumes[idx] = x_edge * hex_area * height
                else:
                    volumes[idx] = hex_area * height

            elif self.geom_type == "x-y-z":
                ys = self.calc_ys()
                xs = self.calc_xs()
                volumes[idx] = height * (ys[idx[1] + 1] - ys[idx[1]]) \
                    * (xs[idx[2] + 1] - xs[idx[2]])
        return volumes

    def calc_bounding_box(self):
        """Calculate the box bounding each mesh cell"""
        box = np.zeros((8, 3))
        if self.geom_type == "hex-z":  # Get some constants first
            pitch = (self.xmesh[1] - self.xmesh[0]) / self.ifints[0]
            A = 2. * pitch / np.sqrt(3.)  # Side length
            s = A / 2.
            if self.triangle_option == 'rhomb-120':
                # Points will be:
                #     /1             2/
                #    /               /
                #   /               /
                #  /               /
                # /0             3/
                # On the bottom and then continuing, in the same order,
                # on the top
                # Lower left is easy
                z = self.zmesh[0]
                box[0, :] = np.array([self.xmesh[0], self.ymesh[0], z])
                # Upper left
                y = self.coarse_dims[1] * A - \
                    (self.coarse_dims[1] - 1) * s * np.cos(np.pi / 3.)
                x = y / np.tan(np.pi / 3.)
                box[1, :] = box[0, :] + np.array([x, y, z])
                # Lower right
                x = pitch * self.coarse_dims[1]
                y = self.ymesh[0]
                box[3, :] = box[0, :] + np.array([x, y, z])
                # Upper right
                box[2, :] = box[3, :] + box[1, :]

            elif self.triangle_option == 'rhomb-60':
                # Points will be:
                #     /1             2/
                #    /               /
                #   /               /
                #  /               /
                # /0             3/
                # On the bottom and then continuing, in the same order,
                # on the top
                # Lower left is easy
                z = self.zmesh[0]
                box[0, :] = np.array([self.xmesh[0], self.ymesh[0], z])
                # Upper left
                y = pitch * self.coarse_dims[1] * np.sin(np.pi / 3.)
                # This is the number of hexagons along this dimension,
                # but it goes up to the center of the next row above
                # we want it to go just to the top of the hexes on the
                # row below, so subtract off the distance from the midpt
                # to the bottom side vertices of the hex
                y -= s
                x = y / np.tan(np.pi / 3.)
                box[1, :] = box[0, :] + np.array([x, y, z])
                # Lower right
                x = pitch * (self.coarse_dims[1] - 0.5)
                y = self.ymesh[0]
                box[3, :] = box[0, :] + np.array([x, y, z])
                # Upper right
                box[2, :] = box[3, :] + box[1, :]

        elif self.geom_type == "x-y-z":
            z = self.zmesh[0]
            box[0, :] = np.array([self.xmesh[0], self.ymesh[0], z])
            box[1, :] = np.array([self.xmesh[0], self.ymesh[-1], z])
            box[2, :] = np.array([self.xmesh[-1], self.ymesh[-1], z])
            box[3, :] = np.array([self.xmesh[-1], self.ymesh[0], z])

        else:
            raise NotImplementedError("Only implemented for hex-z, " +
                                      "rhomb-120 and x-y-z geometries!")

        # Now do the axial top by just changing the z value
        box[4:, :] = box[:4, :]
        box[4:, 2] = self.zmesh[-1]
        return box

    def plot_points(self, only_filled=True):
        """Plot the mesh points"""
        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        centroids = self.calc_centroids()
        centroid_index = 0
        plot_xs = []
        plot_ys = []
        plot_zs = []

        for mat_id, idx in self.region_assignments_iterate():
            if only_filled:
                if mat_id != 0:
                    x, y, z = centroids[centroid_index]
                    plot_xs.append(x)
                    plot_ys.append(y)
                    plot_zs.append(z)
            else:
                x, y, z = centroids[centroid_index]
                plot_xs.append(x)
                plot_ys.append(y)
                plot_zs.append(z)
            centroid_index += 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(plot_xs, plot_ys, plot_zs)

        box = self.calc_bounding_box()
        ax.scatter(box[:, 0], box[:, 1], box[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def _iterate(self):
        """Iterate through the coarse and fine mesh indices,
        returning: (c_k, c_j, c_i), (f_k, f_j, f_i), (k, j, i)"""
        k = 0
        for c_k in range(self.coarse_dims[0]):
            for f_k in range(self.kfints[c_k]):
                j = 0
                for c_j in range(self.coarse_dims[1]):
                    for f_j in range(self.jfints[c_j]):
                        i = 0
                        for c_i in range(self.coarse_dims[2]):
                            for f_i in range(self.ifints[c_i]):
                                yield((c_k, c_j, c_i), (f_k, f_j, f_i),
                                      (k, j, i))
                                i += 1
                        j += 1
                k += 1

    def region_assignments_iterate(self, rotate=True):
        """ """
        for c_idx, f_idx, idx in self._iterate():
            if self.assign_to_coarse:
                k, j, i = c_idx
            else:
                k, j, i = f_idx
            # Change the GEODST ordering to match
            # that printed in DIF3D's maps
            reg_assignments = self.reg_assignments[k, :, :]
            if rotate:
                if self.geom_type == "hex-z":
                    if self.triangle_option == 'rhomb-120':
                        reg_assignments = np.rot90(reg_assignments)
            if reg_assignments[j, i] == 0:
                composition_number = 0
            else:
                composition_number = self.nznr[(reg_assignments[j, i]
                                                - 1)]
            yield composition_number, idx


########################################################################


def get_1D(data, geodst):
    """Read the GEODST 1D record (file specifications)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    data = read_record.discard_pad(data)
    data, words = read_record.get_array_of_int(27, data)
    data = read_record.discard_pad(data)
    keys = ["igom", "nzone", "nreg", "nzcl", "ncinti", "ncintj",
            "ncintk", "ninti", "nintj", "nintk", "imb1", "imb2",
            "jmb1", "jmb2", "kmb1", "kmb2", "nbs", "nbcs", "nibcs",
            "nzwbb", "ntriag", "nrass", "nthpt", "ngop(1)", "ngop(2)",
            "ngop(3)", "ngop(4)"]
    geodst["1D"] = OrderedDict()
    for i in range(len(keys)):
        geodst["1D"][keys[i]] = words[i]
    return data, geodst


def get_2D(data, geodst):
    """Read the GEODST 2D record (one dimensional coarse mesh
    interval boundaries and fine mesh intervals)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["2D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    if 0 < igom <= 3:
        ncinti = geodst["1D"]["ncinti"]
        data = read_record.discard_pad(data)
        # Get xmesh
        data, xmesh = read_record.get_array_of_double(ncinti + 1, data)
        # Get ifints
        data, ifints = read_record.get_array_of_int(ncinti, data)
        data = read_record.discard_pad(data)
        geodst["2D"]["xmesh"] = xmesh
        geodst["2D"]["ifints"] = ifints
    return data, geodst


def get_3D(data, geodst):
    """Read the GEODST 3D record (two dimensional coarse mesh
    interval boundaries and fine mesh intervals)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["3D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    if 6 <= igom <= 11:
        ncinti = geodst["1D"]["ncinti"]
        ncintj = geodst["1D"]["ncintj"]
        data = read_record.discard_pad(data)
        # Get xmesh and ymesh
        data, xmesh = read_record.get_array_of_double(ncinti + 1, data)
        data, ymesh = read_record.get_array_of_double(ncintj + 1, data)
        # Get ifints and jfints
        data, ifints = read_record.get_array_of_int(ncinti, data)
        data, jfints = read_record.get_array_of_int(ncintj, data)
        data = read_record.discard_pad(data)
        geodst["3D"]["xmesh"] = xmesh
        geodst["3D"]["ymesh"] = ymesh
        geodst["3D"]["ifints"] = ifints
        geodst["3D"]["jfints"] = jfints
    return data, geodst


def get_4D(data, geodst):
    """Read the GEODST 4D record (three dimensional coarse mesh
    interval boundaries and fine mesh intervals)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["4D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    if igom >= 12:
        ncinti = geodst["1D"]["ncinti"]
        ncintj = geodst["1D"]["ncintj"]
        ncintk = geodst["1D"]["ncintk"]
        data = read_record.discard_pad(data)
        # Get xmesh and ymesh
        data, xmesh = read_record.get_array_of_double(ncinti + 1, data)
        data, ymesh = read_record.get_array_of_double(ncintj + 1, data)
        data, zmesh = read_record.get_array_of_double(ncintk + 1, data)
        # Get ifints and jfints
        data, ifints = read_record.get_array_of_int(ncinti, data)
        data, jfints = read_record.get_array_of_int(ncintj, data)
        data, kfints = read_record.get_array_of_int(ncintk, data)
        data = read_record.discard_pad(data)
        geodst["4D"]["xmesh"] = xmesh
        geodst["4D"]["ymesh"] = ymesh
        geodst["4D"]["zmesh"] = zmesh
        geodst["4D"]["ifints"] = ifints
        geodst["4D"]["jfints"] = jfints
        geodst["4D"]["kfints"] = kfints
    return data, geodst


def get_5D(data, geodst):
    """Read the GEODST 5D record (geometry data)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["5D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    nbs = geodst["1D"]["nbs"]
    if igom > 0 or nbs > 0:
        nreg = geodst["1D"]["nreg"]
        nbcs = geodst["1D"]["nbcs"]
        nibcs = geodst["1D"]["nibcs"]
        nzwbb = geodst["1D"]["nzwbb"]
        nzone = geodst["1D"]["nzone"]
        data = read_record.discard_pad(data)
        data, volr = read_record.get_array_of_float(nreg, data)
        data, bsq = read_record.get_array_of_float(nbs, data)
        data, bndc = read_record.get_array_of_float(nbcs, data)
        data, bnci = read_record.get_array_of_float(nibcs, data)
        data, nzhbb = read_record.get_array_of_int(nzwbb, data)
        data, nzc = read_record.get_array_of_int(nzone, data)
        data, nznr = read_record.get_array_of_int(nreg, data)
        data = read_record.discard_pad(data)
        geodst["5D"]["volr"] = volr
        geodst["5D"]["bsq"] = bsq
        geodst["5D"]["bndc"] = bndc
        geodst["5D"]["bnci"] = bnci
        geodst["5D"]["nzhbb"] = nzhbb
        geodst["5D"]["nzc"] = nzc
        geodst["5D"]["nznr"] = nznr
    return data, geodst


def get_6D(data, geodst):
    """Read the GEODST 6D record (region assignments to
    coarse mesh intervals)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["6D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    nrass = geodst["1D"]["nrass"]
    if igom > 0 and nrass == 0:
        ncinti = geodst["1D"]["ncinti"]
        ncintj = geodst["1D"]["ncintj"]
        ncintk = geodst["1D"]["ncintk"]
        mr = []
        for k in range(ncintk):
            data = read_record.discard_pad(data)
            data, mr_temp = read_record.get_array_of_int((ncinti
                                                          * ncintj),
                                                         data)
            mr.append(mr_temp[:])
            data = read_record.discard_pad(data)
        geodst["6D"]["mr"] = mr
    return data, geodst


def get_7D(data, geodst):
    """Read the GEODST 7D record (region assignments to
    fine mesh intervals)

    Parameters
    ----------
    data : str
        GEODST binary file
    geodst : OrderedDict
        Object to hold values read from binary file

    Returns
    -------
    str
        Truncated GEODST binary file
    OrderedDict
        Updated object with values from 1D record

    """
    geodst["7D"] = OrderedDict()
    igom = geodst["1D"]["igom"]
    nrass = geodst["1D"]["nrass"]
    if igom > 0 and nrass == 1:
        ninti = geodst["1D"]["ninti"]
        nintj = geodst["1D"]["nintj"]
        nintk = geodst["1D"]["nintk"]
        mr = []
        for k in range(nintk):
            data = read_record.discard_pad(data)
            data, mr_temp = read_record.get_array_of_int((ninti
                                                          * nintj),
                                                         data)
            mr.append(mr_temp[:])
            data = read_record.discard_pad(data)
        geodst["7D"]["mr"] = mr
    return data, geodst
