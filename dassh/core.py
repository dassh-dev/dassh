########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2020-12-14
author: matz
Methods to describe the layout of assemblies in the reactor core and
the coolant in the gap between them
"""
# Next steps:
# 1. For each assembly:

# - map subchannels around reactor core
# - assign XY coords to assembly centroids
# - instantiate assembly objects for each assembly type/ID
# - interpolation between inter-assembly gap subchannels
########################################################################
import numpy as np
import copy
from dassh.logged_class import LoggedClass
from dassh.correlations import nusselt_db

# from py4c import geodst


_sqrt3 = np.sqrt(3)
# Directions around assembly
_dirs = {}
# _dirs[0] = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]
_dirs[0] = [(0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0)]
_dirs[2] = _dirs[0][1:] + [_dirs[0][0]]
_dirs[1] = _dirs[0][2:] + _dirs[0][:2]


class Core(LoggedClass):
    """Map the reactor core using the GEODST binary file; set up
    parameters to calculate interassembly gap subchannel temperatures

    Parameters
    ----------
    path_to_geodst : str
        Path to GEODST binary file
    gap_flow_rate : float
        Interassembly gap flow rate (kg/s)
    coolant_obj : float
        DASSH Material object for the interassembly gap coolant
    inlet_temperature : float
        Inlet coolant temperature (K)
    test : bool (optional)
        If testing, do not run all the instantiation methods; instead,
        allow the object to be instantiated without calling them so
        they can be called incrementally and independently

    Notes
    -----
    DASSH is intended for use with GEODST files generated from DIF3D-
    VARIANT. The geometry for VARIANT calculations is somewhat
    restricted to hexagonal geometries, for which each "region"
    corresponds to the location of an assembly. The values in the
    GEODST region maps correspond to region groups -- groups of
    assemblies (i.e. "inner core", "outer core", "driver", etc.).
    Here, "region" is used to refer to the values in the GEODST
    region map; "assembly" is used to refer to the individual
    positions of each region, regardless of its group.

    """

    def __init__(self, geodst_obj, gap_flow_rate, coolant_obj,
                 inlet_temperature=273.15, model='flow',
                 test=False):
        """Instantiate Core object."""
        LoggedClass.__init__(self, 4, 'dassh.core.Core')
        # Check that all region maps in GEODST are square
        if not all(reg_xy.shape[0] == reg_xy.shape[1]
                   for reg_xy in geodst_obj.reg_assignments):
            self.log('error', 'Was expecting all axial MR (region map) '
                              + 'arrays from GEODST py4c object to be '
                              + 'square')

        if model not in ['flow', 'no_flow', 'duct_average', None]:
            msg = 'Do not understand input inter-assembly gap model: '
            self.log('error', msg + model)

        # --------------------------------------------------------------
        # Identify GEODST periodicity: if not full core, it can be
        # either 60 or 120 degree
        self.hex_option = self._identify_periodicity(geodst_obj)
        self.n_ring = self._calc_nring(geodst_obj.reg_assignments[0])
        self.asm_pitch = ((geodst_obj.xmesh[1] - geodst_obj.xmesh[0])
                          / geodst_obj.ifints[0])
        self.asm_pitch /= 100.0  # cm -> m
        self.gap_coolant = coolant_obj
        self.gap_coolant.update(inlet_temperature)
        self.gap_flow_rate = gap_flow_rate
        self.coolant_gap_params = \
            {'Re': 0.0,  # bundle-average Reynolds number
             'Re_sc': np.zeros(2),  # subchannel Reynolds numbers
             'vel': 0.0,  # bundle-average coolant velocity
             'ff': np.zeros(2),  # subchannel friction factors
             'htc': np.zeros(2)}  # heat transfer coefficients
        self.z = [0.0]
        self.model = model

        # --------------------------------------------------------------
        # Don't run the more complex, tested methods if testing; let
        # them be implemented independently in pytest.
        if test:
            return

        # --------------------------------------------------------------
        # ASSEMBLY AND SUBCHANNEL MAPPING
        # Assign each assembly (region) in the X-Y region map an ID
        # number; although the regions will change between axial
        # meshes, the filled positions will not. Therefore, this
        # map only needs to be made for one axial mesh.
        # Sets attribute self.asm_map
        self.asm_map = self.map_asm(geodst_obj.reg_assignments[0])

        # Map neighbors for each assembly based on problem symmetry
        # Again, only one map needed for all axial meshes
        # Sets attribute self.asm_adj
        self.asm_adj = self.map_adjacent_assemblies()

        # Get list of region values corresponding to each assembly
        # ID; this allows us to connect each assembly ID to a type
        # of assembly. This needs to be done for every axial mesh.
        self.regions = self.list_regions(geodst_obj)

    ####################################################################
    # SETUP METHODS
    ####################################################################

    def _identify_periodicity(self, geodst_obj):
        """Identify periodic surface setup (symmetry) in hex geom"""
        hex_option = 0  # full core; no work to do
        if geodst_obj.geom_type in ['hex', 'hex-z']:  # igom
            if geodst_obj.bcs[2, 0] == 4:  # imb4
                hex_option = 1  # 120 symmetry/periodic:
                if geodst_obj.triangle_option == "rhomb-60":  # ntriag
                    hex_option = 2  # 60 symmetry/periodic
        return hex_option

    def _calc_nring(self, geodst_regmap):
        """Calculate the number of assembly rings

        Parameters
        ----------
        geodst_regmap : numpy.ndarray
            Region assignments (N x N) array from a py4c GEODST object

        Returns
        -------
        int
            Number of assembly rings in the core

        """
        if self.hex_option == 2:  # 60 degree periodic
            gmap = np.rot90(geodst_regmap)
            i = 1
            while not all(np.diagonal(gmap, offset=i - len(gmap)) == 0):
                i += 1
            nring = int(i - 1)
        elif self.hex_option == 1:  # 120 degree periodic
            nring = geodst_regmap.shape[0]
        else:
            nring = int((geodst_regmap.shape[0] + 1) / 2)
        return nring

    # ASSEMBLY MAP -----------------------------------------------------

    def map_asm(self, regmap):
        r"""Map the assembly locations in the core.

        Parameters
        ----------
        regmap : numpy.ndarray
            Map of the regions at one axial level obtained by
            processing the GEODST binary file using py4c

        Returns
        -------
        numpy.ndarray
            Map of assemblies in the core

        Notes
        -----
        The assemblies are numbered starting at the center assembly
        (1) and continuing outward around each of the rings. The
        first assembly of the next ring is that located on the
        diagonal immediately above the center assembly. The
        assemblies in each ring are labeled by traveling clockwise
        around the ring.

        A regular hexagon can be divided by three straight lines along
        the long diagonals that pass through the corners and intersect
        at the center. One of these runs straight up and down; the
        second has an angle of 30 degrees from horizontal; the third
        has an angle of 150 degrees from horizontal.

        This assembly numbering scheme and division of the hexagon can
        be used to map the assembly labels from the hexagon to a square
        matrix, which uses mimics the three long diagonals in the pin
        with the rows, columns, and one diagonal.

        Example
        -------
        If the center assembly is labeled "1", then the second ring of
        pins may be labeled:

        NORTH (y)
           \\                EAST (x)
            \\              .#
              2 _____ 3  .#
              /\    / \ #
           7 /___\1/___\ 4
             \   / \   /
              \/_____\/
              6       5

        The map for this 7-assembly core would be:
                             ____
        | 2 3 0 |           | 2 3  \
        | 7 1 4 |   (note:  | 7 1 4 | looks like a hexagon!)
        | 0 6 5 |            \ _6_5_|

        Periodicity
        -----------
        The GEODST file may only specify 1/6 or 1/3 of the core,
        which implies 60 or 120 degree periodicity. In that case,
        the array obtained from the GEODST region map is only of
        the partial core. These cases and the full core case are
        handled separately within this method.

        """
        # Directions turning clockwise around a hexagon
        # First entry is step from an inner ring to the top of an
        # outer ring; the remaining steps are the turns around the
        # hexagon corners
        _dirs = [(-1, -1), (0, 1), (1, 1), (1, 0),
                 (0, -1), (-1, -1), (-1, 0)]
        asm_map = np.zeros((self.n_ring * 2 - 1,
                            self.n_ring * 2 - 1), dtype=int)
        asm_idx = 1
        center = self.n_ring - 1

        # Expand periodic region maps to match size of full core map
        if self.hex_option != 0:
            temp = asm_map
            regmap[1:, 0] = 0  # remove duplicated diagonal
            if self.hex_option == 2:  # 60 degree periodic
                # 60 degree regmap is not necessarily nring x nring
                # Need to explicitly account for that missing space
                regmap = np.rot90(regmap)
                temp[(center - len(regmap) + 1):(center + 1),
                     center:(len(regmap) + center)] = regmap
                regmap = temp
                assert len(np.nonzero(np.diag(regmap))) <= 1
            else:  # 120 degree periodic; self.hex_option == 1
                # 120 degree regmap should be nring x nring
                temp[(self.n_ring - 1):, (self.n_ring - 1):] = regmap
                regmap = temp
                assert len(np.nonzero(np.diag(np.rot90(regmap)))) <= 1

        # Fill the first position at the center assembly
        if regmap[self.n_ring - 1, self.n_ring - 1] != 0:
            asm_map[self.n_ring - 1, self.n_ring - 1] = asm_idx
            asm_idx += 1

        # Loop clockwise around the rings to identify the assemblies
        for ring in range(2, int(self.n_ring + 1)):
            row = self.n_ring - ring
            col = self.n_ring - ring
            positions = 6 * (ring - 1)  # all positions on active ring
            corners = np.arange(0, positions, ring - 1)
            d = 1  # first direction
            for pos in range(0, int(positions)):
                # The active position may be 0 in reg_assignments,
                # meaning that there's no region there. In that case,
                # skip; otherwise, fill empty map entry.
                if regmap[row, col] != 0:
                    asm_map[row, col] = asm_idx
                    asm_idx += 1
                if pos > 0 and pos in corners:
                    d += 1  # change directions at corner
                row, col = row + _dirs[d][0], col + _dirs[d][1]

        # Trim zero rows and columns from periodic cases
        return self._strip_asm_map_zeros(asm_map)

    def _strip_asm_map_zeros(self, asm_map):
        r"""Remove the zero rows and columns from the assembly maps
        generated for the periodic cases

        Notes
        -----
        The zero regions in the hexagon are:
        60 degree periodic          120 degree periodic
        [   _____       ]           [   _____       ]
        [ | \ 0 | \     ]           [ | \ 0 | \     ]
        [ |  \  |  \ 0  ]           [ |  \  |  \ 0  ]
        [ | 0 \ | x \   ]           [ | 0 \ | 0 \   ]
        [ |____\|____\  ]           [ |____\|____\  ]
        [  \    | \ 0 | ]           [  \    | \ x | ]
        [   \ 0 |  \  | ]           [   \ 0 |  \  | ]
        [ 0  \  | 0 \ | ]           [ 0  \  | x \ | ]
        [     \ |____\| ]           [     \ |____\| ]

        """
        if self.hex_option == 2:
            asm_map = asm_map[0:self.n_ring, (self.n_ring - 1):]
        elif self.hex_option == 1:
            asm_map = asm_map[(self.n_ring - 1):, (self.n_ring - 1):]
        else:
            pass
        return asm_map

    # FIND NEIGHBORS (Adjacent assemblies) -----------------------------

    def map_adjacent_assemblies(self):
        r"""Identify assembly neighbors for each assembly in the core.

        Parameters
        ----------
        geodst_regmap : numpy.ndarray
            Map of the regions at one axial level obtained by
            processing the GEODST binary file using py4c

        Returns
        -------
        numpy.ndarray
            Adjacent assemblies for each assembly in the core

        Notes
        -----
        Array returned has shape (n_asm x 6). Empty positions
        along the core edges are returned as zeros. The below shows
        the neighbor index and the assembly number in parentheses

            7(6)  /\  2(7)
                /    \
         6(5)  | 1(1) |  3(2)
               |      |
                \    /      y
            5(4)  \/  4(3)  |__x

        Relative to the assembly map array we made, that order is:

            2 - 3
            | \ |  \
            1 - x - 4
             \  | \ |
                6 - 5

        ...where "x" is the "active assembly" of interest.

        """
        adj = np.zeros((np.max(self.asm_map), 6), dtype=int)
        if self.hex_option == 2:  # 60 DEGREE PERIODIC
            raise NotImplementedError()
            adj[0] = np.ones(6) * 2
            asm_map = self._expand_periodic_asm_map()
            for row in range(0, self.n_ring - 1):
                for col in range(1, self.n_ring):
                    id = asm_map[row, col]
                    if id != 0:
                        adj[id - 1] = self._get_neighbors(asm_map,
                                                          (row, col))

        elif self.hex_option == 1:  # 120 DEGREE PERIODIC
            raise NotImplementedError()
            adj[0] = np.array([2, 3] * 3)
            asm_map = self._expand_periodic_asm_map()
            for row in range(1, self.n_ring + 1):
                for col in range(1, self.n_ring):
                    id = asm_map[row, col]
                    if id != 0:
                        adj[id - 1] = self._get_neighbors(asm_map,
                                                          (row, col))

        else:  # FULL CORE
            for row in range(0, len(self.asm_map)):
                for col in range(0, len(self.asm_map[row])):
                    id = self.asm_map[row, col]
                    if id != 0:
                        adj[id - 1] = self._get_neighbors(self.asm_map,
                                                          (row, col))
        # 2020-09-25: This change "rotates" the way in which adjacent
        # assemblies are counted to line up the X-Y axes in DASSH with
        # those in DIF3D
        # adj = adj[:, [5, 0, 1, 2, 3, 4]]
        return adj

    def _get_neighbors(self, asm_map, loc):
        """Identify the neighbors for one assembly

        Parameters
        ----------
        asm_map : numpy.ndarray
            Map (expanded if periodic case) of the assemblies at one
            axial level obtained by processing the GEODST binary file
            using py4c
        loc : tuple
            Row, column location of active assembly ID in asm_map

        Returns
        -------
        numpy.ndarray
            IDs for the assemblies adjacent to the active assembly

        Notes
        -----
        Array returned has shape (1 x 6). Empty positions
        along the core edges are returned as zeros.

        """
        adjacent_asm = np.zeros(6, dtype=int)
        # The order in which neighbors are counted is different for
        # each periodicity case depending on the orientation of the
        # assembly map - want to be counting out along the major
        # diagonal onto the next rings.
        # dirs = {}
        # dirs[0] = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]
        # dirs[2] = dirs[0][1:] + [dirs[0][0]]
        # dirs[1] = dirs[0][2:] + dirs[0][:2]
        for i in range(0, len(_dirs[self.hex_option])):
            address = tuple(sum(x) for x in
                            zip(loc, _dirs[self.hex_option][i]))
            if all(idx >= 0 for idx in address):
                try:
                    adjacent_asm[i] = asm_map[address]
                except IndexError:
                    pass
        return adjacent_asm

    def _expand_periodic_asm_map(self):
        """Expand a periodic assembly map to show assemblies
        in the next segment

        Notes
        -----
        This method enables the mapping of neighbors by expanding
        the periodic assembly ID map beyond the bounds of the
        mapped segment into the adjacent ones, allowing the mapping
        method to handle those assemblies that are on the segment
        border.

        """
        asm_map = copy.deepcopy(self.asm_map)
        to_add = np.zeros(len(asm_map), dtype=int)
        if self.hex_option == 2:  # 60 degree periodic
            asm_map[-1, :] = asm_map[:, 0][::-1]
            to_add[:(len(asm_map) - 1)] = asm_map[:, 1][1:]
            to_add.shape = (len(to_add), 1)
            asm_map = np.hstack((to_add, asm_map))
        elif self.hex_option == 1:  # 120 degree periodic
            asm_map[:, 0] = asm_map[0, :]
            to_add[:-1] = asm_map[:, 1][1:]
            asm_map = np.vstack((to_add, asm_map))
        else:
            pass
        return asm_map

    # CATALOG REGIONS --------------------------------------------------

    def list_regions(self, geodst_obj):
        """Catalog the GEODST region IDs with the DASSH assembly IDs.
        This is done for each axial region map.

        Parameters
        ----------
        regmap : numpy.ndarray
            Region map at an axial level from GEODST py4c object

        """
        regions = []
        for rmap_z in geodst_obj.reg_assignments:
            reg = np.zeros(np.max(self.asm_map), dtype=int)
            # 60 degree periodic asm_map may need expansion
            # to be of size n_ring x n_ring
            if self.hex_option == 2:
                tmp = np.zeros((self.n_ring, self.n_ring), dtype=int)
                tmp[(self.n_ring - len(rmap_z)):,
                    0:len(rmap_z)] = np.rot90(rmap_z)
                rmap_z = tmp
            for row in range(0, len(self.asm_map)):
                for col in range(0, len(self.asm_map[row])):
                    if self.asm_map[row, col] != 0:
                        reg[self.asm_map[row, col] - 1] = rmap_z[row,
                                                                 col]
            regions.append(reg)
        return np.vstack(regions)

    ####################################################################
    # SETUP METHODS (cont); called outside instantiation
    ####################################################################

    def load(self, asms):
        """Load assemblies into core positions and import their
        characteristics

        Parameters
        ----------
        asms : list
            List of DASSH.Assembly objects

        Returns
        -------
        None

        Notes
        -----
        Although the inter-assembly gap coolant is tracked on the
        finest mesh, information is required for all assemblies in
        order to approximate duct wall and coolant temperatures back
        and forth between finer and coarser meshes.

        """
        self.n_asm = len(asms)
        # Remove the assembly locations that we didn't load
        self.asm_map[self.asm_map > self.n_asm] = 0
        # Set up the gap mesh parameters based on the assembly with
        # the most pins (creates the finest meshing)
        idx = 0
        n_pin = 0
        for ai in range(len(asms)):
            if asms[ai].has_rodded and asms[ai].rodded.n_pin > n_pin:
                n_pin = asms[ai].rodded.n_pin
                idx = ai  # <-- use this assembly

        # Get the "x" coordinates at which the duct temperatures
        # for each assembly will be given; this is for use *along
        # each hex side* including corners.
        self.x_pts = asms[idx]._finest_xpts

        # If it has precalculated least-squares fit params, take 'em
        if hasattr(asms[idx], '_lstsq_params'):
            self._lstsq_params = asms[idx]._lstsq_params

        # Pull some parameters from that assembly
        # Subchannels per asm and per hex side (incl. no corners)
        self._sc_per_asm = 6 * (len(self.x_pts) - 1)
        self._sc_per_side = len(self.x_pts) - 2

        # Some things I don't need to store but want to pull out
        asm_dftf = asms[idx].duct_oftf  # Duct outer FTF
        asm_pin_pitch = 0.0
        if asms[idx].has_rodded:
            asm_pin_pitch = asms[idx].rodded.pin_pitch

        # Set up interassembly gap subchannel adjacency
        self.asm_sc_adj = self.map_interassembly_sc()
        self.sc_types = np.array(self.determine_sc_types()) - 1
        self.sc_adj = self.find_adjacent_sc()

        # Set up inter-assembly gap subchannel sizes and dimensions
        self.d = {}
        self.d['gap'] = self.asm_pitch - asm_dftf  # Gap "width"
        hex_side = asm_dftf / _sqrt3  # Hex side length
        # duct exterior "wall corner" length
        self.d['wcorner'] = 0.5 * (hex_side - (asm_pin_pitch
                                               * self._sc_per_side))

        # Subchannel to subchannel centroid distances
        self.L = [[0.0, 0.0], [0.0, 0.0]]
        self.L[0][0] = asm_pin_pitch  # edge - edge
        # Edge-corner includes (1) half edge-edge, (2) wall-corner
        # distance, and (3) the dist to the centroid of the lil
        # equilateral triangle at the center of the corner subchannel
        # (edge length = d_gap), which is 1/3 of its height
        self.L[0][1] = (0.5 * self.L[0][0] + self.d['wcorner']
                        + _sqrt3 * self.d['gap'] / 2 / 3)
        self.L[1][0] = self.L[0][1]
        # Corner-corner distance is similar to edge corner (w/o term 1)
        self.L[1][1] = self.d['wcorner'] + _sqrt3 * self.d['gap'] / 6

        # Gap subchannel overall params
        self.gap_params = {}
        # Subchannel area - corner includes the lil triangle formed
        # between three assemblies
        self.gap_params['area'] = np.zeros(2)
        self.gap_params['area'][0] = self.L[0][0] * self.d['gap']
        self.gap_params['area'][1] = (3 * self.d['wcorner']
                                      * self.d['gap']
                                      + self.d['gap']**2 * _sqrt3 / 4)
        self.gap_params['de'] = np.zeros(2)
        if self.L[0][0] != 0.0:
            self.gap_params['de'][0] = (4 * self.gap_params['area'][0]
                                        / 2 / self.L[0][0])
        self.gap_params['de'][1] = (4 * self.gap_params['area'][1]
                                    / 6 / self.d['wcorner'])

        # Total area calculation: split into pieces
        # # 1. Length along to assembly hex sides
        # A1 = self.n_asm * hex_side * 6 * self.d['gap']
        # # 2. "little triangle" formed between every 3 assemblies
        # A2 = np.floor(self.n_asm / 3.0) * self.d['gap']**2 * _sqrt3 / 4
        # self.gap_params['total area'] = A1 + A2
        self.gap_params['total area'] = 0.0
        for sc in range(len(self.sc_types)):
            self.gap_params['total area'] += \
                self.gap_params['area'][self.sc_types[sc]]

        # Hydraulic diameter = 4*A/wp (wetted perim)
        self.gap_params['total wp'] = 6 * hex_side * self.n_asm
        self.gap_params['total de'] = (4 * self.gap_params['total area']
                                       / self.gap_params['total wp'])

        # Interior coolant temperatures; shape = n_axial_mesh x n_sc
        self.coolant_gap_temp = (np.ones(len(self.sc_adj))
                                 * self.gap_coolant.temperature)
        # Update coolant gap params based on inlet temperature
        self._update_coolant_gap_params(self.gap_coolant.temperature)

        # --------------------------------------------------------------
        # HEAT TRANSFER CONSTANTS - set up similarly to self.L
        if self.model == 'flow':
            self.ht_consts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            # Conduction between coolant channels
            for i in range(2):
                for j in range(2):
                    if self.L[i][j] != 0.0:
                        self.ht_consts[i][j] = \
                            (self.d['gap']
                             * self.gap_params['total area']
                             / self.L[i][j]
                             / self.gap_flow_rate
                             / self.gap_params['area'][i])
            # Convection from coolant to duct wall (units: m-s/kg)
            # Edge -> wall 1
            if self.gap_params['area'][0] != 0:
                self.ht_consts[0][2] = (self.L[0][0]
                                        * self.gap_params['total area']
                                        / self.gap_flow_rate
                                        / self.gap_params['area'][0])
            # Corner -> wall 1
            self.ht_consts[1][2] = (2 * self.d['wcorner']
                                    * self.gap_params['total area']
                                    / self.gap_flow_rate
                                    / self.gap_params['area'][1])

        # Set up convection/conduction utility attributes
        self._make_conv_mask()
        self._make_cond_mask()

        # Set up energy balance attribute
        self._ebal = {}

        # Keep two energy balances:
        # 1) Track the energy given from the assemblies to the gap
        #    coolant; this should match roughly with what the assembly
        #    energy balance reports is lost through the outermost duct.
        #    The match won't be exact because the assembly calculation
        #    assumes heat from the corner subchannels is transferred
        #    through an area equal to that at the midplane of the duct,
        #    whereas here it is transferred through the area at the
        #    outside of the duct, since that is the same for all
        #    assemblies. I have confirmed that adjusting for this gives
        #    the matching result. This array will not sum to zero.
        self._ebal['asm'] = np.zeros((self.n_asm, self._sc_per_asm))

        # 2) Track the energy balance on the gap coolant itself. Even
        #    if both assemblies are giving heat the the gap coolant,
        #    the assembly with higher temperatures will give more /
        #    accept less. Therefore, this balance will sum to zero and
        #    will reflect the transfer of heat between assemblies.
        self._ebal['interasm'] = np.zeros((self.n_asm, self._sc_per_asm))

    # MAP INTER-ASSEMBLY GAP -------------------------------------------

    def map_interassembly_sc(self):
        """Map the interassembly gap subchannels that surround each
        assembly in the core.

        Returns
        -------
        numpy.ndarray
            Contains the indices for the interassembly gap subchannels
            that surround each assembly (sc_per_side+1 x 6)

        Notes
        ----------
        sc_per_side : int
            Number of *side* subchannels per side; equal to the total
            number of subchannels surrounding the assembly minus the
            six corners and divided by the six sides.
        """
        sc_idx = 0
        asm_adj_sc = []  # subchannels adjacent to each asm
        for asm in range(np.max(self.asm_map)):
            # pre-allocate array to fill with subchannel indices
            # this array is for the whole assembly
            adj_sc = np.zeros((6, self._sc_per_side + 1), dtype=int)
            for side in range(6):
                if self._need_to_count_side(asm, side):
                    adj_sc[side][0:self._sc_per_side] = \
                        np.arange(sc_idx + 1,
                                  sc_idx + 1 + self._sc_per_side)
                    sc_idx += self._sc_per_side  # update the sc index
                    # Check if you need to count the trailing corner
                    if self._need_to_count_corner(asm, side):
                        sc_idx += 1
                        adj_sc[side][-1] = sc_idx
                    # If you don't need to count a new corner sc, get
                    # the existing corner from the adjacent assembly
                    else:
                        adj_sc[side][-1] = \
                            self._find_corner_sc(asm, side, asm_adj_sc)
                else:
                    # get the subchannels that live here, including
                    # the trailing corner, which must already be
                    # defined if these side subchannels are defined.
                    adj_sc[side] = \
                        self._find_side_sc(asm, side, asm_adj_sc)
            # Add the assembly subchannel array to the bulk array
            asm_adj_sc.append(adj_sc)
        return np.array(asm_adj_sc)

    def _find_side_sc(self, asm, side, asm_adj_sc):
        r"""Find existing side subchannels

        Parameters
        ----------
        asm : int
            Active assembly ID (index)
        side : int
            Hexagon side; side 0 is the primary diagonal
        asm_adj_sc : list
            List of numpy.ndarray containing the indices for the
            interassembly gap subchannels that surround each assembly
            (sc_per_side+1 x 6)

        Returns
        -------
        numpy.ndarray
            Indices of existing side (and trailing corner) subchannels
            along the current side of the active assembly

        Notes
        -----
        We are walking clockwise around the active assembly hexagon.
        If this function is being called, the subchannels along this
        side have been defined for the adjacent assembly. We want to
        get these subchannels from the adjacent assembly.

        Because we are walking clockwise around the active assembly,
        we are walking counterclockwise along the faces of each of the
        adjacent assemblies. This means once we identify the which
        subchannels exist in the gap between these two assemblies, we
        need to:
        (1) Flip their order
        (2) Drop the corner subchannel (it was the trailing corner
            for the adjacent assembly but because the order is
            reversed it is the leading corner for the active assembly)
        (3) Get the trailing corner, which is the trailing corner from
            the "previous" side of the adjacent assembly.

        Graphic example
        ---------------
        Interassembly gap subchannels 1 - 5 have already been defined
        for assembly "Neighbor". As we walk clockwise around assembly
        "Active", we encounter these subchannels in the opposite
        direction.

            *                       *
                *               *
                    * __1__ *
                    * __2__ *
        Neighbor    * __3__ *   Active
                    * __4__ *
                    *   5   *
                *               *
            *                       *

        When this function is called, we will have already obtained
        corner subchannel 5 from marching up the preceding side of
        the active assembly; we will want to obtain side subchannels
        4, 3, and 2, as well as corner subchannel 1, which we know
        has been defined because the side channels have been defined.

        """
        neighbor = self.asm_adj[asm][side]
        neighbor_side = side - 3
        # if neighbor_side < 0:
        #     neighbor_side += 6
        neighbor_side_sc = asm_adj_sc[neighbor - 1][neighbor_side]
        neighbor_side_sc = neighbor_side_sc[:-1]  # drop corner
        neighbor_side_sc = neighbor_side_sc[::-1]  # flip direction
        # get last entry from neighbor asm previous side
        neighbor_side_sc = np.append(neighbor_side_sc,
                                     asm_adj_sc[neighbor - 1]
                                               [neighbor_side - 1][-1])
        return neighbor_side_sc

    def _find_corner_sc(self, asm, side, asm_adj_sc):
        r"""Find the (existing) corner subchannel that "ends" the
        current hexagon side.

        Parameters
        ----------
        asm : int
            Active assembly ID (index)
        side : int
            Hexagon side; side 0 is the primary diagonal
        asm_adj_sc : list
            List of numpy.ndarray containing the indices for the
            interassembly gap subchannels that surround each assembly
            (sc_per_side+1 x 6)

        Returns
        -------
        numpy.ndarray
            Indices of existing side (and trailing corner) subchannels
            along the current side of the active assembly

        Notes
        -----
        If this function is being called, it's because the subchannels
        along the current side have NOT yet been defined. This means
        that the neighboring assembly across the side subchannels has
        not had its interassembly gap subchannels defined. Therefore,
        we should not bother to look at it to determine the corner
        subchannel. Instead, we'll look at the "next" assembly,
        referred to here as "neighbor plus one".


        Graphic example
        ---------------
        Interassembly gap subchannels 1 - 4 are being defined for the
        active assembly. The subchannels for "Neighbor" have not yet
        been defined. The subchannels for "Neighbor +1" have been
        defined; we are seeking to determine corner subchannel "c".

            *      Neighbor +1      *
             /  *               *  \
            *  a /  *       *  \ e   *
                *  b /  *  \ d  *
                    * __c__ *
                    * __4__ *
        Neighbor    * __3__ *   Active
                    * __2__ *
                    *   1   *
                *               *
            *                       *

        """
        # loc = np.where(self.asm_map == asm + 1)
        # neighbor assembly across the next face (neighbor plus one)
        neighbor_p1 = self.asm_adj[asm][side - 5]
        # neighbor_p1_loc = (loc[0] + _dirs[self.hex_option][side - 5][0],
        #                    loc[1] + _dirs[self.hex_option][side - 5][1])
        # neighbor_p1_side = side - 2
        return asm_adj_sc[neighbor_p1 - 1][side - 2][-1]

    def _need_to_count_side(self, asm, side):
        """Determine whether an interassembly gap side channel needs
        to be counted or whether it has already been counted by the
        adjacent assembly"""
        neighbor = self.asm_adj[asm][side]
        if neighbor > 0:
            # Redefine neighbor according to the assembly map
            # If any of the indices in the neighbor location are
            # outside the assembly map (either less than zero or
            # greater than the assembly length), need to count the
            # side.
            loc = np.where(self.asm_map == asm + 1)
            neighbor_loc = (loc[0] + _dirs[self.hex_option][side][0],
                            loc[1] + _dirs[self.hex_option][side][1])
            if (not all(idx >= 0 and idx < len(self.asm_map)
                        for idx in neighbor_loc)):
                return True

            # Otherwise, the neighbor is within the bounds of the
            # assembly map array and can be defined.
            else:
                neighbor = self.asm_map[neighbor_loc]

                # If the neighbor ID is 0, the side must be defined.
                if neighbor == 0:
                    return True

                # If the neighbor ID is greater than that of the active
                # assembly, its gap subchannels have not been counted,
                # and therefore the side must be defined.
                elif asm + 1 < neighbor:
                    return True

                # If none of the above are true, then gap subchannels
                # along this side have been defined and we don't need
                # to define them.
                else:
                    return False

        # If the neighbor assembly in the neighbors matrix is defined
        # as 0, then we have to define gap subchannels along this side.
        else:
            return True

    def _need_to_count_corner(self, asm, side):
        """Determine whether an interassembly gap corner channel
        needs to be counted or whether it has already been counted
        by one of the adjacent assemblies.

        Notes
        -----
        If this method is being called, the gap subchannels on the
        active side needed to be defined. This means that the gap
        subchannels for the immediate neighbor have NOT been defined.
        Therefore, we need to look at the "neighbor plus one" assembly
        to determine whether we need to define a new gap corner
        subchannel.

        """
        # neighbor plus one
        neighbor_p1 = self.asm_adj[asm][side - 5]
        if neighbor_p1 > 0:
            # Redefine neighbor-plus-one according to the assembly
            # map. If any of the indices in the neighbor-plus-one
            # location are outside the assembly map (either less than
            # zero or greater than the assembly length), need to count
            # the side.
            loc = np.where(self.asm_map == asm + 1)
            loc = (loc[0] + _dirs[self.hex_option][side - 5][0],
                   loc[1] + _dirs[self.hex_option][side - 5][1])
            if (not all(idx >= 0 and idx < len(self.asm_map)
                        for idx in loc)):
                return True

            # Otherwise, the neighbor-plus-one is within the bounds
            # of the assembly map array and can be defined.
            else:
                neighbor_p1 = self.asm_map[loc]

                # If the neighbor-plus-one ID is 0, the side must be
                # defined.
                if neighbor_p1 == 0:
                    return True

                # If the neighbor-plus-one ID is greater than that of
                # the active assembly, its gap subchannels have not
                # been counted; therefore the side must be defined.
                elif asm + 1 < neighbor_p1:
                    return True

                # If none of the above are true, then gap subchannels
                # along this side have been defined and we don't need
                # to define them.
                else:
                    return False

        # If the neighbor-plus-one assembly has ID equal to 0, then the
        # corner gap subchannel must be defined.
        else:
            return True

    # FIND ADJACENT INTERASSEMBLY GAP SUBCHANNELS ----------------------

    def determine_sc_types(self):
        """Determine whether interassembly gap subchannels are edge
        or corner type based on where they occur in the asm_adj_sc
        array"""
        n_gap_sc = np.max(self.asm_sc_adj)
        sc_types = [1 for i in range(n_gap_sc)]
        for ai in range(len(self.asm_sc_adj)):
            for side in range(len(self.asm_sc_adj[ai])):
                corner_sc = self.asm_sc_adj[ai][side, -1]
                if not sc_types[corner_sc - 1] == 2:
                    sc_types[corner_sc - 1] = 2
        return sc_types

    def find_adjacent_sc(self):
        """Use the array mapping interassembly gap subchannels to
        adjacent assemblies to identify which subchannels are adjacent
        to each other"""
        n_gap_sc = np.max(self.asm_sc_adj)
        sc_adj = np.zeros((n_gap_sc, 3), dtype='int')

        for ai in range(len(self.asm_sc_adj)):
            asm_sc = self.asm_sc_adj[ai]
            for side in range(len(asm_sc)):
                for sci in range(len(asm_sc[side]) - 1):

                    # Look to trailing corner on previous side
                    if sci == 0:
                        sc = asm_sc[side, sci]
                        if asm_sc[side - 1, -1] not in sc_adj[sc - 1]:
                            idx_to_fill = np.where(sc_adj[sc - 1] == 0)[0][0]
                            sc_adj[sc - 1, idx_to_fill] = asm_sc[side - 1, -1]

                        sc = asm_sc[side - 1, -1]
                        if asm_sc[side, sci] not in sc_adj[sc - 1]:
                            idx_to_fill = np.where(sc_adj[sc - 1] == 0)[0][0]
                            sc_adj[sc - 1, idx_to_fill] = asm_sc[side, sci]

                    # For the sc in current index: map the sc in next index
                    sc = asm_sc[side, sci]
                    if asm_sc[side, sci + 1] not in sc_adj[sc - 1]:
                        idx_to_fill = np.where(sc_adj[sc - 1] == 0)[0][0]
                        sc_adj[sc - 1, idx_to_fill] = asm_sc[side, sci + 1]

                    # For the sc in next index; map the sc in current index
                    sc = asm_sc[side, sci + 1]
                    if asm_sc[side, sci] not in sc_adj[sc - 1]:
                        idx_to_fill = np.where(sc_adj[sc - 1] == 0)[0][0]
                        sc_adj[sc - 1, idx_to_fill] = asm_sc[side, sci]

        return sc_adj

    ####################################################################
    # TEMPERATURE PROPERTIES
    ####################################################################

    @property
    def avg_coolant_gap_temp(self):
        """Return the average temperature of the gap coolant
        subchannels at the last axial level"""
        tot = 0.0
        for i in range(len(self.coolant_gap_temp)):
            tot += (self.gap_params['area'][self.sc_types[i]]
                    * self.coolant_gap_temp[i])
        return tot / self.gap_params['total area']

    def adjacent_coolant_gap_temp(self, id):
        """Return the coolant temperatures in gap subchannels
        around a specific assembly at the last axial level

        Parameters
        ----------
        id : int
            Assembly ID

        Notes
        -----
        If necessary, approximate the temperatures from the interasm
        gap mesh to the assembly duct mesh.

        """
        return self.coolant_gap_temp[self.asm_sc_adj[id] - 1].flatten()

    ####################################################################
    # UPDATE GAP COOLANT PARAMS
    ####################################################################

    def _update_coolant_gap_params(self, temp):
        """Update correlated core inter-assembly gap coolant parameters
        based on current average coolant temperature

        Parameters
        ----------
        temp : list
            Average coolant temperature in inter-assembly gap

        """
        self.gap_coolant.update(temp)

        # Bypass velocity
        self.coolant_gap_params['vel'] = \
            (self.gap_flow_rate
             / self.gap_coolant.density
             / self.gap_params['total area'])

        # Bypass reynolds number
        self.coolant_gap_params['Re'] = \
            (self.gap_flow_rate
             * self.gap_params['total de']
             / self.gap_coolant.viscosity
             / self.gap_params['total area'])

        # Subchannel Reynolds numbers
        self.coolant_gap_params['Re_sc'] = \
            (self.gap_coolant.density * self.gap_params['de']
             * self.coolant_gap_params['vel']
             / self.gap_coolant.viscosity)

        # Heat transfer coefficient (via Nusselt number); Nu is equal
        # for all duct walls b/c they are the same material and coolant
        # properties are the same everywhere. The only difference is
        # surface area (accounted for in the temperature calculation).
        if self.model is None:
            self.coolant_gap_params['htc'] = 0.0
        else:
            nu = nusselt_db.calculate_sc_Nu(
                self.gap_coolant,
                self.coolant_gap_params['Re_sc'])
            self.coolant_gap_params['htc'] = \
                (self.gap_coolant.thermal_conductivity
                 * nu / self.gap_params['de'])

    ####################################################################
    # COOLANT TEMPERATURE CALCULATION
    ####################################################################

    def calculate_gap_temperatures(self, dz, asm_duct_temps):
        """Calculate the temperature of the inter-assembly coolant

        Parameters
        ----------
        dz : float
            Axial step size
        asm_duct_temps : list
            List of outer duct surface temperatures (K) for each
            assembly in the core (length must match gap meshing)

        Returns
        -------
        None

        """
        self._update_coolant_gap_params(self.avg_coolant_gap_temp)
        # asm_duct_temps = self.approximate_duct_temps(asm_duct_temps)
        # asm_duct_temps = np.array(asm_duct_temps)
        # print(asm_duct_temps.shape)

        # Update energy balance
        self._ebal['asm'] += \
            self._update_energy_balance(dz, asm_duct_temps)

        # Calculate new coolant gap temperatures
        if self.model == 'flow':
            dT = self._convection_model(dz, asm_duct_temps)
            self.coolant_gap_temp += dT

        elif self.model == 'no_flow':
            self.coolant_gap_temp = self._conduction_model(asm_duct_temps)

        elif self.model == 'duct_average':
            self.coolant_gap_temp = self._duct_avg_model(asm_duct_temps)

        else:  # self.model == None:
            # No change to coolant gap temp, do nothing
            pass

        # Update energy balance
        self._ebal['interasm'] += \
            self._update_energy_balance(dz, asm_duct_temps)

    def _update_energy_balance(self, dz, approx_duct_temps):
        """Track the energy added to the coolant from each duct wall
        mesh cell; summarize at the end for assembly-assembly energy
        balance"""
        # Convection constant
        C = (np.array([self.L[0][0], 2 * self.d['wcorner']])
             * dz
             * self.coolant_gap_params['htc'])
        C = C[self.sc_types[:self._sc_per_asm]]
        adj_cool = self.coolant_gap_temp[self.asm_sc_adj - 1]
        adj_cool = adj_cool.reshape(adj_cool.shape[0], -1)
        return C * (approx_duct_temps - adj_cool)

    def _convection_model(self, dz, approx_duct_temps):
        """Inter-assembly gap convection model

        Parameters
        ----------
        dz : float
            Axial mesh height
        approx_duct_temps : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature change in the inter-assembly gap coolant

        """
        dT = np.zeros(len(self.sc_adj))
        for sci in range(len(self.sc_adj)):
            dT_cond = 0.0
            dT_conv = 0.0
            type_i = self.sc_types[sci]

            # Convection to/from duct wall
            # identify adjacent assemblies to find duct wall temps
            asm, side, loc = np.where(self.asm_sc_adj == sci + 1)
            for i in range(len(asm)):
                duct_temp_idx = side[i] * (self._sc_per_side + 1) + loc[i]
                # adj_duct_temp = approx_duct_temps[asm[i], duct_temp_idx]
                adj_duct_temp = approx_duct_temps[asm[i]][duct_temp_idx]

                # if sci == 0:
                #     print(adj_duct_temp)
                dT[sci] += \
                    (self.coolant_gap_params['htc'][type_i]
                     * dz * self.ht_consts[type_i][2]
                     * (adj_duct_temp - self.coolant_gap_temp[sci])
                     / self.gap_coolant.heat_capacity)
                if sci == 1:
                    dT_conv += (self.coolant_gap_params['htc'][type_i]
                                * dz * self.ht_consts[type_i][2]
                                * (adj_duct_temp
                                   - self.coolant_gap_temp[sci])
                                / self.gap_coolant.heat_capacity)

            # Conduction to/from adjacent coolant subchannels
            for adj in self.sc_adj[sci]:
                if adj == 0:
                    continue
                sc_adj = adj - 1
                type_a = self.sc_types[sc_adj]

                dT[sci] += (self.gap_coolant.thermal_conductivity
                            * dz * self.ht_consts[type_i][type_a]
                            * (self.coolant_gap_temp[sc_adj]
                               - self.coolant_gap_temp[sci])
                            / self.gap_coolant.heat_capacity)
                if sci == 1:
                    dT_cond += (self.gap_coolant.thermal_conductivity
                                * dz * self.ht_consts[type_i][type_a]
                                * (self.coolant_gap_temp[sc_adj]
                                   - self.coolant_gap_temp[sci])
                                / self.gap_coolant.heat_capacity)

            # if sci == 1:
            #     print('{:.10e}'.format(dT_cond), '{:.10e}'.format(dT_conv))

        return dT

    def _conduction_model(self, approx_duct_temps):
        """Inter-assembly gap conduction model

        Parameters
        ----------
        approx_duct_temps : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature in the inter-assembly gap coolant

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant.

        The contact resistance between the bulk liquid and the duct
        wall is calculated using a heat transfer coefficient based on
        the actual velocity of the interassembly gap flow

        """
        # CONVECTION TO/FROM DUCT WALL
        # R_conv = np.array(
        #     [1 / (1 / self.L[0][0] / self.coolant_gap_params['htc'][0]
        #           + (self.d['gap'] / 2 / self.L[0][0]
        #              / self.gap_coolant.thermal_conductivity)),
        #      1 / (1 / (2 * self.d['wcorner'])
        #           / self.coolant_gap_params['htc'][1]
        #           + (self.d['gap'] / 2 / (2 * self.d['wcorner'])
        #              / self.gap_coolant.thermal_conductivity))])
        R_conv = np.array(
            [1 / (self.d['gap'] / 2 / self.L[0][0]
                  / self.gap_coolant.thermal_conductivity),
             1 / (self.d['gap'] / 2 / (2 * self.d['wcorner'])
                  / self.gap_coolant.thermal_conductivity)])
        # R_conv = np.array(
        #     [1 / (1 / self.L[0][0] / self.coolant_gap_params['htc'][0]),
        #      1 / (1 / (2 * self.d['wcorner'])
        #           / self.coolant_gap_params['htc'][1])])
        R_conv = R_conv[self.sc_types]

        # if not hasattr(self, '_conv_util'):
        #     self._make_conv_mask()  # creates lookup indices and masks

        # Lookup temperatures and mask as necessary
        T = approx_duct_temps[tuple(self._conv_util['inds'][0])]
        T += (approx_duct_temps[tuple(self._conv_util['inds'][1])]
              * self._conv_util['mask1'])
        T += (approx_duct_temps[tuple(self._conv_util['inds'][2])]
              * self._conv_util['mask2'])
        T *= R_conv  # Scale by the convection resistance

        # Get the total convection resistance, which will go in the
        # denominator at the end
        C_conv = copy.deepcopy(R_conv)
        C_conv += R_conv * self._conv_util['mask1']
        C_conv += R_conv * self._conv_util['mask2']

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        # Set up masks, if necessary
        # if not hasattr(self, '_Rcond'):
        #     self._make_cond_mask()

        # Now do some thangs c'mon
        R_cond = self._Rcond * self.gap_coolant.thermal_conductivity
        adj_cool_temp = self.coolant_gap_temp[self.sc_adj - 1] * R_cond
        C_cond = np.sum(R_cond, axis=1)  # will add to denom at the end

        # Round it out bb
        T += np.sum(adj_cool_temp, axis=1)
        return T / (C_cond + C_conv)

    def _duct_avg_model(self, approx_duct_temps):
        """Inter-assembly gap model that simply averages the adjacent
        duct wall surface temperatures

        Parameters
        ----------
        approx_duct_temps : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature in the inter-assembly gap coolant

        Notes
        -----
        Recommended for use when inter-assembly gap flow rate is so
        low that the the axial mesh requirement is intractably small.
        Assumes no thermal contact resistance between the duct wall
        and the coolant.

        The contact resistance between the bulk liquid and the duct
        wall is calculated using a heat transfer coefficient based on
        the actual velocity of the interassembly gap flow

        """
        # if not hasattr(self, '_conv_util'):
        #     self._make_conv_mask()  # creates lookup indices and masks

        # Lookup temperatures and mask as necessary
        T0 = approx_duct_temps[tuple(self._conv_util['inds'][0])]
        T1 = (approx_duct_temps[tuple(self._conv_util['inds'][1])]
              * self._conv_util['mask1'])
        T2 = (approx_duct_temps[tuple(self._conv_util['inds'][2])]
              * self._conv_util['mask2'])

        # Average nonzero values
        return (np.sum((T0, T1, T2), axis=0)
                / np.count_nonzero((T0, T1, T2), axis=0))

    def _make_conv_mask(self):
        """Create masks to quickly handle the subchannel-duct adjacency

        Notes
        -----
        Each gap coolant subchannel will touch either:
        - 1 duct (if it's at the edge of the problem);
        - 2 ducts (if an edge gap between two assemblies);
        - or 3 ducts (if it's a corner gap between 3 assemblies).

        We already have an array that links the subchannels to the
        adjacent assemblies, and from that, we can figure out which
        duct meshes each subchannel is in contact with. Because this
        is static throughout the problem, we can precompute this
        relationship and save it in the form of a mask: a set of 1s
        and 0s that indicate whether to keep or discard a value.

        We will have 3 masks, one for each possible adjacency that a
        subchannel can have. We'll naively grab duct temperatures that
        appear to match with the subchannel, but then we'll apply these
        masks to ensure that they're only added if the adjacency exists

        """
        self._conv_util = {}
        # Collect assembly, hex-side, and side-location indices for
        # each duct mesh; if no match, use -1 as a placeholder; this
        # is what we'll filter on later.
        a = [[], [], []]
        for sci in range(len(self.sc_adj)):
            asm, side, loc = np.where(self.asm_sc_adj == sci + 1)
            # calculate "duct index" based on hex-side and side-loc
            new = side * (self._sc_per_side + 1) + loc
            a[0].append((asm[0], new[0]))
            for i in range(2):
                try:
                    a[i + 1].append((asm[i + 1], new[i + 1]))
                except IndexError:
                    a[i + 1].append((-1, -1))

        # Now we're going to collect the indices where gap and duct
        # match up; these are where we will go pull out temperatures
        # from the incoming duct array. Note that because we used -1
        # as a placeholder and -1 is shorthand for the last value in
        # the array, we'll be pulling some bad values. Anywhere that
        # a = (-1, -1), the indices will also equal -1. This is what
        # the masks will handle.
        inds = []
        for i in range(3):
            inds.append(np.moveaxis(np.array(a[i]), -1, 0))
        self._conv_util['inds'] = inds

        # Now let's create the masks. Anywhere that self._inds = -1,
        # we will set the mask equal to 0 so that any values captured
        # by that index are eliminated. There are only two masks bc
        # the first temperature returned is always valid (since there
        # is always at least one duct-gap connection)
        self._conv_util['mask1'] = self._conv_util['inds'][1][0] >= 0
        self._conv_util['mask2'] = self._conv_util['inds'][2][0] >= 0

    def _make_cond_mask(self):
        """Just like for the convection lookup, make a mask that can
        accelerate the adjacent subchannel conduction lookup

        Notes
        -----
        This doubles as a shortcut to storing the constant parts of
        the conduction resistance

        """
        # as stored, min(self.sc_adj) is 1. This is not useful for
        # Python indexing, so we'll subtract 1 to make the values
        # useful as Python indices. There are 3 subchannel adjacencies
        # possible, but most subchannels only have 2. These are stored
        # as 0s but will become -1s after the subtraction. We need a
        # mask to filter them out.
        tmp = self.sc_adj - 1  # N_subchannel x 3 (max 3 adj possible)
        adj_types = self.sc_types[tmp]  # Adjacent types!
        old_L = np.array(self.L)  # old L
        new_L = np.zeros((len(self.sc_types), 3))  # what L will become
        # For each gap subchannel i
        # type_i = self.sc_types[i]
        # type_a = adj_types[i] (up to 3)
        for i in range(3):
            new_L[:, i] = old_L[self.sc_types, adj_types[:, i]]
        self._Rcond = self.d['gap'] / new_L
        # Since new_L had some bad values due to the -1s in tmp, we
        # will now filter them out of our array before we store it
        tmp = tmp >= 0
        self._Rcond = self._Rcond * tmp

    ####################################################################
    # MAP ASSEMBLY XY COORDINATES
    ####################################################################

    def map_assembly_xy(self):
        """Determine the X-Y positions of the assemblies in the core.

        Create a vector containing the x,y positions of the assemblies
        in the hexagonal lattice.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array (N_asm x 2) containing the assembly center X-Y coords

        """
        xy = np.zeros((np.max(self.asm_map), 2))
        normals = [4 * np.pi / 3, np.pi, 2 * np.pi / 3,
                   np.pi / 3, 0.0, 5 * np.pi / 3]
        # loc = (0.0, 0.0)
        # Directions turning clockwise around a hexagon
        # First entry is step from an inner ring to the top of an outer ring
        # The remaining steps are the turns around the hexagon corners
        _turns = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)]
        idx = 1
        loc = (0.0, 0.0)
        for ring in range(2, int(self.n_ring + 1)):
            d = 0
            row = self.n_ring - ring
            col = self.n_ring - ring
            # first step always to the right
            loc = (self.asm_pitch * (ring - 1), 0.0)
            if self.asm_map[row, col] != 0:
                xy[idx] = loc
                idx += 1
            # next steps walk around the ring
            col += 1  # already did first step
            positions = 6 * (ring - 1)  # all positions on active ring
            corners = np.arange(0, positions, ring - 1)
            for pos in range(1, int(positions)):
                loc = (loc[0] + self.asm_pitch * np.cos(normals[d]),
                       loc[1] + self.asm_pitch * np.sin(normals[d]))
                # The active position may be 0 in reg_assignments,
                # meaning that there's no region there. In that case,
                # skip; otherwise, fill empty map entry.
                if self.asm_map[row, col] != 0:
                    xy[idx] = loc
                    idx += 1
                # change directions if you reach a corner
                if pos > 0 and pos in corners:
                    d += 1
                # update row and column for next position
                row, col = row + _turns[d][0], col + _turns[d][1]

        return xy


########################################################################
# AXIAL CONSTRAINT
########################################################################


def calculate_min_dz(core_obj, temp_lo, temp_hi):
    """Evaluate dz for the inter-assembly gap at inlet and outlet
    temperatures; minimum value is taken to be constraining

    Parameters
    ----------
    core_obj : DASSH Core object
        Contains inter-assembly gap geometry specifications
    temp_lo : float
        Assembly inlet temperature (K)
    temp_hi : float
        Assembly outlet temperature (K; specified or estimated)

    Returns
    -------
    float
        Minimum required dz for stability at any temperature

    """
    if core_obj.model != 'flow':
        return None, None

    dz = []
    sc_code = []

    # Hold the original value of the temperature to reset after
    _temp_in = core_obj.gap_coolant.temperature

    # Calculate subchannel mass flow rates
    sc_mfr = [core_obj.gap_flow_rate
              * core_obj.gap_params['area'][i]
              / core_obj.gap_params['total area']
              for i in range(len(core_obj.gap_params['area']))]

    for temp in [temp_lo, temp_hi]:
        core_obj._update_coolant_gap_params(temp)
        # Only corner -> corner interassembly gap subchannels
        # NOTE: this means that the user ONLY specified 1-pin assemblies
        # lol, how ridiculous is that?
        if core_obj._sc_per_asm == 6:
            return (_cons9_999(
                sc_mfr[1],
                core_obj.L[1][1],
                core_obj.d['gap'],
                core_obj.d['wcorner'],
                core_obj.gap_coolant.thermal_conductivity,
                core_obj.gap_coolant.heat_capacity,
                core_obj.coolant_gap_params['htc'][1]), '9-999')

        else:
            sc_code.append('9-888')
            dz.append(
                _cons9_888(sc_mfr[1],
                           core_obj.L[0][1],
                           core_obj.d['gap'],
                           core_obj.d['wcorner'],
                           core_obj.gap_coolant.thermal_conductivity,
                           core_obj.gap_coolant.heat_capacity,
                           core_obj.coolant_gap_params['htc'][1]))

            # Edge subchannel --> corner/corner subchannel
            if core_obj._sc_per_asm == 12:
                sc_code.append('8-99')
                dz.append(
                    _cons8_99(sc_mfr[0],
                              core_obj.L[0][0],
                              core_obj.L[0][1],
                              core_obj.d['gap'],
                              core_obj.gap_coolant.thermal_conductivity,
                              core_obj.gap_coolant.heat_capacity,
                              core_obj.coolant_gap_params['htc'][0]))

            # Edge subchannel --> edge/corner subchannel
            else:
                sc_code.append('8-89')
                dz.append(
                    _cons8_89(sc_mfr[0],
                              core_obj.L[0][0],
                              core_obj.L[0][1],
                              core_obj.d['gap'],
                              core_obj.gap_coolant.thermal_conductivity,
                              core_obj.gap_coolant.heat_capacity,
                              core_obj.coolant_gap_params['htc'][0]))

            # Edge subchannel --> edge/edge subchannel
            if core_obj._sc_per_asm >= 24:
                sc_code.append('8-88')
                dz.append(
                    _cons8_88(sc_mfr[0],
                              core_obj.L[0][0],
                              core_obj.d['gap'],
                              core_obj.gap_coolant.thermal_conductivity,
                              core_obj.gap_coolant.heat_capacity,
                              core_obj.coolant_gap_params['htc'][0]))

    # Reset the coolant temperature
    core_obj._update_coolant_gap_params(_temp_in)
    # print(min_dz)
    min_dz = np.min(dz)
    return min_dz, sc_code[dz.index(min_dz)]


def _cons8_88(m8, L88, d_gap, k, Cp, h_gap):
    """dz constrant for edge gap sc touching 2 edge gap sc"""
    term1 = 2 * h_gap * L88 / m8 / Cp       # conv to inner/outer ducts
    term2 = 2 * k * d_gap / m8 / Cp / L88   # cond to adj bypass edge
    return 1 / (term1 + term2)


def _cons8_89(m8, L88, L89, d_gap, k, Cp, h_gap):
    """dz constrant for edge gap sc touching edge, corner gap sc"""
    term1 = 2 * h_gap * L88 / m8 / Cp   # conv to inner/outer ducts
    term2 = k * d_gap / m8 / Cp / L88   # cond to adj bypass edge
    term3 = k * d_gap / m8 / Cp / L89   # cond to adj bypass corner
    return 1 / (term1 + term2 + term3)


def _cons8_99(m8, L88, L89, d_gap, k, Cp, h_gap):
    """dz constrant for edge gap sc touching 2 corner gap sc"""
    term1 = 2 * h_gap * L88 / m8 / Cp   # conv to inner/outer ducts
    term2 = 2 * k * d_gap / m8 / Cp / L89   # cond to adj bypass corner
    return 1 / (term1 + term2)


def _cons9_888(m9, L89, d_gap, wc, k, Cp, h_gap):
    """Interasm gap corner connected to three interasm gap edge"""
    term1 = 6 * wc * h_gap / m9 / Cp        # conv in/out duct
    term2 = 3 * k * d_gap / m9 / Cp / L89   # cond to adj interasm edge
    return 1 / (term1 + term2)


def _cons9_999(m9, L99, d_gap, wc, k, Cp, h_gap):
    """Interasm gap corner connected to three interasm gap corners"""
    term1 = 6 * wc * h_gap / m9 / Cp        # conv in/out duct
    term2 = 3 * k * d_gap / m9 / Cp / L99   # cond adj interasm corner
    return 1 / (term1 + term2)
