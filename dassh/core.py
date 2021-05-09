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
date: 2021-05-03
author: matz
Methods to describe the layout of assemblies in the reactor core and
the coolant in the gap between them
"""
########################################################################
import numpy as np
import copy
from dassh.logged_class import LoggedClass
from dassh.correlations import nusselt_db


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

    def __init__(self, asm_list_input, asm_pitch, gap_flow_rate,
                 coolant_obj, inlet_temperature=273.15, model='flow',
                 test=False):
        """Instantiate Core object."""
        LoggedClass.__init__(self, 4, 'dassh.core.Core')
        if model not in ['flow', 'no_flow', 'duct_average', None]:
            msg = 'Do not understand input inter-assembly gap model: '
            self.log('error', msg + model)

        # --------------------------------------------------------------
        # Identify GEODST periodicity: if not full core, it can be
        # either 60 or 120 degree
        self.n_ring = count_rings(len(asm_list_input))
        self.n_asm = np.sum(~np.isnan(asm_list_input))
        self.hex_option = 0  # full core only, no periodicity
        self.asm_pitch = asm_pitch  # m
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
        self.asm_map = map_asm(asm_list_input)

        # Map neighbors for each assembly based on problem symmetry
        # Again, only one map needed for all axial meshes
        # Sets attribute self.asm_adj
        self.asm_adj = map_adjacent_assemblies(self.asm_map)

    ####################################################################
    # SETUP METHODS
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
        assert len(asms) == self.n_asm
        self.duct_oftf = asms[0].duct_oftf  # Duct outer FTF
        self.hex_side_len = self.duct_oftf / _sqrt3
        # Set up inter-assembly gap subchannel sizes and dimensions
        self.d_gap = self.asm_pitch - self.duct_oftf  # Gap "width"

        # Set up interassembly gap subchannel attributes

        # (1) Geometric parameters of assembly with which subchannels
        # are aligned: pin pitch, corner perimeter, side sc per hex side
        self._geom_params = self._collect_sc_geom_params(asms)

        # (2) assembly-subchannel adjacency - return as nested lists
        # for later use; store as numpy array
        asm_adj_sc = self._map_asm_gap_adjacency()
        # Combine nested lists: a little ugly bc not all same length
        max_scpa = max([sum([len(x) for x in y]) for y in asm_adj_sc])
        self._asm_sc_adj = np.zeros((self.n_asm, max_scpa), dtype=int)
        for a in range(self.n_asm):
            tmp = np.array([x for l in asm_adj_sc[a] for x in l])
            tmp = tmp.flatten()
            self._asm_sc_adj[a, :tmp.shape[0]] = tmp

        # (3) boundaries betwee assembly-adjacent subchannels
        self._asm_sc_xbnds = self._calculate_gap_xbnds()

        # (4) types of assembly-adjacent subchannels
        # (5) global array of subchannel types
        tmp = self._determine_gap_sc_types()
        self._asm_sc_types = [np.array(
            [li for l in x for li in l]).flatten() for x in tmp[0]]
        self._sc_types = np.array(tmp[1])

        # Calculate some follow-up parameters:
        # Number of subchannels
        self.n_sc = np.max(self._asm_sc_adj)

        # Number of subchannels adjacent to each assembly
        self._n_sc_per_asm = np.count_nonzero(self._asm_sc_adj, axis=1)

        # Global subchannel-subchannel adjacency
        self._sc_adj = self._find_adjacent_sc(asm_adj_sc)
        # Subchannel area, hydraulic diameter, distances to neighbors
        self.gap_params = {}
        self.gap_params['wp'] = self._calculate_sc_wp()
        self.gap_params['asm wp'] = self._calculate_asm_sc_wp()
        self.gap_params['area'] = self._calculate_sc_area()
        self.gap_params['L'] = self._calculate_dist_between_sc()
        self.gap_params['de'] = self._calculate_sc_de()

        # Core-total parameters
        self.gap_params['total area'] = np.sum(self.gap_params['area'])
        self.gap_params['total wp'] = 6 * self.hex_side_len * self.n_asm
        self.gap_params['total de'] = (4 * self.gap_params['total area']
                                       / self.gap_params['total wp'])

        # Fractional area
        self.gap_params['area frac'] = (self.gap_params['area']
                                        / self.gap_params['total area'])

        # Flow parameters
        self._sc_mfr = self.gap_flow_rate * self.gap_params['area frac']
        if self.model == 'flow':
            self._inv_sc_mfr = 1 / self._sc_mfr

        # Reynolds number constant
        self.coolant_gap_params['_Re_sc'] = \
            self._sc_mfr * self.gap_params['de'] / self.gap_params['area']

        # Interior coolant temperatures; shape = n_axial_mesh x n_sc
        self.coolant_gap_temp = np.ones(self.n_sc)
        self.coolant_gap_temp *= self.gap_coolant.temperature

        # Set up convection/conduction utility attributes
        self._make_conv_mask()
        self._make_cond_mask()

        # Update coolant gap params based on inlet temperature
        self._update_coolant_gap_params(self.gap_coolant.temperature)

        # Track the energy given from the assemblies to the gap
        # coolant; this should match roughly with what the assembly
        # energy balance reports is lost through the outermost duct.
        # The match won't be exact because the assembly calculation
        # assumes heat from the corner subchannels is transferred
        # through an area equal to that at the midplane of the duct,
        # whereas here it is transferred through the area at the
        # outside of the duct, since that is the same for all
        # assemblies. I have confirmed that adjusting for this gives
        # the matching result. This array will not sum to zero.
        # self.ebal = np.zeros(())
        self.ebal = {}
        self.ebal['asm'] = np.zeros(self._asm_sc_adj.shape)

    # MAP INTER-ASSEMBLY GAP; DEFINE GEOMETRY --------------------------

    def _collect_sc_geom_params(self, asm_list):
        """Save attributes from assemblies that define gap subchannel
        geometry

        Parameters
        ----------
        asm_list : list
            List of DASSH Assembly objects

        Returns
        -------
        dict
            Dict contains two items:
            - 'dims': numpy.ndarray, N_asm x 6 x 2
                Pin pitch and corner perimter for gap subchannels on
                each hex side of each assembly
            - 'sc_per_side': numpy.ndarray, N_asm x 6
                Number of gap edge subchannels on each hex side of
                each assembly

        """
        dims = np.zeros((self.n_asm, 6, 2))
        edge_sc_per_side = np.zeros((self.n_asm, 6), dtype=int)
        for asm in range(self.n_asm):
            for side in range(6):
                # Figure out from which assembly to get mesh params
                if self.asm_adj[asm][side] - 1 >= 0:
                    adj = asm_list[self.asm_adj[asm][side] - 1]
                    asm_with_mesh_params, sc_per_side = \
                        _which_asm_has_finer_mesh(asm_list[asm], adj)
                else:
                    asm_with_mesh_params, sc_per_side = \
                        _which_asm_has_finer_mesh(asm_list[asm], None)

                # Save the defining pin-pitch and corner perimeter
                pin_pitch = 0.0
                if asm_with_mesh_params.has_rodded:
                    pin_pitch = asm_with_mesh_params.rodded.pin_pitch
                    dwc = asm_with_mesh_params.rodded.d['wcorner'][-1, -1]
                else:
                    pin_pitch = 0.0
                    dwc = 0.5 * asm_with_mesh_params.duct_oftf / _sqrt3
                dims[asm, side] = [pin_pitch, dwc]
                # Save the defining edge_sc_per_side
                edge_sc_per_side[asm, side] = sc_per_side
        return {'dims': dims, 'sc_per_side': edge_sc_per_side}

    def _map_asm_gap_adjacency(self):
        """Map the interassembly gap subchannels that surround each
        assembly in the core.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Nested list-of-lists containing assembly-subchannel
            adjacency per-hex-side

        """
        sc_idx = 0            # running subchannel index
        asm_adj_sc = []       # subchannels adjacent to each asm
        for asm in range(self.n_asm):
            # pre-allocate temp arrays to fill with values when counting
            tmp_asm_adj_sc = []
            for side in range(6):

                # Index the subchannels along this hex side
                tmp, sc_idx = self._index_gap_sc(
                    asm,
                    side,
                    sc_idx,
                    self._geom_params['sc_per_side'][asm][side],
                    asm_adj_sc)
                tmp_asm_adj_sc.append(tmp)

            # Add the temporary arrays to main arrays
            asm_adj_sc.append(tmp_asm_adj_sc)

        return asm_adj_sc

    def _calculate_gap_xbnds(self):
        """Calculate the boundaries between the subchannels along each
        assembly hex side

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Boundaries for the interassembly gap subchannels
            that surround each assembly (sc_per_side+1 x 6)

        """
        asm_sc_xbnds = []  # 1D (along hex side) coords of gap SC bnds
        for asm in range(self.n_asm):
            tmp_sc_xbnds = []
            for side in range(6):
                # Figure out from which assembly to get mesh params
                # if self.asm_adj[asm][side] - 1 > 0:
                #     adj = asm_list[self.asm_adj[asm][side] - 1]
                #     asm_with_mesh_params, sc_per_side = \
                #         _which_asm_has_finer_mesh(asm_list[asm], adj)
                # else:
                #     asm_with_mesh_params, sc_per_side = \
                #         _which_asm_has_finer_mesh(asm_list[asm], None)
                #
                # Get the boundaries of the subchannels
                hex_side_len = self.duct_oftf / _sqrt3
                starting_x = hex_side_len * side
                # Edge length = pin pitch
                # Corner "half-perimeter"
                pp, dwc = self._geom_params['dims'][asm, side]
                # Add corner-edge boundary based on starting point
                tmp_sc_xbnds.append(starting_x + dwc)
                # Add subsequent edge boundaries
                sc_per_side = self._geom_params['sc_per_side'][asm, side]
                for sci in range(sc_per_side):
                    tmp_sc_xbnds.append(tmp_sc_xbnds[-1] + pp)

            # Add the temporary arrays to main arrays
            asm_sc_xbnds.append(tmp_sc_xbnds)

        # Convert to numpy array and return
        # max_scpa = max([len(x) for x in asm_sc_xbnds])
        # _asm_sc_xbnds = np.zeros((self.n_asm, max_scpa))
        _asm_sc_xbnds = np.zeros(self._asm_sc_adj.shape)
        for a in range(self.n_asm):
            tmp = np.array(asm_sc_xbnds[a])
            _asm_sc_xbnds[a, :tmp.shape[0]] = tmp
        return _asm_sc_xbnds

    def _calculate_gap_xpts(self, asm_list):
        """Determine the center point of each gap/duct subchannel
        connection for each assembly

        Parameters
        ----------
        asm_list : list
            List of DASSH Assembly objects

        Notes
        -----
        Currently not used. May refine and active in the future

        """
        asm_sc_xpts = []  # 1D (along hex side) coords of gap SC
        for a in range(self.n_asm):
            tmp_sc_xpts = []
            for side in range(6):
                # Figure out from which assembly to get mesh params
                if self.asm_adj[a][side] - 1 > 0:
                    adj = asm_list[self.asm_adj[a][side] - 1]
                    asm, scps = _which_asm_has_finer_mesh(asm_list[a], adj)
                else:
                    asm, scps = _which_asm_has_finer_mesh(asm_list[a], adj)

                xpts = [r.x_pts for r in asm.region]
                len_of_xpts = [len(x) for x in xpts]
                max_len = max(len_of_xpts)
                matching_xpts = [x for x in xpts if len(x) == max_len]
                tmp_sc_xpts.append(matching_xpts[0])
            # Add the temporary arrays to main arrays
            asm_sc_xpts.append(tmp_sc_xpts)
        return asm_sc_xpts

    def _determine_gap_sc_types(self):
        """Determine the gap subchannel types around each assembly
        and globally

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            Tuple of two lists containing (1) the subchannel types
            relative to each loaded assembly and (2) relative to the
            global subchannel indexing, respectively

        """
        sc_types = []         # Global (1D) gap SC types
        asm_sc_types = []     # Assembly adjacent gap SC types
        for asm in range(self.n_asm):
            # pre-allocate temp arrays to fill with values when counting
            tmp_asm_sc_types = []
            for side in range(6):
                to_add = []
                scps = self._geom_params['sc_per_side'][asm][side]
                # If newly indexed subchannels: count new types
                if self._need_to_count_side(asm, side):
                    # Newly counted edge subchannels along that side
                    to_add = [0 for i in range(scps)]
                    # Check if you need to count the trailing corner
                    if self._need_to_count_corner(asm, side):
                        to_add.append(1)
                sc_types += to_add

                # Add assembly-adjacent subchannel types to asm list
                tmp_asm_sc_types.append([0 for i in range(scps)])
                tmp_asm_sc_types[-1].append(1)

            # Add the temporary arrays to main arrays
            asm_sc_types.append(tmp_asm_sc_types)
        return asm_sc_types, sc_types

    def _calculate_dist_between_sc_OLD(self):
        """Calculate distance between subchannel centroids

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array (N_sc x 3) of distances to adjacent subchannels

        Notes
        -----
        Carried out in three loops:
        1. Determine distances per assembly, per hex side
        2. Combine hex sides --> distances per assembly
        3. Globalize --> distances between all subchannels

        """
        asm_sc_dist = []
        for asm in range(self.n_asm):
            tmp_asm_sc_dist = []
            for side in range(6):
                # Figure out from which assembly to get mesh params
                # if self.asm_adj[asm][side] - 1 > 0:
                #     adj = asm_list[self.asm_adj[asm][side] - 1]
                #     asm_with_mesh_params, scps = \
                #         _which_asm_has_finer_mesh(asm_list[asm], adj)
                # else:
                #     asm_with_mesh_params, scps = \
                #         _which_asm_has_finer_mesh(asm_list[asm], None)
                scps = self._geom_params['sc_per_side'][asm, side]
                pp = self._geom_params['dims'][asm, side, 0]
                # Get the distances between subchannels
                gap_side_len = self.duct_oftf / _sqrt3
                # (add lil extra beyond duct surface corner to reach
                # center of corner channel along middle of gap)
                gap_side_len += _sqrt3 * self.d_gap / 3
                # print(gap_side_len, self.duct_oftf, self.d_gap)
                L = np.zeros((scps + 1, 3))
                if scps >= 1:  # At least one edge SC
                    # Will for sure have edge-corner connection
                    L11 = pp
                    L12 = 0.5 * (gap_side_len - scps * L11) + 0.5 * L11
                    L[0, 0] = L12  # First edge, look back: edge-corner
                    L[-2, 1] = L12  # Last edge, look fwd: edge-corner
                    L[-1, 0] = L12  # Last corner, look back: edge-corner
                    if scps > 1:  # More than one edge SC: L11 too
                        L[1:-1, 0] = L11  # other edge look back: edge-edge
                        L[:-2, 1] = L11  # other edge SC look fwd: edge-edge
                else:  # Corner-corner connection
                    L[0, 2] = gap_side_len
                # print(asm, side, L)
                tmp_asm_sc_dist.append(L)

            # Add the temporary arrays to main arrays: need second loop
            # because relies on result from first loop
            for side in range(6):
                # Need to fill in the trailing corner "looking forward" dist
                # Example: hex side 1 traiing corner does not have a value
                # filled in to look forward to the next side. Need to take
                # the first edge subchannel on hex side 2 "looking back"
                # distance and pass it to the hex side 1 trailing corner.
                L12_prev = tmp_asm_sc_dist[side][0, 0]
                tmp_asm_sc_dist[side - 1][-1, 1] = L12_prev
            tmp_asm_sc_dist = np.vstack(tmp_asm_sc_dist)
            asm_sc_dist.append(tmp_asm_sc_dist)

        # Now globalize - relies on previously assigned attributes
        L_global = np.zeros((self.n_sc, 3))
        # print('n_sc', self.n_sc)
        for a in range(self.n_asm):
            # print(a, asm_sc_dist[a].shape, self._asm_sc_adj[a].shape)
            for i in range(self._asm_sc_adj[a].shape[0]):
                sci = self._asm_sc_adj[a][i] - 1
                if sci < 0:
                    continue
                if np.all(L_global[sci] == 0):
                    L_global[sci] = asm_sc_dist[a][i]
                else:
                    if self._asm_sc_types[a][i] == 0:
                        continue
                    else:
                        # If all values filled, no need to replace any
                        if not np.any(L_global[sci] == 0):
                            continue
                        else:  # Fill in remaining corner value
                            s = np.count_nonzero(
                                self._asm_sc_types[a][:i])
                            L12 = (self._geom_params['dims'][a, s, 1]
                                   + _sqrt3 * self.d_gap / 6)
                            L12 += 0.5 * self._geom_params['dims'][a, s, 0]
                            L_global[sci, 2] = L12
                            # L_global[sci, 2] = asm_sc_dist[a][i, 1]
        return L_global

    def _calculate_dist_between_sc(self):
        """Calculate distance between subchannel centroids

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array (N_sc x 3) of distances to adjacent subchannels

        Notes
        -----
        Carried out in three loops:
        1. Determine distances per assembly, per hex side
        2. Combine hex sides --> distances per assembly
        3. Globalize --> distances between all subchannels

        """
        gap_side_len = self.duct_oftf / _sqrt3
        gap_side_len += _sqrt3 * self.d_gap / 3
        lil_bit = _sqrt3 * self.d_gap / 6
        L_global = np.zeros((self.n_sc, 3))
        for i in range(self.n_sc):
            # SKIP IF DONE: if all cols filled, don't need to write more
            filled_pos = np.count_nonzero(L_global[i])
            if filled_pos - self._sc_types[i] == 2:
                continue
            ip1 = i + 1
            asm, loc = np.where(self._asm_sc_adj == ip1)
            # Edge subchannels: adj asm share properties, so just use
            # those from the first in the lookup list
            if self._sc_types[i] == 0:
                side = np.count_nonzero(self._asm_sc_types[asm[0]][:loc[0]])
                pp, dwc = self._geom_params['dims'][asm[0], side]
                for j in range(3):
                    sc_adj = self._sc_adj[i, j] - 1
                    if sc_adj < 0:
                        continue
                    if self._sc_types[sc_adj] == 1:
                        L_global[i, j] = 0.5 * pp + dwc + lil_bit
                    else:
                        L_global[i, j] = pp
            # Corner subchannels: look to neighbors
            else:
                for j in range(3):
                    sc_adj = self._sc_adj[i, j] - 1
                    if sc_adj < 0:
                        continue
                    asm_adj, loc_adj = \
                        np.where(self._asm_sc_adj == self._sc_adj[i, j])
                    side_adj = np.count_nonzero(
                        self._asm_sc_types[asm_adj[0]][:loc_adj[0]])
                    pp, dwc = self._geom_params['dims'][asm_adj[0], side_adj]
                    if self._sc_types[sc_adj] == 1:
                        L_global[i, j] = gap_side_len
                    else:
                        L_global[i, j] = 0.5 * pp + dwc + lil_bit
        return L_global

    def _index_gap_sc(self, asm, side, sc_id, sc_per_side, already_idx):
        """Count gap subchannel indices along an assembly side

        Parameters
        ----------
        asm : int
            Active assembly index
        side : int
            Active hex side
        sc_id : int
            Active gap subchannel index
        sc_per_side : int
            Number of gap edge subchannels along this hex side
        already_idx : list
            List of lists containing the already-indexed adjacency
            between previous assemblies and gap subchannels

        Returns
        -------
        list
            Subchannel indices along the active hex side of the
            active assembly

        """
        if self._need_to_count_side(asm, side):
            # Count edge subchannels along that side
            to_add = list(np.arange(sc_id + 1, sc_id + 1 + sc_per_side))
            sc_id += sc_per_side  # update the sc index
            # Check if you need to count the trailing corner
            if self._need_to_count_corner(asm, side):
                sc_id += 1
                to_add.append(sc_id)
            # If you don't need to count a new corner sc, get
            # the existing corner from the adjacent assembly
            else:
                to_add.append(
                    self._find_corner_sc(
                        asm, side, already_idx))
        else:
            # get the subchannels that live here, including
            # the trailing corner, which must already be
            # defined if these side subchannels are defined.
            to_add = self._find_side_sc(asm, side, already_idx)
        return to_add, sc_id

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

    def _find_adjacent_sc(self, asm_sc_adj):
        """Use the array mapping interassembly gap subchannels to
        adjacent assemblies to identify which subchannels are adjacent
        to each other

        Parameters
        ----------
        asm_sc_adj : list
            Nested lists containing subchannel indices (base 1)
            adjacent to each assembly; size = N_asm x 6 x N_sc_on_side
            (note that "N_sc_on_side" can vary)

        Returns
        -------
        numpy.ndarray
            Array (N_gap_sc x 3) indicating adjacency between gap
            subchannels

        """
        sc_adj = np.zeros((self.n_sc, 3), dtype='int')
        for ai in range(len(asm_sc_adj)):
            asm_sc = asm_sc_adj[ai]
            for side in range(len(asm_sc)):
                for sci in range(len(asm_sc[side]) - 1):
                    # Look to trailing corner on previous side
                    if sci == 0:
                        # Fill trailing corner's value into active index
                        sc = asm_sc[side][sci]
                        if asm_sc[side - 1][-1] not in sc_adj[sc - 1]:
                            idx = np.where(sc_adj[sc - 1] == 0)[0][0]
                            sc_adj[sc - 1, idx] = asm_sc[side - 1][-1]
                        # Fill active index into trailing corner
                        sc = asm_sc[side - 1][-1]
                        if asm_sc[side][sci] not in sc_adj[sc - 1]:
                            # ADDED 2021-04-22
                            # Wanted to make side 0 trailing corner
                            # adjacency order of the rest of the SC
                            if side == 0:
                                idx = 1
                            else:
                                idx = np.where(sc_adj[sc - 1] == 0)[0][0]
                            sc_adj[sc - 1, idx] = asm_sc[side][sci]
                    # For the sc in current index: map the sc in next index
                    sc = asm_sc[side][sci]
                    if asm_sc[side][sci + 1] not in sc_adj[sc - 1]:
                        idx = np.where(sc_adj[sc - 1] == 0)[0][0]
                        sc_adj[sc - 1, idx] = asm_sc[side][sci + 1]
                    # For the sc in next index; map the sc in current index
                    sc = asm_sc[side][sci + 1]
                    if asm_sc[side][sci] not in sc_adj[sc - 1]:
                        idx = np.where(sc_adj[sc - 1] == 0)[0][0]
                        sc_adj[sc - 1, idx] = asm_sc[side][sci]
        return sc_adj

    def _calculate_sc_wp(self):
        """Calculate wetted perimeter of each gap subchannel

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Wetted perimeter (m) of each subchannel (N_sc x 1)

        """
        # Loop over all assemblies to calculate WP of adjacent SC
        wp = np.zeros(self.n_sc)
        hex_perim = self.duct_oftf * 6 / np.sqrt(3)
        for a in range(self.n_asm):
            xtmp = self._asm_sc_xbnds[a]
            xtmp = xtmp[self._asm_sc_adj[a] > 0]
            for i in range(len(xtmp) - 1):
                sci = self._asm_sc_adj[a][i]  # <-- remember, base-1 idx
                if sci < 1:
                    continue
                # Just duct wetted perimeter; mult by width later
                wp[sci - 1] += xtmp[i + 1] - xtmp[i]
            # WP of the last one needs to wrap around to the first
            sci = self._asm_sc_adj[a][i + 1]
            wp[sci - 1] += hex_perim - xtmp[-1] + xtmp[0]

        # Corrections for outermost subchannels.
        # Edge subchannels need WP0 x 2
        for i in range(self.n_sc):
            asm, loc = np.where(self._asm_sc_adj == i + 1)
            if self._sc_types[i] == 0:
                if len(asm) == 1:  # <-- this means it's an outer SC
                    wp[i] *= 2  # haven't counted "non-asm" wall
            else:  # Treat corners adjacent one or two assemblies
                if len(asm) == 1:
                    wp[i] *= 2
                    wp[i] += 2 * self.d_gap / _sqrt3
                elif len(asm) == 2:
                    x = np.zeros((2, 2))
                    for a in range(2):
                        scps = self._geom_params['sc_per_side'][asm[a]]
                        tmp = np.cumsum(scps)
                        tmp += np.arange(0, 6, 1)
                        s1 = np.where(tmp == loc[a])[0][0]
                        if s1 == 5:
                            s2 = 0
                        else:
                            s2 = s1 + 1
                        x[a, 0] = self._geom_params['dims'][asm[a]][s1][1]
                        x[a, 1] = self._geom_params['dims'][asm[a]][s2][1]
                    # Choose the nonshared ones
                    dwc = np.zeros(2)
                    for a in range(2):
                        if x[a, 0] in x[a - 1]:
                            dwc[a] = x[a, 1]
                        else:
                            dwc[a] = x[a, 0]
                    wp[i] += dwc[0] + dwc[1]
                else:
                    continue
        return wp

    def _calculate_asm_sc_wp(self):
        """Calculate wetted perimeter of each gap subchannel relative
        to its adjacent assembly

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Wetted perimeter (m) of each subchannel (N_asm x N_scpa)

        """
        # Loop over all assemblies to calculate WP of adjacent SC
        wp = np.zeros((self._asm_sc_xbnds.shape))
        hex_perim = self.duct_oftf * 6 / np.sqrt(3)
        for a in range(self.n_asm):
            xtmp = self._asm_sc_xbnds[a]
            xtmp = xtmp[self._asm_sc_adj[a] > 0]
            for i in range(len(xtmp) - 1):
                if self._asm_sc_adj[a][i] < 1:  # <-- remember, base-1 idx
                    continue
                # Just duct wetted perimeter; mult by width later
                wp[a, i] = xtmp[i + 1] - xtmp[i]
            # WP of the last one needs to wrap around to the first
            wp[a, i + 1] += hex_perim - xtmp[-1] + xtmp[0]
        return wp

    def _calculate_sc_area(self):
        """Calculate the flow area of each gap subchannel

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Area (m^2) of each gap subchannel (N_sc x 1)

        """
        # Already have WP, which includes all adjacency. If adjacent
        # to 2 or 3 neighbors, need: WP / 2 as the "length" of the SC
        # otherwise: use WP
        # Width is the gap width
        # NOTE: for corners, need to add that lil center triangle if
        # next to a neighbor; if alone, add different thing
        corner_neighbor = self.d_gap**2 * _sqrt3 / 4
        corner_no_neighbor = self.d_gap**2 * _sqrt3 / 3
        area = self.gap_params['wp'] * self.d_gap
        for i in range(self.n_sc):
            asm, loc = np.where(self._asm_sc_adj == i + 1)
            if self._sc_types[i] == 0:
                area[i] *= 0.5
            else:
                if len(asm) == 1:
                    area[i] = self.gap_params['asm wp'][asm[0], loc[0]]
                    area[i] *= self.d_gap
                    area[i] += corner_no_neighbor
                else:
                    area[i] *= 0.5
                    area[i] += corner_neighbor
        return area

    def _calculate_sc_de(self):
        """Calculate hydraulic diameter of each gap subchannel

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Hydraulic diameter (m) of each subchannel (N_sc x 1)

        """
        # De = 4A / WP; already calculated area, WP
        return 4 * self.gap_params['area'] / self.gap_params['wp']

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
        # Collect convection constants in array: need "wetted perimeter"
        # of subchannel connection with each adjacent assembly (up to 3)
        self._conv_util['const'] = np.zeros((self.n_sc, 3))
        for sci in range(self.n_sc):
            asm, loc = np.where(self._asm_sc_adj == sci + 1)
#            asm = [ai for ai in range(len(self._asm_sc_adj))
#                   if sci + 1 in self._asm_sc_adj[ai]]
#            loc = [np.where(self._asm_sc_adj[ai] == sci + 1)[0][0]
#                   for ai in asm]
            # calculate "duct index" based on hex-side and side-loc
            # new = side * (self._sc_per_side + 1) + loc
            # Connection 0: always at least one gap-duct connection
            a[0].append((asm[0], loc[0]))
            # Try the remaining two possible connections
            for i in range(2):
                try:
                    a[i + 1].append((asm[i + 1], loc[i + 1]))
                except IndexError:
                    a[i + 1].append((-1, -1))

            # Calculate convection constant based on sc-duct connections
            for i in range(len(asm)):
                self._conv_util['const'][sci, i] = \
                    self.gap_params['asm wp'][asm[i], loc[i]]

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

        # Poop
        if self.model == 'no_flow':
            self._conv_util['const'] = \
                2 * self._conv_util['const'] / self.d_gap

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
        # tmp = self._sc_adj - 1  # N_subchannel x 3 (max 3 adj possible)
        # adj_types = self.sc_types[tmp]  # Adjacent types!
        # old_L = np.array(self.L)  # old L
        # new_L = np.zeros((len(self.sc_types), 3))  # what L will become
        # For each gap subchannel i
        # type_i = self.sc_types[i]
        # type_a = adj_types[i] (up to 3)
        # for i in range(3):
        #     new_L[:, i] = old_L[self.sc_types, adj_types[:, i]]
        self._Rcond = np.divide(self.d_gap, self.gap_params['L'],
                                out=np.zeros_like(self.gap_params['L']),
                                where=(self.gap_params['L'] != 0))
        # Since new_L had some bad values due to the -1s in tmp, we
        # will now filter them out of our array before we store it
        # tmp = tmp >= 0
        # self._Rcond = self._Rcond * tmp

    ####################################################################
    # TEMPERATURE PROPERTIES
    ####################################################################

    @property
    def avg_coolant_gap_temp(self):
        """Return the average temperature of the gap coolant
        subchannels at the last axial level"""
        # tot = 0.0
        # for i in range(len(self.coolant_gap_temp)):
        #     tot += (self.gap_params['area'][self.sc_types[i]]
        #             * self.coolant_gap_temp[i])
        # return tot / self.gap_params['total area']
        # return np.sum(self.coolant_gap_temp
        #               * self.gap_params['area frac'])
        return np.dot(self.coolant_gap_temp,
                      self.gap_params['area frac'])

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
        return self.coolant_gap_temp[self._asm_sc_adj[id] - 1].flatten()

    def adjacent_coolant_gap_htc(self, id):
        """Return heat transfer coefficients in gap subchannels
        around a specific assembly

        Parameters
        ----------
        id : int
            Assembly ID

        """
        # return self.coolant_gap_params['htc'][
        #     self.sc_types[self.asm_sc_adj[id] - 1]].flatten()
        return self.coolant_gap_params['htc'][
            self._asm_sc_adj[id] - 1].flatten()

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

        # Inter-assembly gap average velocity
        self.coolant_gap_params['vel'] = \
            (self.gap_flow_rate
             / self.gap_coolant.density
             / self.gap_params['total area'])

        # Gap-average Reynolds number
        self.coolant_gap_params['Re'] = \
            (self.gap_flow_rate
             * self.gap_params['total de']
             / self.gap_coolant.viscosity
             / self.gap_params['total area'])

        # Subchannel Reynolds numbers
        self.coolant_gap_params['Re_sc'] = \
            (self.coolant_gap_params['_Re_sc']  # <-- = m_i * De_i / A_i
             / self.gap_coolant.viscosity)

        # Heat transfer coefficient (via Nusselt number)
        # Although coolant properties are global and velocity is the
        # same everywhere, the subchannels do not have equal hydraulic
        # diameter. Therefore will all have unique Nu
        if self.model is None:
            self.coolant_gap_params['htc'] = np.zeros(self.n_sc)
        elif self.model == 'flow':
            nu = nusselt_db.calculate_sc_Nu(
                self.gap_coolant,
                self.coolant_gap_params['Re_sc'])
            self.coolant_gap_params['htc'] = \
                (self.gap_coolant.thermal_conductivity
                 * nu / self.gap_params['de'])
        else:  # Nu == 1
            # self.coolant_gap_params['htc'] = \
            #     (self.gap_coolant.thermal_conductivity
            #      / self.gap_params['de'])
            self.coolant_gap_params['htc'] = np.ones(self.n_sc)
            self.coolant_gap_params['htc'] *= 0.5 * self.d_gap
        # self.coolant_gap_params['htc'] = np.array([4e4, 4e4])
        #self._Rcond[:, 1:] *= 0.0

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
        T_avg = self.avg_coolant_gap_temp
        self._update_coolant_gap_params(T_avg)
        self._update_energy_balance(dz, asm_duct_temps)

        # Calculate new coolant gap temperatures
        if self.model == 'flow':
            dT = self._flow_model(dz, asm_duct_temps)
            self.coolant_gap_temp += dT

        elif self.model == 'no_flow':
            self.coolant_gap_temp = self._noflow_model(asm_duct_temps)

        elif self.model == 'duct_average':
            self.coolant_gap_temp = self._duct_average_model(asm_duct_temps)

        else:  # self.model == None:
            # No change to coolant gap temp, do nothing
            pass

    def _update_energy_balance(self, dz, approx_duct_temps):
        """Track the energy added to the coolant from each duct wall
        mesh cell; summarize at the end for assembly-assembly energy
        balance"""
        # Convection constant
        h = self.coolant_gap_params['htc'][self._asm_sc_adj - 1]
        # Adj_cool = N_asm x 6 x n_sc
        adj_cool = self.coolant_gap_temp[self._asm_sc_adj - 1]
        # Smush to be N_asm x (6 * n_sc)
        # adj_cool = adj_cool.reshape(adj_cool.shape[0], -1)
        # adj_cool[self.asm_sc_adj < 1] = 0.0
        self.ebal['asm'] += (h * self.gap_params['asm wp']
                             * dz * (approx_duct_temps - adj_cool))

    def _flow_model(self, dz, t_duct):
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
        # CONVECTION TO/FROM DUCT WALL
        # conv_const = np.array([
        #     self.coolant_gap_params['htc'][0] * self.L[0][0],
        #     self.coolant_gap_params['htc'][1] * (2 * self.d['wcorner'])
        # ])
        # conv_const = conv_const[self.sc_types]

        # Look up temperatures, take difference with gap mesh, mask
        # as necessary, and add to total dT
        # dT = (approx_duct_temps[tuple(self._conv_util['inds'][0])]
        #       - self.coolant_gap_temp)
        # dT += (self._conv_util['mask1']
        #        * (approx_duct_temps[tuple(self._conv_util['inds'][1])]
        #           - self.coolant_gap_temp))
        # dT += (self._conv_util['mask2']
        #        * (approx_duct_temps[tuple(self._conv_util['inds'][2])]
        #           - self.coolant_gap_temp))
        # dT *= self._conv_util['const']  # Scale by the HTC and HT area
        C = (self._conv_util['const']
             * self.coolant_gap_params['htc'][:, None])
        dT = C[:, 0] * (t_duct[tuple(self._conv_util['inds'][0])]
                        - self.coolant_gap_temp)
        dT += C[:, 1] * (t_duct[tuple(self._conv_util['inds'][1])]
                         - self.coolant_gap_temp)
        dT += C[:, 2] * (t_duct[tuple(self._conv_util['inds'][2])]
                         - self.coolant_gap_temp)

        # tmp = [t_duct[tuple(self._conv_util['inds'][0])][54],
        #        t_duct[tuple(self._conv_util['inds'][1])][54],
        #        t_duct[tuple(self._conv_util['inds'][2])][54]]
        # print('new model C', C[54])
        # print('new model Tw', tmp)
        # print('new model dT', dT[54])
        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        dT += (self.gap_coolant.thermal_conductivity
               * np.sum((self._Rcond *
                        (self.coolant_gap_temp[self._sc_adj - 1]
                         - self.coolant_gap_temp[..., None])), axis=1))

        # Multiply by dz / m_i / Cp and return
        # poop = dT * dz * self._inv_sc_mfr / self.gap_coolant.heat_capacity
        # print('new model dT', poop[54])
        return (dT * dz * self._inv_sc_mfr
                / self.gap_coolant.heat_capacity)

    def _noflow_model(self, t_duct):
        """Inter-assembly gap conduction model

        Parameters
        ----------
        t_duct : numpy.ndarray
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
        R_conv = (self._conv_util['const']
                  * self.gap_coolant.thermal_conductivity
                  / (0.5 * self.d_gap))
        # R_conv = np.array(
        #     [1 / (1 / self.L[0][0] / self.coolant_gap_params['htc'][0]),
        #      1 / (1 / (2 * self.d['wcorner'])
        #           / self.coolant_gap_params['htc'][1])])

        # if not hasattr(self, '_conv_util'):
        #     self._make_conv_mask()  # creates lookup indices and masks

        # Lookup temperatures and mask as necessary
        T = R_conv[:, 0] * t_duct[tuple(self._conv_util['inds'][0])]
        T += R_conv[:, 1] * t_duct[tuple(self._conv_util['inds'][1])]
        T += R_conv[:, 2] * t_duct[tuple(self._conv_util['inds'][2])]
        # Get the total convection resistance, which will go in the
        # denominator at the end
        C_conv = R_conv[:, 0] + R_conv[:, 1] + R_conv[:, 2]
        # C_conv = copy.deepcopy(R_conv)
        # C_conv += R_conv * self._conv_util['mask1']
        # C_conv += R_conv * self._conv_util['mask2']

        # CONDUCTION TO/FROM OTHER COOLANT CHANNELS
        # Set up masks, if necessary
        # if not hasattr(self, '_Rcond'):
        #     self._make_cond_mask()

        # Now do some thangs c'mon
        R_cond = self._Rcond * self.gap_coolant.thermal_conductivity
        adj_ctemp = self.coolant_gap_temp[self._sc_adj - 1] * R_cond
        # C_cond = np.sum(R_cond, axis=1)  # will add to denom at the end
        C_cond = R_cond[:, 0] + R_cond[:, 1] + R_cond[:, 2]
        # Round it out bb
        T += adj_ctemp[:, 0] + adj_ctemp[:, 1] + adj_ctemp[:, 2]
        return T / (C_cond + C_conv)

    def _duct_average_model(self, t_duct):
        """Inter-assembly gap model that simply averages the adjacent
        duct wall surface temperatures

        Parameters
        ----------
        t_duct : numpy.ndarray
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
        T0 = t_duct[tuple(self._conv_util['inds'][0])]
        T1 = (t_duct[tuple(self._conv_util['inds'][1])]
              * self._conv_util['mask1'])
        T2 = (t_duct[tuple(self._conv_util['inds'][2])]
              * self._conv_util['mask2'])

        # Average nonzero values
        return (np.sum((T0, T1, T2), axis=0)
                / np.count_nonzero((T0, T1, T2), axis=0))

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
        normals = [2 * np.pi / 3,
                   np.pi,
                   4 * np.pi / 3,
                   5 * np.pi / 3,
                   0.0,
                   np.pi / 3]
        # loc = (0.0, 0.0)
        # Directions turning counterclockwise around a hexagon
        # First entry is step from an inner ring to the top of an outer ring
        # The remaining steps are the turns around the hexagon corners
        _turns = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]
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
            row += 1  # already did first step
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
# CORE MAPPING METHODS
########################################################################


def count_rings(n_asm):
    """Identify number of rings given list of assembly
    ring/position inputs"""
    nr = int(np.ceil(0.5 * (1 + np.sqrt(1 + 4 * (n_asm - 1) // 3))))
    return nr


def map_asm(asm_list):
    r"""Map the assembly locations in the core.

    Parameters
    ----------
    asm_list : list
        List of assembly assignments to positions, by position.
        Length is equal to the total number of positions possible
        in the core hexagon

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
    assemblies may be labeled:

      (y)                    (x)
          \     7          .#
           \  _____      #.
        6   /\    / \   2
           /___\1/___\
           \   / \   /
        5   \/_____\/   3
                4

    The map for this 7-assembly core would be as shown below; note
    the rotation so that the first assembly in the new ring starts
    at the top left position.
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
    # _dirs = [(-1, -1), (0, 1), (1, 1), (1, 0),
    #          (0, -1), (-1, -1), (-1, 0)]
    _dirs = [(-1, -1), (1, 0), (1, 1),  (0, 1), (-1, 0), (-1, -1), (0, -1)]
    nr = count_rings(len(asm_list))
    asm_map = np.zeros((nr * 2 - 1, nr * 2 - 1), dtype=int)
    asm_idx = 1
    pos_idx = 1
    # Fill center position
    if not np.isnan(asm_list[0]):
        asm_map[nr - 1, nr - 1] = asm_idx
        asm_idx += 1
    pos_idx += 1
    # Fill rings
    for ring in range(2, int(nr + 1)):
        row = nr - ring
        col = nr - ring
        positions = 6 * (ring - 1)  # all positions on active ring
        corners = np.arange(0, positions, ring - 1)
        d = 1  # first direction
        for pos in range(0, int(positions)):
            if not np.isnan(asm_list[pos_idx - 1]):
                asm_map[row, col] = asm_idx
                asm_idx += 1
            pos_idx += 1
            if pos > 0 and pos in corners:
                d += 1  # change directions at corner
            row, col = row + _dirs[d][0], col + _dirs[d][1]
    return asm_map


def map_adjacent_assemblies(asm_map):
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
    adj = np.zeros((np.max(asm_map), 6), dtype=int)
    _dirs = [(0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0)]
    for row in range(0, len(asm_map)):
        for col in range(0, len(asm_map[row])):
            id = asm_map[row, col]
            if id != 0:
                tmp = np.zeros(6, dtype=int)
                for i in range(0, len(_dirs)):
                    address = tuple(sum(x) for x in
                                    zip((row, col), _dirs[i]))
                    if all(idx >= 0 for idx in address):
                        try:
                            tmp[i] = asm_map[address]
                        except IndexError:
                            pass
                adj[id - 1] = tmp
    return adj


########################################################################
# GAP SUBCHANNEL MAPPING METHODS
########################################################################

#
# def _which_asm_has_finer_mesh(asm, neighbor=None):
#     """Determine which assembly has finer meshing
#
#     Parameters
#     ----------
#     asm1 : DASSH Assembly object
#         Active assembly around which we are creating mesh
#     asm2 (optional) : DASSH Assembly object
#         Neighboring assembly across active hex side
#
#     Returns
#     -------
#     tuple
#         DASSH Assembly object
#             One of the two input DASSH Assembly objects with the
#             finer meshing in its pin bundle region
#         int
#             Number of gap edge subchannels per side
#
#     """
#     # Active assembly edge subchannels per hex side
#     sc_per_side = 0  # Assume unrodded, no side subchannels
#     if asm.has_rodded:
#         sc_per_side = asm.rodded.n_ring - 1
#     if neighbor is None:
#         return asm, sc_per_side
#     else:
#         # Neighbor edge subchannels per hex side
#         sc_per_side_adj = 0  # Assume unrodded, no side subchannels
#         if neighbor.has_rodded:
#             sc_per_side_adj = neighbor.rodded.n_ring - 1
#         # If both assemblies have same number of sc_per_side:
#         # Choose the spacing with the smaller pin pitch
#         if sc_per_side == sc_per_side_adj:
#             if asm.rodded.pin_pitch < neighbor.rodded.pin_pitch:
#                 return asm, sc_per_side
#             else:
#                 return neighbor, sc_per_side_adj
#         # Otherwise, choose the meshing with more edge subchannels
#         elif sc_per_side < sc_per_side_adj:
#             return neighbor, sc_per_side_adj
#         else:
#             return asm, sc_per_side
#


def _which_asm_has_finer_mesh(asm, neighbor=None):
    """Determine which assembly has finer meshing

    Parameters
    ----------
    asm1 : DASSH Assembly object
        Active assembly around which we are creating mesh
    asm2 (optional) : DASSH Assembly object
        Neighboring assembly across active hex side

    Returns
    -------
    tuple
        DASSH Assembly object
            One of the two input DASSH Assembly objects with the
            finer meshing in its pin bundle region
        int
            Number of gap edge subchannels per side

    """
    if neighbor is None:  # Use active asm parameters on this hex side
        sc_per_side = 0
        if asm.has_rodded:
            sc_per_side = asm.rodded.n_ring - 1
        return asm, sc_per_side

    else:  # Need to compare them
        # If one asm is rodded and the other isn't, use the rodded asm
        if asm.has_rodded and not neighbor.has_rodded:
            return asm, asm.rodded.n_ring - 1
        elif not asm.has_rodded and neighbor.has_rodded:
            return neighbor, neighbor.rodded.n_ring - 1
        # If both unrodded, trivial
        elif not asm.has_rodded and not neighbor.has_rodded:
            return asm, 0
        # If both rodded, need to compare
        else:
            sc_per_side = asm.rodded.n_ring - 1
            sc_per_side_adj = neighbor.rodded.n_ring - 1

            # If both assemblies have same number of sc_per_side:
            # Choose the spacing with the smaller pin pitch
            if sc_per_side == sc_per_side_adj:
                if asm.rodded.pin_pitch < neighbor.rodded.pin_pitch:
                    return asm, sc_per_side
                else:
                    return neighbor, sc_per_side_adj
            # Otherwise, choose the meshing with more edge subchannels
            elif sc_per_side < sc_per_side_adj:
                return neighbor, sc_per_side_adj
            else:
                return asm, sc_per_side


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
        term1 = (core_obj.coolant_gap_params['htc']
                 * np.sum(core_obj._conv_util['const'], axis=1)
                 * core_obj._inv_sc_mfr
                 / core_obj.gap_coolant.heat_capacity)
        term2 = (core_obj.gap_coolant.thermal_conductivity
                 * core_obj.d_gap
                 * core_obj._inv_sc_mfr
                 / core_obj.gap_coolant.heat_capacity
                 / np.sum(core_obj.gap_params['L'], axis=1))
        dz = 1 / (term1 + term2)

        # # Only corner -> corner interassembly gap subchannels
        # # NOTE: this means that the user ONLY specified 1-pin assemblies
        # # lol, how ridiculous is that?
        # # if core_obj._sc_per_asm == 6:
        # if core_obj._asm_sc_adj.shape[1] == 6:
        #     return (_cons9_999(
        #         sc_mfr[1],
        #         core_obj.L[1][1],
        #         core_obj.d['gap'],
        #         core_obj.d['wcorner'],
        #         core_obj.gap_coolant.thermal_conductivity,
        #         core_obj.gap_coolant.heat_capacity,
        #         core_obj.coolant_gap_params['htc'][1]), '9-999')
        #
        # else:
        #     sc_code.append('9-888')
        #     dz.append(
        #         _cons9_888(sc_mfr[1],
        #                    core_obj.L[0][1],
        #                    core_obj.d['gap'],
        #                    core_obj.d['wcorner'],
        #                    core_obj.gap_coolant.thermal_conductivity,
        #                    core_obj.gap_coolant.heat_capacity,
        #                    core_obj.coolant_gap_params['htc'][1]))
        #
        #     # Edge subchannel --> corner/corner subchannel
        #     if core_obj._sc_per_asm == 12:
        #         sc_code.append('8-99')
        #         dz.append(
        #             _cons8_99(sc_mfr[0],
        #                       core_obj.L[0][0],
        #                       core_obj.L[0][1],
        #                       core_obj.d['gap'],
        #                       core_obj.gap_coolant.thermal_conductivity,
        #                       core_obj.gap_coolant.heat_capacity,
        #                       core_obj.coolant_gap_params['htc'][0]))
        #
        #     # Edge subchannel --> edge/corner subchannel
        #     else:
        #         sc_code.append('8-89')
        #         dz.append(
        #             _cons8_89(sc_mfr[0],
        #                       core_obj.L[0][0],
        #                       core_obj.L[0][1],
        #                       core_obj.d['gap'],
        #                       core_obj.gap_coolant.thermal_conductivity,
        #                       core_obj.gap_coolant.heat_capacity,
        #                       core_obj.coolant_gap_params['htc'][0]))
        #
        #     # Edge subchannel --> edge/edge subchannel
        #     if core_obj._sc_per_asm >= 24:
        #         sc_code.append('8-88')
        #         dz.append(
        #             _cons8_88(sc_mfr[0],
        #                       core_obj.L[0][0],
        #                       core_obj.d['gap'],
        #                       core_obj.gap_coolant.thermal_conductivity,
        #                       core_obj.gap_coolant.heat_capacity,
        #                       core_obj.coolant_gap_params['htc'][0]))

    # Reset the coolant temperature
    core_obj._update_coolant_gap_params(_temp_in)
    # print(min_dz)
    min_dz = np.min(dz)
    # return min_dz, sc_code[dz.index(min_dz)]
    return min_dz, 'X-XXX'


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
