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
date: 2020-06-10
author: Milos Atz
This module defines the Subchannel class to map coolant and structural
subchannels in and between hexagonal fuel assemblies.
"""
# Still need to implement SC-PIN adjacency (N_sc x 3)
########################################################################
import numpy as np


# _sqrt3over2 = 0.866025403784439
_invSqrt3over2 = 1.154700538379252
_sqrt3over3 = 0.5773502691896257
_sqrt3 = 1.7320508075688772


class Subchannel(object):
    """Map subchannels and neighbors in hexagonal fuel assemblies.

    Define the location and adjacency of the subchannels in the
    assembly. This is done by constructing a array mapping the
    the relative position of subchannels in the assembly. Like for
    the fuel pin labels, the subchannels are defined in concentric
    rings around the assembly. The array informs connections between
    subchannels and (1) other subchannels, and (2) adjacent fuel
    pins, as well as the X-Y position of the subchannel centroids.
    This object also defines the types of subchannels.

    Parameters
    ----------
    n_ring : int
        Number of pin rings (incl. center pin) in the assembly
    pin_pitch : float
        Pin center-to-center pitch distance
    pin_diameter : float
        Diameter of pin outer clad
    pin_map : numpy.ndarray
        Mapping of pin IDs into array (from PinLattice)
    pin_xy : numpy.ndarray
        XY coordinates of pins in the assembly (from PinLattice)
    duct_ftf : list
        List of tuples containing inner and outer duct
        flat-to-flat distances
    test: bool (optional)
        If testing, do not run all the instantiation methods; instead,
        allow the object to be instantiated without calling them so
        they can be called incrementally and independently

    Attributes
    ----------
    n_sc : dict
        Number of subchannels of each type in different assembly regions
    type : numpy.ndarray
        Type of each subchannel in the assembly
    sc_adj : numpy.ndarray
        Subchannel neighbors for each subchannel in the assembly
    pin_adj : numpy.ndarray
        Subchannel neighbors for each pin in the assembly
    sc_xy : numpy.ndarray
        Array (N_sc x 2) containing the X-Y coordinates of each
        subchannel in the assembly

    """

    _edge_angle = [np.pi / 3, 0.0, 5 * np.pi / 3, 4 * np.pi / 3,
                   np.pi, 2 * np.pi / 3]
    _corner_angle = [np.pi / 6, 11 * np.pi / 6, 3 * np.pi / 2,
                     7 * np.pi / 6, 5 * np.pi / 6, np.pi / 2]

    def __init__(self, n_ring, pin_pitch, pin_diameter,
                 pin_map, pin_xy, duct_ftf, test=False):
        """Instantiate a subchannelSetup object."""
        # Count the different types of subchannels
        self.n_sc = {}
        self.n_sc['coolant'] = {}
        self.n_sc['coolant']['interior'] = 6 * (n_ring - 1)**2
        self.n_sc['coolant']['edge'] = (n_ring - 1) * 6
        self.n_sc['coolant']['corner'] = 6
        self.n_sc['coolant']['total'] = 6 * (n_ring**2
                                             - n_ring + 1)

        # Assemblies with multiple ducts have the same number of SC
        # in each duct/bypass
        self.n_sc['duct'] = {}
        self.n_sc['duct']['edge'] = (n_ring - 1) * 6
        self.n_sc['duct']['corner'] = 6
        self.n_sc['duct']['total'] = 6 * n_ring

        # Bypass channels - if present, same number as duct channels
        self.n_sc['bypass'] = {}
        if len(duct_ftf) > 1:
            self.n_sc['bypass']['edge'] = self.n_sc['duct']['edge']
            self.n_sc['bypass']['corner'] = 6
            self.n_sc['bypass']['total'] = self.n_sc['duct']['total']
        else:
            self.n_sc['bypass']['edge'] = 0
            self.n_sc['bypass']['corner'] = 0
            self.n_sc['bypass']['total'] = 0
        # Get the total number of subchannels
        self.n_sc['total'] = (self.n_sc['coolant']['total']
                              + (2 * len(duct_ftf) - 1)
                              * self.n_sc['duct']['total'])

        # --------------------------------------------------------------
        if test:
            return

        # --------------------------------------------------------------
        # Subchannel types:
        self.type = self.setup_sc_type(n_ring, duct_ftf)

        # Subchannel map
        self._int_map = self._make_interior_sc_map(n_ring)
        self._ext_map = self._make_exterior_sc_map(n_ring)
        self._map = np.add(self._int_map, self._ext_map)

        # Subchannel-subchannel adjacency
        # See method(s) below for more details and docstrings
        self.sc_adj = self.find_sc_sc_neighbors(n_ring, duct_ftf)

        # Subchannel-pin adjacency
        # (N_pin x 6); subchannels adjacent to each pin
        # See method(s) below for more details and docstrings
        self.pin_adj = self.find_pin_sc_neighbors(n_ring, pin_map)

        # Pin-subchannel adjacency
        # (N_coolant_sc x 6); pins adjacent to each coolant subchannel
        self.rev_pin_adj = self.reverse_pin_neighbors()

        # Subchannel X-Y position
        self.xy = self.find_sc_xy(n_ring, pin_pitch, pin_diameter,
                                  pin_xy, duct_ftf)

        # UPDATE ARRAYS TO PYTHON INDEXING (Type 1 --> Type 0)
        self.type -= 1
        self.sc_adj -= 1
        self.pin_adj -= 1

    ####################################################################
    # SUBCHANNEL TYPE SETUP
    ####################################################################

    def setup_sc_type(self, n_ring, duct_ftf):
        """Set up the subchannel types.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        duct_ftf : list
            List of tuples containing the inner/outer flat-to-flat
            distances for each duct in the assembly

        Notes
        -----
        Coolant:    (1) Interior  (2) Edge      (3) Corner
        Duct:       (4) Edge      (5) Corner
        Bypass:     (6) Edge      (7) Corner

        (Inter-asm gap not attributed to any assembly)

        """
        sc_type = np.ones(self.n_sc['coolant']['interior'], dtype="int")
        # Append coolant SC types 2 and 3 for each side
        ext = np.ones(int(n_ring - 1), dtype="int") * 2
        ext = np.append(ext, np.array([3], dtype="int"))
        for side in range(0, 6):  # loop over hex sides
            sc_type = np.append(sc_type, ext)
        # Append duct wall SC types 4,5 for each side; same number of
        # SC as for edge/corner coolant channels with same arrangement
        duct = ext + 2
        bypass = duct + 2
        for i in range(0, len(duct_ftf)):
            if i > 0:  # bypass only occurs within outer ducts i > 0
                for side in range(0, 6):
                    sc_type = np.append(sc_type, bypass)
            for side in range(0, 6):  # loop over hex sides
                sc_type = np.append(sc_type, duct)
        assert len(sc_type) == self.n_sc['total']
        return sc_type

    ####################################################################
    # SUBCHANNEL MAP
    ####################################################################

    @classmethod
    def _make_interior_sc_map(cls, n_ring):
        """Generate a map of the interior subchannels.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly

        Returns
        -------
        numpy.ndarray
            Map of the locations of the interior subchannels

        Notes
        -----
        Walk a "ring" around an array labeling the interior-type
        subchannels that occupy the row-column positions. The shape
        of the rings is designed to show the subchannel-pin adjacency

        Example
        -------
        For an assembly with 3 pin rings, there are two rings of
        interior subchannels (one between the first and second rings
        and another between the second and third rings). The final
        result for this system is obtained by filling in entries
        for each subchannel ring:

        _step 1               _step 2               Final result
        (around center pin)  (between ring 2-3)   Int. subchannel map
        | 00  00  00  00 |   | 00  24  07  00 |   | 00  24  07  00 |
        | 00  00  00  00 |   | 22  23  08  09 |   | 22  23  08  09 |
        | 00  06  01  00 |   | 21  00  00  10 |   | 21  06  01  10 |
        | 00  05  02  00 | + | 20  00  00  11 | = | 20  05  02  11 |
        | 00  04  03  00 |   | 19  00  00  12 |   | 19  04  03  12 |
        | 00  00  00  00 |   | 18  17  14  13 |   | 18  17  14  13 |
        | 00  00  00  00 |   | 00  16  15  00 |   | 00  16  15  00 |

        The map below distorts the array to illustrate the ring
        arrangement of subchannels around pins from the example.
        The pin locations are at the "x" marks; pick any "x" and
        you can see the subchannels that surround it. The 00 entries
        are shown to lie outside of the interior subchannel domain.

                       X
        |  00   X   24   07   X   00  |
        |X   22   23   X   08   09   X|
        |  21   X   06   01   X   10  |
        |X   20   05   X   02   11   X|
        |  19   X   04   03   X   12  |
        |X   18   17   X   14   13   X|
        |  00   X   16   15   X   00  |
                       X

        """
        map = np.zeros((4 * n_ring - 1, 2 * n_ring), dtype="int")
        for ring in range(2, n_ring + 1):
            # Starting position
            loc = ((n_ring - ring) * 2 + 2, n_ring)
            sc_id = np.amax(map) + 1
            map[loc] = sc_id
            # Begin walking through the ring
            loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
            for i in range(0, ring - 2):  # Walk down the weird side
                loc, map, sc_id = cls._step(loc, 'right', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
            for i in range(0, ring - 2):
                loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
            for i in range(0, ring - 2):
                loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'left', map, sc_id)
            loc, map, sc_id = cls._step(loc, 'down', map, sc_id)
            loc, map, sc_id = cls._step(loc, 'left', map, sc_id)
            loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
            for i in range(0, ring - 2):  # Walk up the weird other side
                loc, map, sc_id = cls._step(loc, 'left', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
            for i in range(0, ring - 2):
                loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
            for i in range(0, ring - 2):
                loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
                loc, map, sc_id = cls._step(loc, 'right', map, sc_id)
            # Take the last _step up
            loc, map, sc_id = cls._step(loc, 'up', map, sc_id)
        return map

    def _make_exterior_sc_map(self, n_ring):
        r"""Generate map of the exterior (edge and corner) subchannels.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly

        Walk a "ring" around an array labeling the exterior-type
        subchannels that occupy the row-column positions. The exterior
        subchannels are the edge and corner subchannels. The shape
        of the rings is designed to show the subchannel-pin adjacency.
        Because each pin on the outer ring only touches 5 subchannels,
        this method is different from the one the labels the interior
        subchannels in that one of the six possible subchannel
        positions is skipped at each of the six hexagon faces.

        Returns
        -------
        numpy.ndarray
            Array mapping the locations of the edge, corner subchannels

        Notes
        -----
        If the positions around pin i are labeled clockwise:

                            \ 6 | 1 /
                              \ | /
                            5  > <  2
                              / | \
                            / 4 | 3 \

        Then depending on which assembly hexagonal face pin i is
        located, one of the positions is skipped:

                     skip pos. 6    skip pos. 1
           (incl. 90 deg corner) /\ (incl. 30 deg. corner)
                                /  \
                skip pos. 5    |    |  skip pos. 2
        (incl. 150 deg corner) |    |  (incl. 330 deg. corner)
                                \  /
                    skip pos. 4  \/  skip pos. 3
         (incl. 210 deg corner)     (incl. 270 deg. corner)

        Example
        -------
        For an assembly with 3 pin rings, subchannels 25-42 are
        exterior subchannels. The array that holds the pin IDs
        for this system is:

        | 00 00 00 42 00 00 |
        | 00 00 41 25 00 00 |
        | 39 40 00 00 26 00 |
        | 00 00 00 00 00 27 |
        | 38 00 00 00 00 28 |
        | 00 00 00 00 00 00 |
        | 37 00 00 00 00 29 |
        | 36 00 00 00 00 00 |
        | 00 35 00 00 31 30 |
        | 00 00 34 32 00 00 |
        | 00 00 33 00 00 00 |

        """
        map = np.zeros((4 * n_ring - 1, 2 * n_ring), dtype="int")
        sc_id = self.n_sc['coolant']['interior']
        loc = (1, n_ring - 1)  # starting location
        for i in range(0, n_ring):  # NE side: skip "sector 1"
            loc, map, sc_id = self._step(loc, 'right', map, sc_id)
            loc = self._move(loc, 'down')  # need an extra step down
        loc = self._move(loc, 'up')  # undo extra move from last iter
        for i in range(0, n_ring):  # E side: skip "sector 2"
            loc, map, sc_id = self._step(loc, 'down', map, sc_id)
            loc = self._move(loc, 'down')  # need an extra step down
        loc = self._move(loc, 'up')  # undo extra move from last iter
        for i in range(0, n_ring):  # SE side: skip "sector 3"
            loc, map, sc_id = self._step(loc, 'left', map, sc_id)
            loc = self._move(loc, 'down')  # need an extra step right
        loc = self._move(loc, 'up')  # undo extra move from last iter
        for i in range(0, n_ring):  # SW side: skip "sector 4"
            loc, map, sc_id = self._step(loc, 'up', map, sc_id)
            loc = self._move(loc, 'left')  # need an extra step
        loc = self._move(loc, 'right')  # undo extra move from last iter
        for i in range(0, n_ring):  # W side: skip "sector 5"
            loc, map, sc_id = self._step(loc, 'up', map, sc_id)
            loc = self._move(loc, 'up')  # need an extra step
        loc = self._move(loc, 'down')  # undo extra move from last iter
        for i in range(0, n_ring):  # NW side: skip "sector 6"
            loc, map, sc_id = self._step(loc, 'right', map, sc_id)
            loc = self._move(loc, 'up')  # need an extra step right
        return map

    @classmethod
    def _step(cls, loc, dir, map, sc_id):
        """Take a _step into an adjacent position and fill it

        Parameters
        ----------
        loc : tuple
            The row,column entries of the current position
        dir : str
            The direction ("left", "right", "up", or "down") to _move
        map : numpy array
            The array map being updated
        sc_id : int
            The current subchannel ID number

        Returns
        -------
        tuple
            Updated position
        numpy.ndarray
            Updated array of subchannel map
        int
            Updated subchannel ID

        """
        new_loc = cls._move(loc, dir)
        new_scid = sc_id + 1
        map[new_loc] = new_scid
        return new_loc, map, new_scid

    @staticmethod
    def _move(loc, dir):
        """_move to an adjacent array entry

        Parameters
        ----------
        loc : tuple
            (row, col) of the current location in the array
        dir : str
            Direction ('left, right, up, down') in which to _step

        Returns
        -------
        tuple
            (row, col) after taking _step

        """
        if dir == 'left':
            loc = (loc[0], loc[1] - 1)
        elif dir == 'right':
            loc = (loc[0], loc[1] + 1)
        elif dir == 'up':
            loc = (loc[0] - 1, loc[1])
        elif dir == 'down':
            loc = (loc[0] + 1, loc[1])
        else:
            msg = 'Direction must be one of: [left, right, up, down]'
            raise ValueError(msg)
        return loc

    ####################################################################
    # SUBCHANNEL-SUBCHANNEL NEIGHBORS
    # For each subchannel, determine what subchannels are next to it
    ####################################################################

    def find_sc_sc_neighbors(self, n_ring, duct_ftf):
        """Define the connections between neighboring subchannels.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        duct_ftf : list
            List of tuples containing duct inner/outer flat-to-flat
            distances for each duct in the assembly

        Notes
        -----
        The columns of the neighbors array imply the type of
        subchannel to which the connection is being established:
        Columns 1 - 3: Connection with interior subchannels
            Three columns are reserved for connections of a subchannel
            and an adjacent interior subchannel - for an interior
            subchannel in contact with three other interior subchannels,
            all three of these columns are filled. Edge subchannels
            connect with one interior subchannel. Corner subchannels do
            not touch any interior subchannels.
        Columns 4 - 5: Connection with edge subchannels
            Edge subchannels always touch two other edge subchannels,
            which may be corners; corner subchannels always touch two
            edge subchannels. Interior subchannels along the assembly
            periphery will be in contact with one edge subchannels.
        Columns 5 - 6: Connection with duct subchannels
        Columns 7 - 8: Connection with bypass subchannels (optional)
        Columns 9 - 10: Connection with outer duct (optional)
        ...

        """
        # Array has 5 columns for interior coolant subchannels
        # First duct ring adds 2 cols; every subsequent duct ring
        # adds 4 more (2 for the duct, 2 for the bypass flow).
        ncol = 5 + 4 * len(duct_ftf) - 2  # 5 cols for coolant subchannels
        sc_sc = np.zeros((self.n_sc['total'], ncol), dtype="int")
        sc_sc = self._connect_int_sc(sc_sc)
        # print(sc_sc)
        sc_sc = self._connect_int_ext_sc(sc_sc)
        # print(sc_sc)
        sc_sc = self._connect_ext_sc(sc_sc)
        # print(sc_sc)
        sc_sc = self._connect_duct_bypass_sc(sc_sc, n_ring, duct_ftf)
        return sc_sc

    def _connect_int_sc(self, sc_adj):
        """Define the connections between interior subchannels

        Parameters
        ----------
        sc_adj : numpy.ndarray
            Array defining subchannel neighbors

        Returns
        -------
        numpy.ndarray
            Updated subchannel neighbor definitions

        """
        for col in range(1, self._int_map.shape[1] - 1):
            # Vertical connections: at least one, at most two
            temp = self._int_map[:, col]
            temp = temp[temp != 0]
            if temp.size == 0:
                continue
            else:
                for i in range(1, len(temp) - 1):
                    sc_adj[temp[i] - 1, 0] = temp[i - 1]
                    sc_adj[temp[i] - 1, 1] = temp[i + 1]
                # first and last entries from "temp" are separate
                sc_adj[temp[0] - 1, 1] = temp[1]
                sc_adj[temp[-1] - 1, 0] = temp[-2]
            # Horizontal connections (at most 1); we are mapping the
            # connections between the current column ("j") and column
            # "j+1". The first row with a connection is the one for
            # which both column "j" and column "j+1" are nonzero.
            try:
                r1 = min(np.intersect1d((self._int_map[:, col]
                                         .nonzero()[0]),
                                        (self._int_map[:, col + 1]
                                         .nonzero()[0])))
            except ValueError:  # trying to compare with zero columns
                continue
            else:  # make connection every other row
                for row in range(r1, self._int_map.shape[0], 2):
                    sc_i = self._int_map[row, col]
                    sc_ip1 = self._int_map[row, col + 1]
                    if sc_i == 0 or sc_ip1 == 0:
                        continue
                    else:
                        sc_adj[sc_i - 1, 2] = sc_ip1
                        sc_adj[sc_ip1 - 1, 2] = sc_i
        return sc_adj

    def _connect_int_ext_sc(self, sc_adj):
        """Define connections between interior and exterior subchannels.

        Connections between interior and exterior (edge, corner)
        subchannels are handled individually - this method figures
        out which interior subchannels are missing an interior
        connection; the position of the missing connection informs
        the direction in which to look in the assembly subchannel
        map for the adjacent edge subchannel.

        Parameters
        ----------
        sc_adj : numpy.ndarray
            Array defining subchannel neighbors

        Returns
        -------
        numpy.ndarray
            Updated subchannel neighbor definitions

        """
        # Total number of missing interior-exterior subchannel
        # connections; connect only via interior-edge subchannels
        n_missing_total = self.n_sc['coolant']['edge']
        idx_missing = 0
        for i in range(0, self.n_sc['coolant']['interior']):
            # Find subchannel with missing value in map
            row, col = np.where(self._map == i + 1)
            # Looking for missing connections in the first three cols
            missing = np.where(sc_adj[i][:3] == 0)
            if len(missing[0]) > 0:  # some missing, count that
                idx_missing += 1
            for idx in range(0, len(missing[0])):
                if missing[0][idx] == 0:  # look upward for value
                    sc_adj[i, idx + 3] = self._map[row - 1, col]
                    sc_adj[self._map[row - 1, col] - 1, idx] = i + 1
                elif missing[0][idx] == 1:  # look down
                    sc_adj[i, idx + 3] = self._map[row + 1, col]
                    sc_adj[self._map[row + 1, col] - 1, idx] = i + 1
                else:  # look horizontallyf
                    # Because we're going clockwise, if we're on the
                    # first half of subchannels w/ missing connections
                    # we're on the RIGHT side of the asm
                    if idx_missing < n_missing_total / 2:
                        sc_adj[i, idx + 3] = self._map[row, col + 1]
                        sc_adj[self._map[row, col + 1] - 1, idx] = i + 1
                    else:  # SC on the left side of asm (higher numbers)
                        sc_adj[i, idx + 3] = self._map[row, col - 1]
                        sc_adj[self._map[row, col - 1] - 1, idx] = i + 1
        return sc_adj

    def _connect_ext_sc(self, sc_adj):
        """Connect edge and corner subchannels to each other.

        Parameters
        ----------
        sc_adj : numpy.ndarray
            Array defining subchannel neighbors

        Returns
        -------
        numpy.ndarray
            Updated subchannel neighbor definitions

        Notes
        -----
        Because they are arranged in a ring, each exterior
        subchannel touches the subchannels immediately in front and
        behind it

        """
        ext_sc = np.arange(self.n_sc['coolant']['interior'] + 1,
                           self.n_sc['coolant']['total'] + 1, 1)
        for i in range(-1, len(ext_sc) - 1):
            # backward connection: i+1 back to i
            sc_adj[ext_sc[i + 1] - 1, 3] = ext_sc[i]
            # forward connection: i forward to i+1
            sc_adj[ext_sc[i] - 1, 4] = ext_sc[i + 1]
        return sc_adj

    def _connect_duct_bypass_sc(self, sc_adj, n_ring, duct_ftf):
        """Connect successive duct and bypass rings.

        Parameters
        ----------
        sc_adj : numpy.ndarray
            Array defining subchannel neighbors
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        duct_ftf : list
            List of tuple containing the inner/outer duct flat-to-flat
            distances for each duct in the assembly

        Returns
        -------
        numpy.ndarray
            Updated subchannel neighbor definitions

        Notes
        -----
        The duct and bypass subchannels are organized in rings; each
        ring has the same number of subchannels. This method connects
        each ring to the previous.

        """
        sc = (np.arange(1, self.n_sc['duct']['total'] + 1, 1)
              + self.n_sc['coolant']['total'])
        for r in range(1, 2 * len(duct_ftf)):  # r:ring
            # duct ring 1: cols 5-6; bypass ring 1: cols 7-8, ...
            # relationship between ring and cols: 2r+3, 2r+4
            for i in range(-1, len(sc) - 1):  # i is current sc
                # backward connection: i+1 back to i (ring r)
                sc_adj[sc[i + 1] - 1, 2 * r + 3] = sc[i]
                # forward connection: i forward to i+1 (ring r)
                sc_adj[sc[i] - 1, 2 * r + 4] = sc[i + 1]
                # inward connection: i+1 (ring r) to i+1 (ring r-1)
                sc_adj[sc[i + 1] - 1, 2 * r + 2] = \
                    sc[i + 1] - 6 * n_ring
                # outward connection i+1 (ring r-1) up to i+1 (ring r)
                sc_adj[sc[i] - 1 - 6 * n_ring, 2 * r + 3] = sc[i]
            sc = sc + 6 * n_ring
        return sc_adj

    ####################################################################
    # PIN-SUBCHANNEL ADJACENCY
    # Determine the subchannels that neighbor each pin
    ####################################################################

    def find_pin_sc_neighbors(self, n_ring, pin_map):
        """Determine the subchannels that neighbor each pin.

        When this method is called, the subchannel position map has
        already been created and is linked to the subchannel map
        for the pins by "stamping" a 3x2 cutout of the map centered
        where each pin should be.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        pin_map : numpy.ndarray
            Map of pin layout in the assembly

        Returns
        -------
        numpy.ndarray
            Array (n_pin x 6) indicating the subchannels that neighbor
            each pin.

        """
        pin_sc_adj = np.zeros((len(pin_map[pin_map != 0]), 6),
                              dtype="int")
        for row in range(0, len(pin_map)):
            loc = (row, n_ring - row)  # start loc
            for p in pin_map[row]:
                if p != 0:
                    pin_sc_adj[p - 1, 0] = self._map[loc]
                    pin_sc_adj[p - 1, 1] = self._map[loc[0] + 1, loc[1]]
                    pin_sc_adj[p - 1, 2] = self._map[loc[0] + 2, loc[1]]
                    pin_sc_adj[p - 1, 3] = self._map[loc[0] + 2,
                                                     loc[1] - 1]
                    pin_sc_adj[p - 1, 4] = self._map[loc[0] + 1,
                                                     loc[1] - 1]
                    pin_sc_adj[p - 1, 5] = self._map[loc[0], loc[1] - 1]
                loc = (loc[0] + 1, loc[1] + 1)  # update location
        return pin_sc_adj

    def reverse_pin_neighbors(self):
        """Determine pins that are adjacent to each subchannel

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array (N_coolant_sc x 3) of adjacent pin indices for each
            subchannel; where less than 3 pins are adjacent, the array
            is filled with -1 which are later filtered

        Notes
        -----
        This is the "inverse" of the "pin_adj" attribute and is useful
        for calculating the power assigned to each subchannel

        """
        butt = [[] for i in range(self.n_sc['coolant']['total'])]
        for pin in range(len(self.pin_adj)):
            for sc in self.pin_adj[pin]:
                if sc > 0:
                    butt[sc - 1].append(pin)
        for poop in butt:
            while len(poop) < 3:
                poop.append(-1)
        return np.array(butt)

    ####################################################################
    # SUBCHANNEL XY
    # Determine the XY positions of the centroid of each subchannel
    ####################################################################

    def find_sc_xy(self, n_ring, pin_pitch, pin_diameter,
                   pin_xy, duct_ftf):
        """Determine X-Y coordinates of the subchannels.

        Parameters
        ----------
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        pin_pitch : float
            Pin center-to-center pitch distance
        pin_diameter : float
            Diameter of pin outer clad
        pin_xy : numpy.ndarray
            XY coordinates of pins in the assembly (from PinLattice)
        duct_ftf : list
            List of tuples containing the inner and outer flat-to-
            flat distances of the duct walls, ordered from inner
            to outer ducts.

        Returns
        -------
        numpy.ndarray
            Array (N_sc x 2) of subchannel X-Y coordinates

        Notes
        -----
        This method uses the subchannel-pin adjacency array and the
        X-Y coordinates of the pins to define the X-Y coordinate of
        each subchannel based on its location relative to each fuel
        pin, with the center pin located at (0, 0).

        """
        # Preallocate sc_xy array. The coolant subchannel coords are
        # entered into this array; the duct and bypass subchannel
        # coords are appended to it.
        sc_xy = np.zeros((self.n_sc['coolant']['total'], 2))
        # COOLANT SUBCHANNELS -----------------------------------------
        sc_xy = self._find_interior_xy(sc_xy, pin_xy, pin_pitch)
        sc_xy = self._find_edge_xy(sc_xy, pin_xy, n_ring, pin_pitch,
                                   min(duct_ftf[0]))
        sc_xy = self._find_corner_xy(sc_xy, pin_xy, n_ring, pin_pitch,
                                     pin_diameter, min(duct_ftf[0]))
        # DUCT AND BYPASS SUBCHANNELS ----------------------------------
        sc_xy = self._find_duct_bypass_xy(sc_xy, n_ring, pin_pitch,
                                          pin_diameter, duct_ftf)
        return sc_xy

    def _find_interior_xy(self, scxy, pin_xy, pitch):
        """Determine the X-Y locations of the interior subchannels.

        Parameters
        ----------
        scxy : numpy.ndarray
            Array (N_sc x 2) of subchannel X-Y coordinates
        pin_xy : numpy.ndarray
            Array (Npin x 2) of pin X-Y coordinates
        pitch : float
            Pin center-to-center pitch distance

        Returns
        -------
        numpy.ndarray
            Updated array (N_sc x 2) of subchannel X-Y coordinates

        """
        # The interior subchannels are equilateral triangles. The
        # distance between corners (side length, "a") is the pin
        # center-to-center pitch distance. The distance from a corner
        # to the centroid is equal to 2h/3, where h = a*sqrt(3)/2.
        dc_int = pitch * _sqrt3over3
        # The angle of the interior subchannel centroid relative to
        # the center of the pin can be inferred by its position in the
        # pin-subchannel adjacency array. For position 0, 1, 2, 3, 4,
        # or 5, the corresponding angle is 60, 0, 300, 240, 180, or
        # 120 degrees (clockwise), respectively.
        int_angle = [np.pi / 3, 0.0, 5 * np.pi / 3,
                     4 * np.pi / 3, np.pi, 2 * np.pi / 3]
        for i in range(1, self.n_sc['coolant']['interior'] + 1):
            loc = np.where(self.pin_adj == i)
            p = loc[0][0]  # take the first value
            sector = loc[1][0]  # location relative to pin
            x0, y0 = pin_xy[p]
            scxy[i - 1, 0] = x0 + (np.cos(int_angle[sector]) * dc_int)
            scxy[i - 1, 1] = y0 + (np.sin(int_angle[sector]) * dc_int)
        return scxy

    def _find_corner_xy(self, scxy, pin_xy, n_ring, pitch, dpin, dftf):
        """Determine the X-Y locations of the corner subchannels.

        Parameters
        ----------
        scxy : numpy.ndarray
            Array (N_sc x 2) of subchannel X-Y coordinates
        pin_xy : numpy.ndarray
            Array (Npin x 2) of pin X-Y coordinates
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        pitch : float
            Pin center-to-center pitch distance
        dpin : float
            Pin diameter
        dftf : float
            Flat-to-flat distance for the inner-most assembly
            duct wall

        Returns
        -------
        numpy.ndarray
            Updated array (N_sc x 2) of subchannel X-Y coordinates

        Notes
        -----
        Corner: The centroid of the corner cell is located in the
        line projected from the pin row, halfway between the corner
        pin outer diameter and the inner duct wall corner. The inner
        duct flat-to-flat distance is given; based on that, the long
        diagonal is:

        ld = ftf * 2 / sqrt(3).

        The distance between the pin center and the duct corner is:

        0.5 * (ld - (2 * n_ring - 1) * pitch).

        The factor in front of the pitch is the number of pins along
        the long diagonal in the assembly.

        The corner centroids occur on the diagonals of the assembly
        hexagon, so the angles are 30, 330, 270, 210, 150, and 90
        degrees, respectively.
        """
        # Distance between pin center and corner subchannel centroid
        dc_corner = 0.25 * (_invSqrt3over2 * dftf + dpin
                            - 2 * (n_ring - 1) * pitch)
        corner = 0  # track the outward hex face
        for i in np.where(self.type == 3)[0]:
            loc = np.where(self.pin_adj == i + 1)
            msg = 'Incorrect number of pin-corner connections'
            assert len(loc[0]) == 1, msg
            p = loc[0][0]
            x0, y0 = pin_xy[p]
            scxy[i, 0] = x0 + (np.cos(self._corner_angle[corner])
                               * dc_corner)
            scxy[i, 1] = y0 + (np.sin(self._corner_angle[corner])
                               * dc_corner)
            corner += 1  # advance the corner index
        return scxy

    def _find_edge_xy(self, scxy, pin_xy, n_ring, pitch, dftf):
        """Determine the X-Y locations of the edge subchannels.

        Parameters
        ----------
        scxy : numpy.ndarray
            Array (N_sc x 2) of subchannel X-Y coordinates
        pin_xy : numpy.ndarray
            Array (Npin x 2) of pin X-Y coordinates
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        pitch : float
            Pin center-to-center pitch distance
        dftf : float
            Flat-to-flat distance for the inner-most assembly
            duct wall

        Returns
        -------
        numpy.ndarray
            Updated array (N_sc x 2) of subchannel X-Y coordinates

        Notes
        -----
        The centroid of the edge subchannel is located halfway between
        the pin center and the inner duct wall. The distance from the
        pin center to the edge subchannel centroid is found by applying
        the Pythagorean theorem using (1) 0.5 * pitch, and; (2) the
        difference between the duct inner flat-to-flat distance and the
        distance between the outermost pins along the flat duct wall.

        The outward normals from the hexagon faces are [np.pi/3, 0.0,
        5*np.pi/3, 4*np.pi/3, np.pi, 2*np.pi/3].

        """
        dy = 0.5 * pitch
        dx = 0.25 * (dftf - _sqrt3 * (n_ring - 1) * pitch)
        d_edge = np.sqrt(dy**2 + dx**2)  # pythagorean thm
        theta = np.arcsin(dy / d_edge)
        face = 0  # track the outward hex face
        # Pin index: take the first pin that matches; this means
        # we're looking "forward" (clockwise). The only exception
        # is the last edge subchannel, which could come from the
        # first pin in the outer ring but instead by this method
        # comes from the last.
        p = np.where(self.pin_adj
                     == self.n_sc['coolant']['interior'] + 1)[0][0]
        for i in range(self.n_sc['coolant']['interior'] + 1,
                       self.n_sc['coolant']['total']):
            x0, y0 = pin_xy[p]
            if self.type[i - 1] == 3:  # corner subchannel
                face += 1  # advance the corner index
                continue
            elif self.type[i - 1] == 2:  # edge subchannel
                dx = np.cos(self._edge_angle[face] - theta) * d_edge
                dy = np.sin(self._edge_angle[face] - theta) * d_edge
                scxy[i - 1, 0] = x0 + dx
                scxy[i - 1, 1] = y0 + dy
                p += 1  # advance to the next pin
            else:
                raise ValueError('Bad subchannel type: this is a bug')
        return scxy

    def _find_duct_bypass_xy(self, scxy, n_ring, pitch, dpin, ftf):
        """Determine X-Y locations of the duct and bypass subchannels.

        Parameters
        ----------
        scxy : numpy.ndarray
            Array (N_sc x 2) of subchannel X-Y coordinates
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        pitch : float
            Pin center-to-center pitch distance
        dpin: float
            Pin diameter
        ftf : list
            List of tuples containing the inner and outer flat-to-flat
            distances for each duct in the assembly.

        Returns
        -------
        numpy.ndarray
            Updated array (N_sc x 2) of subchannel X-Y coordinates

        Notes
        -----
        Duct and bypass subchannels are concentric rings around the
        assembly; all have the same number of subchannels. Each has
        four connections to other subchannels: two to adjacent
        subchannels in the same ring, one each to the subchannels on
        the rings inside and outside. The X-Y coordinates of the
        subchannels on each concentric ring can be obtained by adding
        to the X-Y coordinates of the adjacent subchannel on the
        inner ring.

        """
        # Start from edge/corner coolant subchannel X-Y coords
        # find first instance of outer coolant sc (type == 2 or 3)
        start = np.min(np.concatenate((np.where(self.type == 2)[0],
                                       np.where(self.type == 3)[0])))
        # Find the ID of the first duct subchannel (type==4 or 5)
        type_idx = np.min(np.concatenate((np.where(self.type == 4)[0],
                                          np.where(self.type == 5)[0])))
        # Can define the first duct X-Y coords outside the loop
        de, dc = self._get_ring0_c2c(n_ring, ftf[0], pitch, dpin)
        inner = self._get_ring_xy(scxy[start:], n_ring,
                                  de, dc, type_idx)
        scxy = np.concatenate((scxy, inner))
        type_idx = type_idx + 6 * n_ring  # update for next loops
        for i in range(1, len(ftf)):  # bypass,duct,bypass,duct,...
            # Bypass
            de, dc = self._get_ring_c2c(ftf[i - 1][1] - ftf[i - 1][0],
                                        ftf[i][0] - ftf[i - 1][1])
            temp = self._get_ring_xy(inner, n_ring, de, dc, type_idx)
            scxy = np.concatenate((scxy, temp))
            inner = temp
            type_idx += 6 * n_ring
            # Duct
            de, dc = self._get_ring_c2c(ftf[i][0] - ftf[i - 1][1],
                                        ftf[i][1] - ftf[i][0])
            temp = self._get_ring_xy(inner, n_ring, de, dc, type_idx)
            scxy = np.concatenate((scxy, temp))
            inner = temp
            type_idx += 6 * n_ring
        return scxy

    def _get_ring_xy(self, prev_xy, n_ring, d_edge, d_corner,
                     starting_idx):
        """Get the X-Y coordinates of a duct or bypass ring based on
        the coordinates from the previous ring.

        Parameters
        ----------
        prev_xy : list
            List of tuples with the X-Y coordinates of the previous ring
        n_ring : int
            Number of pin rings (incl. center pin) in the assembly
        d_edge : float
            Distance between the centroids of the edge subchannels of
            the previous and current rings
        d_corner : float
            Distance between the centroids of the corner subchannels of
            the previous and current rings
        starting_idx : int
            Index corresponding to the subchannel ID of the first
            subchannel in the ring.

        Returns
        -------
        list
            List of tuples containing the X-Y coordinates of the
            subchannels in the current ring.

        """
        rxy = np.zeros((6 * n_ring, 2))
        type_idx = starting_idx
        face = 0
        for sci in range(0, 6 * n_ring):
            if self.type[type_idx] in [5, 7]:  # corner-type
                dx = np.cos(self._corner_angle[face]) * d_corner
                dy = np.sin(self._corner_angle[face]) * d_corner
                face += 1
            else:  # edge-type duct or bypass
                dx = np.cos(self._edge_angle[face]) * d_edge
                dy = np.sin(self._edge_angle[face]) * d_edge
            rxy[sci] = [prev_xy[sci, 0] + dx, prev_xy[sci, 1] + dy]
            type_idx += 1
        return rxy

    def _get_ring0_c2c(self, n_ring, ftf, pitch, dpin):
        """Calculate the distance between edge and corner subchannel
        centroids and the centroids of the adjacent duct wall.

        Parameters
        ----------
        n_ring :
            Number of pin rings (incl. center pin) in the assembly
        ftf : tuple
            Inner and outer flat-to-flat distances of the inner-most
            duct wall
        pitch : float
            Pin center-to-center pitch distance
        dpin : float
            Outer diameter of fuel pin cladding

        Returns
        -------
        tuple
            Tuple of floats for the distance between ring edge- and
            corner-type subchannels.

        """
        d_edge = (0.25 *
                  (ftf[0] - _sqrt3 * (n_ring - 1) * pitch
                   + (ftf[1] - ftf[0])))
        d_corner = (0.5 * ftf[1] / _sqrt3 - 0.25 * dpin
                    - 0.5 * (n_ring - 1) * pitch)
        return d_edge, d_corner

    def _get_ring_c2c(self, delta_ftf_rm1, delta_ftf):
        r"""Calculate the distance between the subchannel centroids
        in concentric duct and bypass rings.

        Parameters
        ----------
        delta_ftf_rm1 : float
            Difference in flat-to-flat distance (thickness) of the
            previous ring. For ducts, this is the distance between the
            outer FTF of the previous duct and the inner FTF of the
            current duct; for bypass, this is the distance between the
            inner FTF of the previous duct and the outer FTF of the
            previous duct.
        delta_ftf : float
            Difference in flat-to-flat distance (thickness) of the
            current ring. For ducts, this is the distance between the
            outer and inner FTF of the current duct; for bypass, this
            is the distance between the inner FTF of the next duct and
            the outer FTF of the previous duct.

        Returns
        -------
        tuple
            Tuple of floats for the distance between ring edge- and
            corner-type subchannels.

        Example
        -------
        For duct 2 centroid:

        Coolant |          |          |          |
        edge SC |  Duct 1  | Bypass 1 |  Duct 2  |
                |          |          |          |
            ftf[0][0]  ftf[0][1]  ftf[1][0]  ftf[1][1]
                |          |          |          |
                |          |          |          |
                |     x    |     x <--|---> x    |
                |          |          |          |
                           :          :          :
                           :          :<-------->:
                           :          : delta_ftf = ftf[1][1]- ftf[1][0]
                           :<-------->:
                             delta_ftf_rm1 = ftf[1][0] - ftf[0][1]

        """
        d_edge = 0.25 * (delta_ftf_rm1 + delta_ftf)
        d_corner = _invSqrt3over2 * d_edge
        return d_edge, d_corner
