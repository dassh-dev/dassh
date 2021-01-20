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
date: 2020-02-10
author: Milos Atz
This module defines the PinLattice class to map fuel pins and
identify their neighbors in hexagonal assemblies.
"""
########################################################################
import numpy as np


# Directions turning clockwise around a hexagon
# First entry is step from an inner ring to the top of an outer ring
# The remaining steps are the turns around the hexagon corners
_directions = [(-1, -1), (0, 1), (1, 1), (1, 0),
               (0, -1), (-1, -1), (-1, 0)]

# This list contains the distances corresponding to the directions
# walked in "_directions" for the (a=1) equilateral triangle; these
# are multiplied by the pin center-to-center distance to get the
# actual x-y coordinates of each pin
_sqrt3over2 = 0.866025403784439  # skip some multiplications
_dxdy = [(0, 1), (_sqrt3over2, -0.5), (0, -1), (-_sqrt3over2, -0.5),
         (-_sqrt3over2, 0.5), (0, 1), (_sqrt3over2, 0.5)]


class PinLattice(object):
    """Map the pins in a hexagonal assembly

    Parameters
    ----------
    n_ring : int
        Number of pin rings (including the center pin) in the assembly
    pitch : float
        Fuel pin center-to-center pitch distance
    pin_diameter : float
        Outer diameter of the fuel pin cladding
    origin : tuple (optional)
        x0, y0 of the assembly centroid (default = (0.0, 0.0))
    test: bool (optional)
        If testing, do not run all the instantiation methods; instead,
        allow the object to be instantiated without calling them so
        they can be called incrementally and independently

    Attributes
    ----------
    n_pin : int
        Number of fuel pins in the assembly
    map : numpy.ndarray
        Map of pin positions on ring
    adj : numpy.ndarray
        Pin neighbors for each pin in the assembly
    xy : list of tuples
        Tuples contain the X-Y coordinates of each fuel pin

    """

    def __init__(self, n_ring, pitch, pin_diameter, origin=(0.0, 0.0),
                 test=False):
        """Initialize PinLattice object"""
        self.n_pin = count_pins(n_ring)
        # Make transition matrix
        self.map = self.make_pin_map(n_ring)
        # Define pin-pin adjacency
        self.adj = self.map_pin_neighbors()
        # Get pin X-Y coordinates
        self.xy = self.map_pin_xy(n_ring, pitch, origin)

    @staticmethod
    def make_pin_map(n_ring):
        r"""Map the pin locations in the hexagon.

        Parameters
        ----------
        n_ring : int
            Number of pin rings in the hexagonal assembly; ring=1 is
            the central pin of the assembly.

        Returns
        -------
        numpy.ndarray
            Map of pins in the assembly

        Notes
        -----
        The pins in the assembly are numbered starting at the center
        pin (pin 1) and continuing outward around each of the rings.
        The first pin of the next ring is the pin located on the
        diagonal immediately above the center pin. The pins in each
        ring are labeled by traveling clockwise around the ring.

        A regular hexagon can be divided by three straight lines along
        the long diagonals that pass through the corners and intersect
        at the center. One of these runs straight up and down; the
        second has an angle of 30 degrees from horizontal; the third
        has an angle of 150 degrees from horizontal.

        This pin numbering scheme and division of the hexagon can be
        used to map the pin labels from the hexagon to a square matrix,
        which uses mimics the three long diagonals in the pin with the
        rows, columns, and one diagonal.

        Example
        -------
        If the center pin is labeled "1", then the second ring of pins may
        be labeled:

        NORTH (y)
           \\                EAST (x)
            \\              .#
              2 _____ 3  .#
              /\    / \ #
           7 /___\1/___\ 4
             \   / \   /
              \/_____\/
              6       5

        The map for this 7-pin assembly would be:
                             ____
        | 2 3 0 |           | 2 3  \
        | 7 1 4 |   (note:  | 7 1 4 | looks like a hexagon!)
        | 0 6 5 |            \ _6_5_|

        """
        pin_map = np.zeros((2 * n_ring - 1,
                            2 * n_ring - 1), dtype=int)
        row_c, col_c = n_ring - 1, n_ring - 1  # center pt
        pin_map[row_c, col_c] = 1  # asm has at least one pin (center)
        for ring in range(2, n_ring + 1):  # skipped if only 1 pin
            row, col = row_c - ring + 1, col_c - ring + 1
            pins = np.arange(get_start_pin(ring),
                             get_end_pin(ring) + 1, 1)
            corners = get_corners(ring)
            d = 1  # first direction
            for i in range(0, len(pins)):
                pin_map[row, col] = pins[i]
                if i > 0 and pins[i] in corners:
                    d += 1  # change directions at corner
                row, col = (row + _directions[d][0],
                            col + _directions[d][1])
        return pin_map

    def map_pin_neighbors(self):
        """Identify pin neighbors for each pin in the assembly.

        Returns
        -------
        numpy.ndarray
            Adjacent pin labels for each pin in the assembly

        Notes
        -----
        Array returned has shape (n_pin x 6). Empty positions (along
        the assembly edges) are returned as zeros.

        """
        neighbors = np.zeros((self.n_pin, 6), dtype=int)
        for i in range(1, self.n_pin + 1):
            loc = np.where(self.map == i)  # find pin of interest
            row, col = loc[0][0], loc[1][0]
            # identify pins around it in specific order
            for j in range(0, len(neighbors[i - 1])):
                row, col = (row + _directions[j][0],
                            col + _directions[j][1])
                if row >= 0 and col >= 0:  # avoid wraparound indexing
                    try:
                        neighbors[i - 1, j] = self.map[row, col]
                    except IndexError:  # out of bounds in map: edge pin
                        continue  # skip; come back in-bounds eventually
                else:
                    continue
        return neighbors

    def map_pin_xy(self, n_ring, pin_pitch, origin):
        """Determine the X-Y positions of the fuel pins.

        Create a vector containing the x,y positions of
        the pins in the hexagonal lattice. Runs similarly to
        make_transition_matrix; maybe will consolidate into one

        Parameters
        ----------
        origin : tuple
            X-Y location of the center pin; reference point for all
            other X-Y positions

        Returns
        -------
        numpy.ndarray
            Array (Npin x 2) containing the pin X-Y coordinates

        """
        xy = np.zeros((self.n_pin, 2))
        loc = origin  # default = (0.0, 0.0); x-y location
        xy[0] = loc  # center pin xy location is the origin
        for ring in range(2, n_ring + 1):  # skipped if only 1 pin
            pins = np.arange(get_start_pin(ring),
                             get_end_pin(ring) + 1, 1)
            corners = get_corners(ring)
            d = 0  # first direction: from inner ring to current ring
            for i in range(0, len(pins)):
                loc = (loc[0] + _dxdy[d][0] * pin_pitch,
                       loc[1] + _dxdy[d][1] * pin_pitch)
                xy[pins[i] - 1] = loc
                if pins[i] in corners:
                    d += 1
            # need one more step to return to the top pin of the ring
            loc = (loc[0] + _dxdy[d][0] * pin_pitch,
                   loc[1] + _dxdy[d][1] * pin_pitch)
        return xy


########################################################################
# METHODS TO DESCRIBE THE IDS AND RINGS IN HEXAGONAL PIN BUNDLES
########################################################################


def get_corners(ring):
    """Identify the corner pins of the hexagonal ring.

    Parameters
    ----------
    ring : int
        Pin ring number (ring=1 corresponds to the center pin)

    Returns
    -------
    numpy.ndarray
        Pin numbers corresponding to the corners in the ring

    """
    return np.arange(get_start_pin(ring), get_end_pin(ring) + 1,
                     ring - 1)


def get_end_pin(ring):
    """Identify the final pin in the hexagonal ring.

    Parameters
    ----------
    ring : int
        Pin ring number (ring=1 corresponds to the center pin)

    Returns
    -------
    int
        Pin number for the final pin in the hexagonal ring.

    """
    return get_start_pin(ring + 1) - 1


def get_start_pin(ring):
    """Identify the first pin in the hexagonal ring.

    Parameters
    ----------
    ring : int
        Pin ring number (ring=1 corresponds to the center pin)

    Returns
    -------
    int
        Pin number for the first pin in the hexagonal ring.

    """
    return count_pins(ring - 1) + 1


def count_pins(n_ring):
    """Count the pins in the assembly.

    Parameters
    ----------
    n_ring : int
        Number of pin rings in assembly, where ring=1 is center pin

    Returns
    -------
    int
        Number of pins in the assembly.

    Notes
    -----
    The number of pin rings is equal to the number of pins along
    each flat face of the hexagon. See: http://oeis.org/A003215

    """
    if n_ring == 0:
        return 0
    else:
        return 3 * (n_ring - 1) * n_ring + 1
